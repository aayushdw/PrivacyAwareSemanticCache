<!-- This documentation captures architecture, design decisions, and real-world insights for the federated learning system. -->

# Federated Learning: Privacy-Preserving Collaborative Model Improvement

This directory implements a privacy-aware federated learning (FL) system using [Flower](https://flower.ai/) that enables multiple clients to collaboratively improve a shared embedding model without sharing raw training data. The system uses LoRA (Low-Rank Adaptation) for efficient adaptation and optionally applies Differential Privacy (DP-SGD) for formal privacy guarantees.

In a traditional setup, improving an embedding model would require centralizing all training data which could be a  privacy risk in some cases. This system inverts that: clients train locally on their own data using a shared base model with LoRA adapters, then send only the adapter weights back to a server. The server aggregates these weights and sends the improved global model back to clients. This cycle repeats over multiple rounds, improving the model collaboratively without exposing raw data.

For applications like semantic caching or search that operate across multiple organizations or user bases, this enables model improvement while respecting data privacy and regulatory constraints (GDPR, HIPAA, etc.).

## Architecture & Design

### The Core Loop: A Flower-Based System

**Flower** (an open-source federated learning framework) was chosen because it cleanly separates the server strategy from client training logic, and it handles the messy networking and fault tolerance for us. Here's how a round works:

1. **Server sends**: Current LoRA adapter weights + model config to all clients
2. **Clients train locally**: Train on triplet data (anchor, positive, negative) using triplet margin loss
3. **Clients send back**: Updated LoRA weights
4. **Server aggregates**: Weighted average of client weights (same as FedAvg)
5. **Repeat**: Next round begins

This is implemented in three key components:

#### 1. Server: LoRAFedAvg Strategy (`server/strategy.py`, `server/fl_server.py`)

The custom `LoRAFedAvg` class extends Flower's standard FedAvg to:

- Send model config (base model name, LoRA parameters, DP settings) during `configure_fit()`
- Aggregate using standard weighted averaging but only on LoRA parameters
- Log per-round metrics (number of clients, average loss)
- Save aggregated weights to disk after each round (in `federated_learning/outputs/round_N/`)

#### 2. Client: Local Training (`client/fl_client.py`, `client/local_trainer.py`)

Clients receive server config at the start of training and:

- Load the base model (e.g., `sentence-transformers/all-mpnet-base-v2`) once per training session
- Apply LoRA configuration from the server (ensuring client and server LoRA architectures match)
- Train on local triplet data using triplet margin loss with margin=0.2
- Optionally apply Differential Privacy (DP-SGD) during training
- Return only the LoRA weight updates

We freeze the base model and only train LoRA adapters. This gives us the efficiency (LoRA adapters are tiny—typically <1% of base model size) and the stability (base model is shared across all clients).

#### 3. Weight Manager (`server/weight_manager.py`)

The server maintains a `ServerWeightManager` that:

- Loads initial LoRA weights (from a pre-trained embedding pipeline output or generates fresh)
- Syncs target module names from saved adapters to ensure clients create matching architectures
- Saves aggregated weights in PEFT-compatible format (`adapter_model.safetensors`, `adapter_config.json`)

**Why this design?** The FL system needs to be agnostic about base model type, so target modules can vary (e.g., MPNet uses `["q", "k", "v", "o"]` while BERT uses `["query", "key", "value", "dense"]`). The weight manager reads the saved adapter config and syncs it server-wide, avoiding manual configuration drift.

### Why Frozen LoRA-A and Only Train LoRA-B?

By default, LoRA creates two low-rank matrices: **lora_A** (input projection, shape `d_in × r`) and **lora_B** (output projection, shape `r × d_out`). We have a config option to freeze lora_A and only train lora_B:

```python
freeze_lora_a: bool = True  # Only train lora_B for stability
```

**Why freeze lora_A?**

1. **Numerical stability in DP-SGD**: When using Opacus for differential privacy, training only lora_B simplifies per-sample gradient computation and reduces noise. Freezing lora_A cuts the parameter count in half without sacrificing too much expressiveness.

2. **Convergence**: In practice, empirical evidence shows lora_B captures most of the adaptation signal. lora_A acts as an input bottleneck; freezing it reduces variance in aggregation.

3. **Reduced communication**: Fewer parameters to send back to the server each round.

This is optional: set `freeze_lora_a: False` in the config to train both matrices if you want more expressiveness.

### Design Decision: Differential Privacy with Opacus

The system optionally applies **Differential Privacy via DP-SGD**, implemented using [Opacus](https://opacus.ai/). When enabled:

```python
enable_dp: bool = False  # DP is off by default
dp_epsilon: float = 8.0  # Privacy budget per client per round
dp_delta: float = 1e-5   # Failure probability
dp_max_grad_norm: float = 1.0  # Clipping bound
```

The privacy engine in `LocalLoRATrainer._setup_privacy_engine()` wraps the model, optimizer, and dataloader with Opacus, which:

- Computes per-sample gradients (required for DP-SGD)
- Clips gradients to a max norm
- Adds Gaussian noise to clipped gradients
- Tracks epsilon spent (privacy budget)

We use `grad_sample_mode="functorch"` when DP is enabled. Why?

The default hooks-based approach in Opacus doesn't work when the gradient chain goes through frozen parameters (like frozen lora_A). Functorch computes per-sample gradients by executing the backward pass repeatedly with per-sample inputs—slower but compatible with frozen parameters.

**When you'd use DP**: When clients are untrusted or regulations require formal privacy. Without DP, a curious server could in theory recover information from weight updates (though this is hard in practice). With DP, we have formal guarantees.

**When you'd skip DP**: When all clients are trusted, or latency is critical—Opacus adds ~2-5x overhead to training time.

### Design Decision: Per-Round Model Saving

After each FL round, the server saves aggregated weights to disk:

This lets you:

- Inspect weight evolution across rounds
- Roll back to a previous round if quality degrades
- Monitor training progress without waiting for all rounds to complete

Safetensors were used because it's safer than pickle (no arbitrary code execution), smaller than NPZ, and compatible with HuggingFace's PEFT library.

## Insights

### 1. Triplet Loss

Unlike many embedding papers, we train with triplet loss (anchor, positive, negative), not InfoNCE or contrastive loss:

```python
criterion = nn.TripletMarginLoss(margin=0.2, p=2)
loss = criterion(anchor_emb, positive_emb, negative_emb)
```

Triplet loss directly optimizes for the semantic cache's use case: "questions with the same intent should be close, different intents should be far apart." The margin=0.2 means embeddings must be at least 0.2 L2 distance apart to satisfy the loss. This is tuned from the embedding pipeline evaluation.

#### 2. **No Per-Example Gradient Tracking in Non-DP Mode**

When DP is disabled, we batch all three examples (anchor, positive, negative) together for efficiency:

```python
# Non-DP: combine and encode all at once
combined_ids = torch.cat([anchor_ids, positive_ids, negative_ids], dim=0)
combined_emb = self._encode(combined_ids, combined_mask)
```

This is faster than encoding them separately. But **when DP is enabled**, we have to compute them separately to get per-sample gradients:

```python
# DP-enabled: encode anchor (goes through DP hooks), others in torch.no_grad()
anchor_emb = self._encode(anchor_ids, anchor_mask)  # With gradients
with torch.no_grad():
    positive_emb = self._encode(positive_ids, positive_mask)  # No gradients
```

This is why DP adds overhead because you lose the batching optimization.

#### 3. **Configuration Coupling Between Server and Clients**

The server sends LoRA config in every training round:

```python
# In strategy.py, configure_fit()
config.update({
    "base_model_name": self.base_model_name,
    "lora_r": self.lora_config["lora_r"],
    "target_modules": ",".join(self.lora_config["target_modules"]),
    ...
})
```

Clients parse this and create their LoRA layers to match. **Why not let clients hardcode it?** Because a single server might support multiple models or LoRA configs, and you want clients to adapt dynamically. Also, the server can update config between rounds (e.g., increase `lora_r` if needed).

#### 4. **The Target Modules Sync**

MPNet uses `["q", "k", "v", "o"]` for attention layers, while BERT uses `["query", "key", "value", "dense"]`. We sync these from the saved adapter config:

```python
# In weight_manager.py, _sync_config_from_adapter()
if saved_modules and sorted(saved_modules) != sorted(self.config.target_modules):
    self.config.target_modules = saved_modules
```

Without this sync, a client might try to load a pre-trained MPNet adapter but create LoRA layers with BERT names—incompatible architectures.

### When This Breaks

#### Scaling to Many Clients

The default FL setup expects ~2-10 clients per round. If you try 100+ clients:

- **Client sampling**: Flower samples `fraction_fit * num_available` clients. With default `fraction_fit=1.0`, you wait for all 100 before aggregating. Set `fraction_fit=0.1` to sample 10% per round.
- **Network bandwidth**: Each round, you send model config + weights × 100. With frozen lora_A, this is smaller, but still non-trivial.
- **Synchronous aggregation**: The server waits for all sampled clients. One slow client delays everyone. Consider async aggregation strategies (not currently implemented).

#### Non-Stationary Data (Data Drift)

If client data distributions shift over time (e.g., questions change seasonally), federated learning might diverge. The global model improves on average, but individual clients' local data might diverge from the global objective. **Mitigation**: Regularly evaluate the global model on held-out client test sets. If loss increases, retrain locally or refresh the adapter.

#### Privacy-Utility Tradeoff

With DP-SGD enabled, higher privacy (`lower epsilon`) means more noise, which hurts model quality:

- `epsilon=1.0` (very strong privacy): Large noise, noticeable accuracy drop
- `epsilon=8.0` (reasonable privacy): Small noise, ~5% accuracy drop
- `epsilon=∞` (no privacy): No noise, best accuracy

Choose epsilon based on your privacy requirements. Legal/compliance usually guides this.

#### Heterogeneous Model Architectures

All clients must use the same base model (e.g., all MPNet). If you have some BERT, some GPT clients, you'd need a different strategy. Current design assumes homogeneity.

### Tradeoffs You'll Notice

#### Stateless Clients vs. Stateful

**Current design**: Clients are stateless. Each round, the server sends model config, clients load the base model fresh, initialize LoRA, train, and return weights. Clients don't remember previous rounds.

**Tradeoff**: Simplicity and fault tolerance (a crashed client restarts fresh next round) vs. the overhead of reloading the base model every round.

**Alternative**: Keep models in memory between rounds. Faster, but clients crash = lost work.

We chose stateless because reliability matters more than latency in federated settings.

#### Synchronous Aggregation vs. Asynchronous

**Current design**: The server waits for all sampled clients to finish before aggregating (synchronous). Simple, easy to debug.

**Alternative**: Aggregate from whoever responds first, use a deadline (asynchronous). Faster, but you drop late clients—might bias toward faster machines.

Synchronous is better for federated settings where fairness matters (don't bias toward fast clients).

#### FedAvg vs. Other Aggregation

We use simple weighted averaging (FedAvg). Alternatives are:

- **FedProx**: Adds a regularization term to discourage client drift (helps with non-IID data)
- **FedOpt** (server-side momentum): Accelerates convergence

We chose FedAvg because LoRA is already low-rank (less diverse than full model training), and adding complexity is a premature optimization. We'll consider these alternatives later on.

## Examples

### Running a Simulation

```bash
python -m federated_learning.scripts.simulate_fl \
  --num-clients 3 \
  --num-rounds 5 \
  --lora-path embedding_pipeline/outputs/models/mpnet-base_lora \
  --base-model sentence-transformers/all-mpnet-base-v2 \
  --output-dir federated_learning/outputs
```

This runs 3 clients, 5 rounds locally in a single process. Output is saved to `federated_learning/outputs/round_1/`, `round_2/`, etc.

### With Differential Privacy

```bash
python -m federated_learning.scripts.simulate_fl \
  --num-clients 3 \
  --num-rounds 5 \
  --enable-dp \
  --dp-epsilon 8.0 \
  --dp-delta 1e-5
```

This adds noise to gradients. Expect ~5-10% slowdown due to per-sample gradient computation.

### Actual Server-Client Setup

Start the server:

```bash
python -m federated_learning.scripts.run_server \
  --num-rounds 5 \
  --min-clients 2
```

In separate terminals, start clients:

```bash
python -m federated_learning.scripts.run_client \
  --train-data-path data/client_data/client_0/train.csv

python -m federated_learning.scripts.run_client \
  --train-data-path data/client_data/client_1/train.csv
```

### Observed Behavior

From test runs with 3 clients, 3 rounds on MPNet:

- **Round 1**: Clients train from pre-trained embedding adapter. Typical loss: 0.25-0.35 per client.
- **Round 2-3**: Loss decreases slightly (0.20-0.30) as global model improves. Convergence is slow because each round is just one local epoch per client.
- **With DP (epsilon=8.0)**: Add ~20-30% to loss due to noise, but model is still usable.

**Why slow convergence?** Federated learning trades sample efficiency for privacy. Each client trains on only their local data once per round; you lose the per-epoch feedback loop.

## Integration Points

### With Embedding Pipeline

The embedding pipeline (`embedding_pipeline/`) trains a base LoRA adapter using centralized data. Its output is used as initialization:

```python
initial_lora_path: str = "embedding_pipeline/outputs/models/mpnet-base_lora"
```

FL then improves this adapter collaboratively. The loop is:

1. Embedding pipeline: Fine-tune on synthetic data, save LoRA adapter
2. FL: Distribute this adapter to clients, refine it over multiple rounds
3. Downstream: Use the final FL-improved adapter in production

### With Data Pipeline

Clients need triplet training data. Use the data splitter to partition a central dataset:

```bash
python -m federated_learning.utils.data_splitter \
  --source data/synthetic_data/triplets.csv \
  --num-clients 3 \
  --output-dir data/client_data
```

This creates `data/client_data/client_0/train.csv`, `client_1/train.csv`, etc., ready for FL clients.

## Getting Started

### Setup

```bash
# Ensure base model and initial LoRA adapter exist
python -c "from embedding_pipeline.flows.main_flow import embedding_evaluation_pipeline; embedding_evaluation_pipeline()"

# Split data for FL clients
python -m federated_learning.utils.data_splitter \
  --source data/synthetic_data/triplets.csv \
  --num-clients 2
```

### Run a Simulation

```bash
python -m federated_learning.scripts.simulate_fl \
  --num-clients 2 \
  --num-rounds 3
```

Check output in `federated_learning/outputs/round_1/`, `round_2/`, etc.

### Run Server-Client Mode

Terminal 1:

```bash
python -m federated_learning.scripts.run_server --num-rounds 3 --min-clients 2
```

Terminal 2:

```bash
python -m federated_learning.scripts.run_client \
  --train-data-path data/client_data/client_0/train.csv
```

Terminal 3:

```bash
python -m federated_learning.scripts.run_client \
  --train-data-path data/client_data/client_1/train.csv
```

### Configuration

Edit `federated_learning/config.py` to customize:

- **LoRA rank** (`lora_r`): Lower = smaller adapters, faster training. Default: 16.
- **Learning rate** (`learning_rate`): Client training rate. Default: 2e-4.
- **DP settings**: `enable_dp`, `dp_epsilon`, `dp_delta`.

### Inspecting Results

```bash
# View final aggregated weights
ls federated_learning/outputs/round_3/

# Load in Python
from safetensors.torch import load_file
weights = load_file("federated_learning/outputs/round_3/adapter_model.safetensors")
```

## Next Steps

- **Extend to multi-server FL**: Currently single server. Extend with hierarchical aggregation for large populations.
- **Add client-side evaluation**: Clients could evaluate the global model on their test data and report metrics to server.
- **Implement async aggregation**: Reduce per-round latency when clients have heterogeneous compute.
- **Privacy-utility profiling**: Benchmark accuracy loss vs. epsilon for your use case.
