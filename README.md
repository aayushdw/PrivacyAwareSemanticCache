# Privacy-Aware Semantic Cache

A semantic cache for LLM queries that lets organizations share improvements to the embedding model without sharing the queries themselves. The stack combines a LoRA-based embedding fine-tuning pipeline, a Flower federated learning server with optional differential privacy, and a synthetic data generator that produces the hard-negative triplets the whole thing trains on.

This repo is the result of trying to answer a fairly concrete question: *can you run a useful semantic cache across mutually-distrustful clients without any of them sending raw queries to a central server?* The short answer turned out to be yes, if you are willing to accept that most of the engineering effort goes into the boring parts: data generation, threshold calibration, and making sure the LoRA shapes line up between the embedding pipeline and the FL clients.

---

## Why this project exists

Semantic caches for LLMs are a good idea in theory and a minefield in practice. The obvious approach is to embed every incoming query, look it up in a vector store, and return the cached response if the cosine similarity to some previous query clears a threshold. That works until you try to tune the threshold on real data, at which point two things become uncomfortable:

1. **A fixed threshold is almost never the right choice.** Different embedding models have different confidence calibration. RoBERTa-Large in this repo lands on an optimal threshold of 0.678, while MPNet-Base prefers 0.694 on the same data. Picking 0.75 because it "sounds reasonable" loses you several points of F1 on both.
2. **A naive cache leaks information.** The embedding model itself is trained on query distributions. If you centralize those queries to fine-tune the model, you have recreated the privacy problem you were trying to avoid.

The architecture here is built around those two observations. The embedding pipeline does precision-constrained threshold tuning per model so every cache decision has a calibrated decision boundary. The federated learning component lets clients contribute to the embedding model's fine-tuning by exchanging only LoRA adapter weights, with an optional DP-SGD layer on top for formal guarantees. The synthetic data generator exists because the training triplets you need to do any of this (anchor, similar-intent positive, different-intent hard negative) do not exist off the shelf for most domains.

## Key components

The project is structured as three subsystems that each do one job and hand off to the next. Each has its own README with the deep dive; this document is the view from above.

```
data/synthetic_data_generator/   --->   embedding_pipeline/   --->   federated_learning/
       (triplet generation)           (eval + LoRA fine-tune)        (collaborative training)
```

### 1. Synthetic Data Generator

A LangGraph-orchestrated async pipeline that takes seed questions and produces labeled triplets using Google Gemini models. For each seed question it makes two parallel Gemini calls: one asking for a semantically equivalent rephrasing (positive) and one asking for a question that shares keywords but has a different intent (the hard negative). That second call is the whole point. Without hard negatives, the downstream embedding model learns keyword overlap instead of semantic intent, and the threshold tuner has nothing interesting to do.

The concurrency model is deliberately layered. The LangGraph orchestrator runs sequentially through load -> distribute -> run_workers -> aggregate. Inside the `run_workers` stage, multiple `ModelWorker` instances run on `asyncio` in parallel, each with its own clock-based rate limiter. Each worker then fans out two API calls per question. The reason for three separate concurrency levels is that each one is solving a different problem: the orchestrator is about reproducibility and debuggability, the workers are about throughput, and the fan-out is about respecting per-request rate limits without a global lock. Trying to merge these into a single async soup was the first thing I tried, and the rate-limiter contention made the whole thing slower than the naive sequential version.

Round-robin distribution across Gemini model variants (`gemma-3-1b`, `gemma-3-12b`, `gemma-3-27b`) is a concession to simplicity. A work-stealing queue would balance load better if the model latencies diverged, but for this workload the difference is small enough that the extra complexity wasn't worth it.

See [data/synthetic_data_generator/README.md](data/synthetic_data_generator/README.md) for the full architecture including sequence diagrams, the throttling formula, and a list of the failure modes I actually hit.

### 2. Embedding Pipeline

A four-stage Prefect pipeline that loads candidate embedding models from a curated registry, evaluates each with precision-constrained threshold tuning, ranks them by F1, and fine-tunes the top performers with LoRA. MLflow tracks everything for reproducibility.

The registry currently holds 14+ sentence-transformer models spanning three tiers (fast, balanced, quality), from MiniLM-L6 at 90MB up to RoBERTa-Large at 1.3GB. Each registry entry carries enough metadata (model size, embedding dim, whether it needs instruction prompts) that the pipeline can make sensible decisions about batch size and whether to inject a `"Represent this question for semantic similarity:"` prefix. That last point matters more than it looks: forgetting the instruction prefix on an Instructor-style model drops F1 by about ten points. The registry is where that kind of model-specific quirk lives so the rest of the pipeline can stay generic.

The evaluation stage is the part I spent the most time on. Instead of picking a fixed cosine-similarity threshold, it sweeps thresholds from 0.50 to 0.99 in 0.01 increments, computes precision/recall/F1 at each, and selects the threshold that maximizes F1 subject to a precision floor (default 0.80). The precision floor is non-negotiable because a false cache hit returns a wrong answer, which is much worse for a semantic cache than a miss. A fixed threshold would let a model with strong recall but weak precision win the ranking; the constraint makes sure that can't happen.

Actual numbers from the training runs in this repo:

| Model | Optimal Threshold | F1 | Precision | Recall | Latency |
|---|---|---|---|---|---|
| MPNet-Base (LoRA) | 0.694 | 0.9234 | 0.9304 | 0.9165 | ~1.2 ms/pair |
| RoBERTa-Large (LoRA) | 0.678 | 0.9304 | 0.9352 | 0.9257 | ~3.1 ms/pair |
| MiniLM-L6 (baseline) | 0.720 | 0.8965 | 0.9104 | ~ | ~0.5 ms/pair |

The gap in optimal thresholds (0.678 vs 0.694) between the two larger models is the kind of detail you only notice if you tune per-model. It reflects the fact that RoBERTa-Large is slightly more confident in its positive pairs, so the same F1 peak sits at a lower threshold. A one-size-fits-all threshold would have penalized whichever model's calibration it didn't match.

Fine-tuning uses LoRA with rank 16, applied to the attention query/key/value projections, trained with triplet margin loss (margin 0.2). LoRA instead of full fine-tuning for the usual reasons: a 1B-parameter model becomes a ~260K-parameter adapter, so gradients are 1-4 MB instead of 4 GB, and the adapter is small enough to ship over the FL protocol every round. Training converges in two epochs under the default patience=1 early-stopping setting, which is tight but prevents training past the peak on a dataset this size.

The reason Prefect sits on top of all of this rather than a plain Python script is task-level fault tolerance. A stalled GPU on the RoBERTa evaluation does not cancel the MiniLM evaluations running in parallel. Each stage can be run independently (`evaluation_only_pipeline()`, `quick_test_pipeline()`) for faster iteration during development. And the MLflow integration gives you replay on any individual task without re-running the whole pipeline.

See [embedding_pipeline/README.md](embedding_pipeline/README.md) for the stage-by-stage breakdown, the full benchmark table, and the four failure modes (memory, precision infeasibility, data imbalance, threshold drift) with recovery steps.

### 3. Federated Learning

A Flower-based system where multiple clients collaboratively improve a shared embedding model by exchanging LoRA adapter weights. The server runs a custom `LoRAFedAvg` strategy that extends Flower's FedAvg to aggregate only the LoRA parameters and to ship the model config (base model name, LoRA rank, target modules, DP settings) as part of `configure_fit` so clients can construct their adapter architecture to match the server's. Aggregated weights get saved after each round in PEFT-compatible `adapter_model.safetensors` format so you can inspect, roll back, or hot-swap without waiting for training to finish.

A few design decisions that are worth calling out because they are not obvious from the code:

**Only LoRA-B is trained; LoRA-A is frozen by default.** LoRA decomposes the adapter into two matrices: `lora_A` (input projection, `d_in x r`) and `lora_B` (output projection, `r x d_out`). Freezing `lora_A` cuts the trainable parameter count in half and, more importantly, makes DP-SGD well-behaved. Opacus's default hook-based per-sample gradient path breaks when the gradient chain passes through a frozen parameter, so DP mode switches to `grad_sample_mode="functorch"`, which runs the backward pass per-sample. It is slower but it actually works with the frozen branch. Empirically `lora_B` captures most of the adaptation signal on this dataset, so the expressiveness loss is small. There is a config flag (`freeze_lora_a=False`) to turn this off if your data is hungrier.

**Differential privacy is optional, via Opacus DP-SGD.** The three knobs are epsilon (privacy budget), delta (failure probability), and max grad norm (clipping bound). Defaults are `epsilon=8.0`, `delta=1e-5`, `max_grad_norm=1.0`, which is a reasonable-privacy / mild-accuracy-hit regime (roughly 5% quality drop in the runs I've done). The `epsilon=1.0` regime is possible but the noise starts to bite; the `epsilon=infinity` regime is just standard FL without privacy. Which one you want is a legal question, not a technical one.

**Clients are stateless across rounds.** Every round, the client reloads the base model from disk, applies the server's LoRA config, trains for one local epoch, returns the updates, and forgets everything. This is slower per round than keeping the model in memory, but it means a crashed client just rejoins on the next round with no recovery logic. In a federated setting where some clients are going to be flaky, the reliability tradeoff is almost always worth it.

**Aggregation is synchronous and uses plain FedAvg.** I considered FedProx and FedOpt. FedProx's proximal term helps with non-IID data, and FedOpt adds server-side momentum, but both add complexity that is hard to justify when you're only training a low-rank adapter. LoRA is constrained enough that client drift is naturally limited, so FedAvg behaves well in practice. If the data becomes much more non-IID, FedProx is the first thing I'd add.

**Target module names are synced from the saved adapter config, not hardcoded.** MPNet uses `["q", "k", "v", "o"]` for attention projections while BERT uses `["query", "key", "value", "dense"]`. The server's weight manager reads the target modules from whatever adapter it loaded and propagates them to clients via `configure_fit`. Without this sync, a client trying to load an MPNet adapter into a LoRA configured with BERT module names silently constructs the wrong architecture and produces garbage updates.

One subtlety in the training loop: with DP disabled, the anchor/positive/negative embeddings are concatenated into a single forward pass for efficiency. With DP enabled, that doesn't work because Opacus needs per-sample gradients, so the anchor is encoded with gradients and the positive/negative pair are encoded inside `torch.no_grad()`. This is why enabling DP costs roughly 2-5x in training time, and it's a fundamental consequence of per-sample gradient computation, not something that can be optimized away.

See [federated_learning/README.md](federated_learning/README.md) for the round-by-round protocol, the full DP setup, simulation scripts, and the three places where this design breaks (scaling past ~100 clients, non-stationary data, heterogeneous base models).

## How the pieces fit together

The data flow is linear but the coupling between components is worth understanding because it is where most of the bugs lived during development:

1. The **synthetic data generator** produces a CSV of triplets. Its only coupling to the rest of the system is the schema (anchor, positive, negative columns) and the assumption that the negatives are genuinely hard, i.e. share keywords with the anchor.
2. The **embedding pipeline** consumes those triplets, evaluates candidate models, and fine-tunes the top performers with LoRA. It outputs a PEFT adapter directory (`embedding_pipeline/outputs/models/{model_key}_lora/`) plus an MLflow-tracked threshold.
3. The **federated learning server** bootstraps from that LoRA adapter. It reads the adapter's `adapter_config.json` to pick up the target module names, then distributes the weights and config to clients, which continue training on their local data.

The tight coupling is in the LoRA configuration. The `lora_r`, `lora_alpha`, and target modules set by the embedding pipeline become the initial state for FL, and changing them between stages means the server can't load the adapter. The weight manager handles this by treating the saved adapter as the source of truth and overriding its own config to match. The first time I ran end-to-end I had a mismatch here and spent a couple of hours on what turned out to be a single config key.

MLflow tracks runs from both the embedding pipeline and the FL server under separate experiment names, which gives you cross-stage lineage for free: you can trace an FL round back to the embedding adapter it started from, back to the evaluation run that selected that model, back to the synthetic dataset it was evaluated on.

## Stack

- **Python 3.12** with a venv at `.venv/`
- **Google Gemini API** (`google-generativeai`) for synthetic data generation
- **LangGraph** for the data generator's pipeline orchestration
- **Prefect** for the embedding pipeline's stage orchestration
- **MLflow** for experiment tracking across both pipelines
- **sentence-transformers** for the base embedding models
- **PEFT** (HuggingFace) for LoRA adapter management
- **Flower** (`flwr`) for federated learning
- **Opacus** for DP-SGD
- **PyTorch** as the ML framework throughout
- **DVC** for dataset versioning
- **ChromaDB** for the cache's vector store

## Running the pipeline end-to-end

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Set your Gemini API key
export GEMINI_API_KEY="your-key-here"

# 3. Generate synthetic triplets
python -m data.synthetic_data_generator.main

# 4. Evaluate and fine-tune embedding models
python -c "from embedding_pipeline.flows.main_flow import embedding_evaluation_pipeline; embedding_evaluation_pipeline()"

# 5. Split triplets for FL clients
python -m federated_learning.utils.data_splitter \
  --source data/synthetic_data/triplets.csv \
  --num-clients 2

# 6. Run an FL simulation against the fine-tuned adapter
python -m federated_learning.scripts.simulate_fl \
  --num-clients 2 \
  --num-rounds 5 \
  --lora-path embedding_pipeline/outputs/models/mpnet-base_lora
```

To run the FL server and clients as separate processes instead of a simulation, use `run_server` and `run_client` from `federated_learning/scripts/` (see the FL README for the full invocation).

## Things I'd do differently next time

A few of the decisions in this repo are ones I'd revisit if I were starting over:

**Client sampling.** The default `fraction_fit=1.0` means the server waits for every sampled client before aggregating. That's fine for a 3-client simulation but falls over at 100 clients. The fix is easy (drop `fraction_fit` to ~0.1) but I'd make the default less pathological.

**Streaming in the data generator.** The generator loads every seed question into memory, which is a non-issue for the 10K-ish question datasets I've tested with but would OOM on 1M. A streaming iterator would be the right fix; I went with the simpler version because the workload didn't demand it.

**Async FL aggregation.** The synchronous aggregation is the right default for fairness, but adding an optional async path for heterogeneous-compute scenarios is something I'd want before deploying this for real.

**Client-side evaluation.** The server currently doesn't get any feedback from clients beyond loss. If clients evaluated the global model on their local test splits and reported metrics, the server could detect when aggregation is hurting a subset of clients and trigger FedProx-style regularization.

None of these are blockers for the research/prototype work this repo is aimed at, but they are the obvious next steps if this were going into production.

## Further reading

- [embedding_pipeline/README.md](embedding_pipeline/README.md) - the full four-stage pipeline, including the failure modes I hit and the tradeoff table
- [federated_learning/README.md](federated_learning/README.md) - the FL protocol, DP-SGD setup, and LoRA-A freezing rationale in detail
- [data/synthetic_data_generator/README.md](data/synthetic_data_generator/README.md) - the concurrency model, throttling formula, and parser failure modes
- [embedding_pipeline/docs/RESULTS.md](embedding_pipeline/docs/RESULTS.md) - raw evaluation and fine-tuning numbers
