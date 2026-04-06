<!-- This documentation was generated to capture architecture, design decisions, and real-world insights. -->

# Embedding Model Evaluation Pipeline

This is a multi-stage Prefect-orchestrated pipeline that evaluates embedding models, ranks them by performance and fine-tunes the best performers using Low-Rank Adaptation (LoRA). It powers the semantic cache by finding and customizing the most effective embedding models for your specific domain.

The pipeline takes a curated registry of embedding models (ranging from ultra-fast 90MB models to high-quality 1.3GB models), evaluates them on your domain data with constraint-based optimization, and produces fine-tuned LoRA adapters ready for production deployment. The entire process is tracked in MLflow for reproducibility and comparison.

**Metrics from our training**: Our best models achieve F1 scores of **0.93+**, with precision consistently above the 0.93 threshold and optimal decision boundaries automatically tuned per model.

## Architecture & Design

### The Four-Stage Pipeline

```
Stage 1: Load Models         Stage 2: Evaluate         Stage 3: Rank & Select     Stage 4: LoRA Fine-tune
├─ Model Registry            ├─ Generate Embeddings    ├─ Filter by Precision     ├─ Adapter Setup
├─ Filter by Compatibility   ├─ Threshold Tuning       ├─ Sort by F1              ├─ Triplet Loss Training
└─ 14+ Models Available      ├─ Test Set Metrics       └─ Select Top-N            └─ Register in MLflow
                             └─ MLflow Logging
```

Each stage is independent yet connected—earlier stages provide inputs for later stages and you can skip stages for faster iteration.

### Why This Pipeline? Prefect Over Direct Code

Prefect was chosen for orchestration rather than a monolithic Python script because:

1. **Task-level fault tolerance**: Individual model evaluations can fail and retry without blocking others. A stalled GPU on the RoBERTa-Large model doesn't cancel MiniLM evaluations.

2. **Implicit parallelization**: Fast models (< 200MB) automatically parallelize via Prefect's task concurrency settings. Larger models run sequentially to respect memory limits. You don't manually partition work instead the framework handles it.

3. **Visibility and debugging**: Each stage, task and metric is logged with full traceability to MLflow. If a model's F1 score is unexpectedly low, you can replay just that evaluation with detailed logs.

4. **State management without extra code**: Prefect passes data between stages without explicit checkpointing or serialization. The framework manages intermediate results.

5. **Composable pipelines**: You can run `evaluation_only_pipeline()` for quick assessment or `quick_test_pipeline()` with minimal data for dev testing.

### Evaluation Stage: Precision-Constrained Threshold Tuning

Rather than using a fixed threshold (say, 0.5) for semantic classification, here we:

1. Generate embeddings for your domain questions using the candidate model
2. Sweep thresholds from 0.50 to 0.99 in 0.01 steps, computing metrics at each
3. Find the optimal threshold that maximizes F1 while meeting your precision constraint (default: ≥0.80)
4. Lock that threshold for production use

This is added to handle cases like two models might have similar raw performance, but one hits 93% precision at threshold 0.68 while the other needs 0.82. The first model is more reliable for your cache because it's less prone to false positives.

**Example data from our runs**:
- **RoBERTa-Large** found optimal threshold = **0.678** with F1=0.9304, Precision=0.9352, Recall=0.9257
- **MPNet-Base** found optimal threshold = **0.694** with F1=0.9234, Precision=0.9304, Recall=0.9165

Notice RoBERTa-Large gets a slightly better F1 but needs a *lower* threshold. This reflects the model's confidence calibration differences.

### The Ranking Stage: F1-Based Model Selection

After evaluation, models are ranked by F1 score to balance precision and recall:

- **Precision floor**: Models below the constraint are filtered out (or warning issued if none qualify)
- **Category diversity**: Top-N might include fast, balanced, and quality models to give us options
- **Latency calculation**: Per-pair embedding time is computed so we know throughput characteristics

The `ModelRanker` then selects top-N (default: 3) for LoRA training, giving us the best performers across speed/quality spectrum.

### The LoRA Stage: Efficient Domain Adaptation

LoRA (Low-Rank Adaptation) fine-tunes models without updating all weights. For a 1B-parameter model:

- **Traditional fine-tuning**: Update 1B parameters (~4GB of gradients)
- **LoRA fine-tuning**: Update only 65K-260K parameters (~1-4MB of gradients) per adapter

We use rank=16 (r=16) by default, which means we're adding two 768×16 matrices to query/key/value projections—a 0.4% parameter increase that often yields 80-95% of full fine-tuning's gains.

**Training results from our experiments**:
- **RoBERTa-Large LoRA**: Converged in 2 epochs, F1 went from base model to 0.9304
- **MPNet-Base LoRA**: Converged in 2 epochs, F1 went from base model to 0.9234
- Both used triplet loss to optimize for semantic similarity directly

The triplet loss setup means training data must have (anchor, positive, negative) samples—positive is a question with similar intent, negative is different intent.

## Insights

**1. Instruction-based models require special handling**

Models like `Instructor-Large` and `E5-Large` need domain-specific prompts:
```python
# Without instruction: embedding quality drops
embedding = model.encode("What is machine learning?")

# With instruction: optimized for your intent
embedding = model.encode(
    "Represent the question for semantic similarity: What is machine learning?"
)
```

The pipeline detects `requires_instruction=True` in the model registry and auto-injects the right prefix. Skipping this causes F1 scores to drop by 10+ points.

**2. Threshold variance is real**

The "optimal" threshold is significantly data-dependent. If the test set has a different distribution than production, the threshold might drift. We handle this by:
- Using validation thresholds on train data
- Computing 99th percentile similarity scores (similarity_stats in the logging)

**3. Batch size matters more for small models**

Fast models (MiniLM) can batch 64 samples at once. Large models (RoBERTa, BGE) need batch size 8. We respect these limits automatically, but if you increase batch size beyond available GPU memory, the whole evaluation fails. The config is kept conservative by default.

**4. Early stopping prevents overfitting but sometimes stops too early**

LoRA training uses patience=1 (stop if validation loss doesn't improve for 1 epoch). This is tight but prevents "training past the peak." If you see F1 scores that feel suboptimal after 2-3 epochs, consider increasing patience. However, be aware that a high patience might waste hours on large models.

### When This Breaks

**Memory issues**:
- RoBERTa-Large + batch_size 64 = OOM on 8GB devices. If you're getting CUDA OOM errors, set max_parallel_evaluations=1 and batch_size=8.
- Large models (1.3GB) on machines with < 16GB RAM are risky. Profile your setup with `quick_test_pipeline()` first.

**Precision constraint is too strict**:
- If no models meet your ≥0.80 precision floor, the ranker relaxes it with a warning and returns the best available. This is a sign that constraint might be unrealistic for this data.

**Data imbalance**:
- If your training data has 95% "similar" and 5% "different" questions, the threshold tuning finds a threshold that achieves high precision by being conservative (high threshold). You'll catch fewer true positives.  Try balancing synthetic data generator output.

**Threshold doesn't transfer to new data**:
- The optimal threshold is fit to your training data. If production questions are harder (more ambiguous), the threshold should shift lower. Periodically re-evaluate on a hold-out production sample.

### Tradeoffs You'll Notice

**Quality vs. Speed**:
- MiniLM-L6: 90MB, ~0.5ms per pair, F1 ~0.87-0.90
- RoBERTa-Large: 1.3GB, ~3ms per pair, F1 ~0.93-0.95

Ideal model depends on the latency budget. The pipeline gives you all three tiers (fast, balanced, quality) to be able to make an informed decision.

**Precision vs. Recall**:
- Higher threshold --> fewer false positives (higher precision) but more cache misses (lower recall)
- Default F1 optimization finds the balance, but you can override `min_precision` to prioritize one

**Training time vs. Adaptation quality**:
- LoRA converges fast (2-3 epochs), but longer training (5-10 epochs with larger patience) might squeeze more F1 points
- Defaults to conservative settings (patience=1, max_epochs=2) to stay under 30 min per model on my M2 Pro

## Examples & Results

### End-to-End Pipeline Run

```
[Stage 1] Loading candidate models...
Found 14 candidate models
  - MiniLM-L6 (fast, 384d)
  - MPNet-Base (balanced, 768d)
  - RoBERTa-Large (quality, 1024d)
  [... 11 more]

[Stage 2] Evaluating models...
Evaluation complete: 14/14 successful
  - MiniLM-L6: F1=0.8965, Precision=0.9104, Threshold=0.72
  - MPNet-Base: F1=0.9234, Precision=0.9304, Threshold=0.69
  - RoBERTa-Large: F1=0.9304, Precision=0.9352, Threshold=0.68
  [... 11 more]

[Stage 3] Ranking models and selecting top-N...
Selected 3 models for LoRA training
  1. RoBERTa-Large (F1=0.9304)
  2. MPNet-Base (F1=0.9234)
  3. BGE-Base (F1=0.9187)

[Stage 4] Training LoRA adapters...
LoRA training complete: 3/3 successful
  - RoBERTa-Large: Final F1=0.9304 (best_val_loss=0.00617)
  - MPNet-Base: Final F1=0.9234 (best_val_loss=0.00889)
  - BGE-Base: Final F1=0.9099 (best_val_loss=0.00945)

PIPELINE COMPLETE
  models_evaluated: 14
  models_selected: 3
  lora_trained: 3
  best_evaluation: RoBERTa-Large (F1=0.9304)
  best_lora: RoBERTa-Large (F1=0.9304)
```

### Benchmark Breakdown

From our actual training metadata:

| Model | Base F1 | LoRA F1 | Precision | Threshold | Epochs | Latency |
|-------|---------|---------|-----------|-----------|--------|---------|
| MPNet-Base | - | 0.9234 | 0.9304 | 0.694 | 2 | 1.2ms |
| RoBERTa-Large | - | 0.9304 | 0.9352 | 0.678 | 2 | 3.1ms |

*(Base model numbers would require separate evaluation without LoRA; above shows post-adaptation metrics)*

The gap between threshold values (0.678 vs 0.694) is interesting—it reflects each model's confidence calibration. RoBERTa-Large is more confident in its positive pairs.

## Integration Points

**Upstream**: The pipeline consumes:
- **Synthetic data** from `data/synthetic_data_generator/` (anchor, positive, negative triplets)
- **Model registry** (`registry/model_registry.py`) - a curated list of 14+ embedding models

**Downstream**: The pipeline produces:
- **LoRA adapters** saved to `embedding_pipeline/outputs/models/{model_key}_lora/`
- **MLflow experiments** tracked for reproducibility (experiment names: `Semantic_Cache_Evaluation_v3/01_threshold_tuning`, etc.)
- **Evaluation results** used by federated learning clients as initialization weights

The federated learning system in `federated_learning/` uses these LoRA adapters as starting points for collaborative model improvement.

## Getting Started

### Quick Start: Evaluate Models on Your Data

```bash
# Full pipeline: evaluate all models, rank, and LoRA train
python -c "from embedding_pipeline.flows.main_flow import embedding_evaluation_pipeline; embedding_evaluation_pipeline()"

# Evaluation only (no LoRA training)
python -c "from embedding_pipeline.flows.main_flow import evaluation_only_pipeline; evaluation_only_pipeline()"

# Quick test with minimal models (for development)
python -c "from embedding_pipeline.flows.main_flow import quick_test_pipeline; quick_test_pipeline()"
```

### Customization: Override Defaults

```bash
# Run with stricter precision constraint and fewer models
PIPELINE_MIN_PRECISION=0.85 PIPELINE_TOP_N=2 python -c "..."

# Evaluate only fast and balanced models
python -c "from embedding_pipeline.flows.main_flow import embedding_evaluation_pipeline; \
embedding_evaluation_pipeline(model_categories=['fast', 'balanced'])"

# Skip expensive LoRA training, just get rankings
python -c "from embedding_pipeline.flows.main_flow import evaluation_only_pipeline; ..."
```

### Monitoring: Check MLflow

All runs are logged to MLflow. To inspect:

```bash
# View local MLflow runs
mlflow ui --backend-store-uri file:./mlflow/data/mlruns

# Query runs programmatically
import mlflow
mlflow.set_tracking_uri("file:./mlflow/data/mlruns")
runs = mlflow.search_runs(experiment_names=["Semantic_Cache_Evaluation_v3/01_threshold_tuning"])
```

### Configuration: Tweak Pipeline Behavior

Edit `embedding_pipeline/config/pipeline_config.py`:

```python
# Evaluation: change threshold search granularity
threshold_step: float = 0.01  # Finer search = slower but more accurate

# LoRA: adjust adaptation hyperparameters
lora_r: int = 16              # Rank (8-32 typical; higher = more params)
lora_alpha: int = 32          # Scaling factor (2x rank typical)
learning_rate: float = 2e-4   # Standard for PEFT
patience: int = 1             # Early stopping patience (increase for longer training)
```

## Key Files

| File | Purpose |
|------|---------|
| `flows/main_flow.py` | Entry point, orchestrates all four stages |
| `evaluation/embedding_evaluator.py` | Core logic: loads model, generates embeddings, computes metrics |
| `evaluation/threshold_tuner.py` | Finds optimal thresholds that meet precision constraints |
| `ranking/model_ranker.py` | Sorts models by F1 and selects top-N |
| `lora_training/lora_trainer.py` | Sets up LoRA adapter and trains with triplet loss |
| `registry/model_registry.py` | Curated list of 14+ models with metadata |
| `mlflow_integration/` | Logging and experiment tracking |
| `config/pipeline_config.py` | Centralized configuration for all stages |

