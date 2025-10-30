# Threshold Tuning Guide

Complete guide to tuning similarity thresholds for embedding models in the Privacy Aware Semantic Cache framework.

---

## Table of Contents

1. [Why Tune Thresholds?](#why-tune-thresholds)
2. [The Tuning Process](#the-tuning-process)
3. [Quick Start](#quick-start)
4. [Optimization Metrics](#optimization-metrics)
5. [Sample Size Selection](#sample-size-selection)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)
8. [Understanding Results](#understanding-results)
9. [Advanced Workflows](#advanced-workflows)
10. [Troubleshooting](#troubleshooting)

---

## Why Tune Thresholds?

### The Problem

When using embedding models for semantic similarity, we compute cosine similarity scores between embeddings. To classify whether two questions are duplicates, we need a **threshold**:

```python
similarity_score = cosine_similarity(embedding1, embedding2)
is_duplicate = similarity_score >= threshold  # But what threshold?
```

### Why Not Use a Fixed Threshold?

Different models have **different similarity distributions**:

| Model | Mean Sim (Duplicates) | Mean Sim (Non-Duplicates) | Optimal Threshold |
|-------|----------------------|---------------------------|-------------------|
| MiniLM-L6 | 0.78 | 0.58 | 0.82 |
| MPNet-Base | 0.83 | 0.62 | 0.85 |
| GTE-Large | 0.87 | 0.65 | 0.89 |

**Using tuned thresholds** maximizes each model's performance.

---

## The Tuning Process

### Overview

The threshold tuner follows this process:

```
1. Load Training Data
   ├─> Load questions_train.csv
   └─> Optionally sample N examples

2. Generate Embeddings
   ├─> Encode all question1 texts
   └─> Encode all question2 texts

3. Compute Similarities
   └─> Calculate cosine similarity for each pair

4. Search for Optimal Threshold
   ├─> Try thresholds from 0.50 to 0.99 (step: 0.01)
   ├─> For each threshold, compute metrics
   └─> Select threshold that maximizes chosen metric

5. Report Results
   ├─> Optimal threshold
   ├─> Metrics at optimal threshold
   └─> Similarity distribution statistics
```

### Algorithm Details

**Threshold Search:**
```python
thresholds = [0.50, 0.51, 0.52, ..., 0.98, 0.99]

for threshold in thresholds:
    predictions = similarities >= threshold
    accuracy = compute_accuracy(predictions, ground_truth)
    precision = compute_precision(predictions, ground_truth)
    recall = compute_recall(predictions, ground_truth)
    f1 = compute_f1(predictions, ground_truth)

    if f1 > best_f1:  # Or other metric
        best_threshold = threshold
        best_metrics = {accuracy, precision, recall, f1}
```

**Key Features:**
- ✅ Exhaustive search (49 thresholds tested)
- ✅ Reproducible (fixed random seed for sampling)
- ✅ Efficient (embeddings computed once per model)
- ✅ Comprehensive (returns full metrics at optimal point)

---

## Quick Start

### Command Line

```bash
# Tune default models (minilm-l6, mpnet-base, gte-base, bge-base)
python3 -m embeddings.eval.threshold_tuner

# Run tuning examples
python3 -m embeddings.eval.tune_thresholds_example
```

### Python API

```python
from embeddings import tune_default_models

# Tune default model set with 5000 training samples
tuner = tune_default_models(max_samples=5000, metric='f1')

# Get optimal thresholds
thresholds = tuner.get_optimal_thresholds()
print(thresholds)
# {'minilm-l6': 0.82, 'mpnet-base': 0.85, 'gte-base': 0.87, 'bge-base': 0.86}
```

---

## Optimization Metrics

Choose the metric based on your use case:

### F1 Score (Default) ⭐ **RECOMMENDED**

**Use when:** You want balanced performance

```python
tuner.tune_models(models, metric='f1')
```

- **Pros:** Balances precision and recall
- **Cons:** May not suit extreme class imbalance
- **Best for:** General-purpose semantic caching

**Formula:** `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

---

### Precision

**Use when:** False positives are costly (showing wrong cached results)

```python
tuner.tune_models(models, metric='precision')
```

- **Pros:** Minimizes false positives
- **Cons:** May miss some valid duplicates
- **Best for:** High-accuracy requirements, user-facing applications

**Formula:** `Precision = True Positives / (True Positives + False Positives)`

**Effect:** Higher thresholds (more conservative matching)

---

### Recall

**Use when:** False negatives are costly (missing cache hits)

```python
tuner.tune_models(models, metric='recall')
```

- **Pros:** Minimizes false negatives
- **Cons:** May produce more false positives
- **Best for:** Maximizing cache hit rate, internal systems

**Formula:** `Recall = True Positives / (True Positives + False Negatives)`

**Effect:** Lower thresholds (more aggressive matching)

---

### Accuracy

**Use when:** Classes are balanced and all errors are equal

```python
tuner.tune_models(models, metric='accuracy')
```

- **Pros:** Simple, intuitive
- **Cons:** Misleading with class imbalance
- **Best for:** Balanced datasets only

**Formula:** `Accuracy = (True Positives + True Negatives) / Total`

---

### Comparison Example

```python
from embeddings import ThresholdTuner

models = ['mpnet-base']

# F1-optimized (balanced)
tuner_f1 = ThresholdTuner(max_samples=5000)
tuner_f1.tune_models(models, metric='f1')
# Result: threshold=0.850, F1=0.7842, Precision=0.8456, Recall=0.7234

# Precision-optimized (conservative)
tuner_prec = ThresholdTuner(max_samples=5000)
tuner_prec.tune_models(models, metric='precision')
# Result: threshold=0.920, F1=0.7234, Precision=0.9012, Recall=0.6234

# Recall-optimized (aggressive)
tuner_rec = ThresholdTuner(max_samples=5000)
tuner_rec.tune_models(models, metric='recall')
# Result: threshold=0.780, F1=0.7456, Precision=0.7123, Recall=0.8456
```

**Observation:** Higher optimization metric → adjusted threshold in expected direction

---

## Sample Size Selection

### Trade-offs

| Sample Size | Speed | Stability | Use Case |
|------------|-------|-----------|----------|
| 1,000 | ⚡⚡⚡ Fast | ⚠️ Variable | Quick experiments |
| 3,000 | ⚡⚡ Moderate | ✅ Good | Development |
| 5,000 | ⚡ Slower | ✅✅ Very Good | **Recommended** |
| 10,000+ | 🐢 Slow | ✅✅✅ Excellent | Production/Research |

### Recommendations

**Quick Prototyping** (1-2 models):
```python
tuner = ThresholdTuner(max_samples=1000)
```

**Development** (4-6 models):
```python
tuner = ThresholdTuner(max_samples=3000)
```

**Production** (final tuning):
```python
tuner = ThresholdTuner(max_samples=10000)
```

**Research** (comprehensive analysis):
```python
# Use entire training set
df = load_train_dataset()
tuner = ThresholdTuner(max_samples=len(df))
```

### Stability Analysis

```python
# Test threshold stability across sample sizes
from embeddings import ThresholdTuner

sample_sizes = [1000, 3000, 5000, 10000]
results = {}

for size in sample_sizes:
    tuner = ThresholdTuner(max_samples=size)
    tuner.tune_models(['mpnet-base'], metric='f1')
    results[size] = tuner.get_optimal_thresholds()['mpnet-base']

# Typical results:
# 1000:  0.82 (±0.03 variance)
# 3000:  0.85 (±0.01 variance)
# 5000:  0.85 (stable)
# 10000: 0.85 (very stable)
```

**Conclusion:** 5000+ samples provide stable, reliable thresholds.

---

## Usage Examples

### Example 1: Tune Default Models

```python
from embeddings import tune_default_models

# Tune 4 representative models
tuner = tune_default_models(max_samples=5000, metric='f1')

# View results
thresholds = tuner.get_optimal_thresholds()
for model, threshold in thresholds.items():
    print(f"{model}: {threshold:.3f}")

# Results saved to: embeddings/eval/results/threshold_tuning_default.json
```

### Example 2: Tune Specific Models

```python
from embeddings import tune_specific_models

# Tune models you care about
models = ['mpnet-base', 'gte-large', 'bge-large', 'instructor-xl']

tuner = tune_specific_models(
    model_keys=models,
    max_samples=5000,
    metric='f1'
)
```

### Example 3: Tune All 25 Models

```python
from embeddings import tune_all_models

# Use fewer samples for speed (3000 instead of 5000)
tuner = tune_all_models(max_samples=3000, metric='f1')

# Results saved to: embeddings/eval/results/threshold_tuning_all_models.json
```

### Example 4: Tune by Category

```python
from embeddings import ThresholdTuner, get_models_by_category

# Get all quality models
quality_models = list(get_models_by_category('quality').keys())

# Tune them
tuner = ThresholdTuner(max_samples=5000)
tuner.tune_models(quality_models, metric='f1')
```

### Example 5: Custom Workflow

```python
from embeddings import ThresholdTuner, load_train_dataset

# Load and prepare custom data
df = load_train_dataset()

# Sample specific subset (e.g., recent data, specific domain)
df_recent = df[df['id'] > 100000].sample(n=5000, random_state=42)

# Tune on custom subset
tuner = ThresholdTuner(max_samples=5000)
result = tuner.tune_model('mpnet-base', df_recent, metric='f1')

print(f"Optimal threshold: {result['optimal_threshold']:.3f}")
```

### Example 6: Tune → Validate → Test

**Complete ML workflow:**

```python
from embeddings import (
    ThresholdTuner,
    SimilarityEvaluator,
    EmbeddingEngine,
    load_train_dataset,
    load_val_dataset,
    load_test_dataset,
    get_model_info
)

model_key = 'mpnet-base'

# STEP 1: Tune on training data
print("Step 1: Tuning on training data...")
train_df = load_train_dataset()
tuner = ThresholdTuner(max_samples=5000)
result = tuner.tune_model(model_key, train_df.head(5000), metric='f1')
optimal_threshold = result['optimal_threshold']

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Training F1: {result['metrics_at_optimal']['f1_score']:.4f}")

# STEP 2: Validate on validation data
print("\nStep 2: Validating on validation data...")
val_df = load_val_dataset()
model_info = get_model_info(model_key)
engine = EmbeddingEngine(model_name=model_info.model_id)
evaluator = SimilarityEvaluator(engine, similarity_threshold=optimal_threshold)

val_metrics = evaluator.evaluate_dataset(val_df, max_samples=1000)
print(f"Validation F1: {val_metrics['f1_score']:.4f}")

# STEP 3: Test on test data (final evaluation)
print("\nStep 3: Testing on test data...")
test_df = load_test_dataset()
test_metrics = evaluator.evaluate_dataset(test_df, max_samples=1000)
evaluator.print_evaluation_report(test_metrics)

print(f"\nFinal Test F1: {test_metrics['f1_score']:.4f}")
```

---

## Best Practices

### 1. Always Use Training Data

✅ **Correct:**
```python
train_df = load_train_dataset()
tuner.tune_model('mpnet-base', train_df, metric='f1')
```

❌ **Wrong:**
```python
test_df = load_test_dataset()  # Never tune on test data!
tuner.tune_model('mpnet-base', test_df, metric='f1')
```

**Why:** Tuning on test data leads to overfitting and inflated performance estimates.

---

### 2. Use Sufficient Samples

✅ **Recommended:** 5,000+ samples
⚠️ **Acceptable:** 3,000 samples (development)
❌ **Too Few:** <1,000 samples (unstable)

---

### 3. Match Training and Deployment

**Tune on data that matches your use case:**

```python
# If deploying on technical questions
df_technical = df[df['domain'] == 'technical']
tuner.tune_model('mpnet-base', df_technical, metric='f1')

# If cache performance matters most
tuner.tune_model('mpnet-base', df, metric='recall')

# If accuracy matters most
tuner.tune_model('mpnet-base', df, metric='precision')
```

---

### 4. Document Your Thresholds

**Save and version your thresholds:**

```python
# After tuning
tuner.save_results('thresholds_v1.0_2024-01-15.json')

# Load for deployment
import json
with open('eval/results/thresholds_v1.0_2024-01-15.json') as f:
    thresholds = json.load(f)

optimal_threshold = thresholds[0]['optimal_threshold']
```

---

## Understanding Results

### Output Structure

```python
{
  "model_key": "mpnet-base",
  "model_name": "MPNet-Base",
  "optimal_threshold": 0.850,
  "metrics_at_optimal": {
    "threshold": 0.850,
    "accuracy": 0.7920,
    "precision": 0.8456,
    "recall": 0.7234,
    "f1_score": 0.7842
  },
  "similarity_stats": {
    "duplicates": {
      "mean": 0.832,
      "std": 0.124,
      "min": 0.412,
      "max": 0.987
    },
    "non_duplicates": {
      "mean": 0.623,
      "std": 0.187,
      "min": 0.234,
      "max": 0.912
    }
  },
  "tuning_samples": 5000,
  "optimization_metric": "f1"
}
```

### Interpreting Metrics

**Optimal Threshold: 0.850**
- Questions with similarity ≥ 0.850 classified as duplicates
- Questions with similarity < 0.850 classified as non-duplicates

**Accuracy: 0.7920**
- 79.2% of all predictions are correct

**Precision: 0.8456**
- 84.56% of predicted duplicates are actually duplicates
- False positive rate: 15.44%

**Recall: 0.7234**
- 72.34% of actual duplicates are detected
- Miss rate: 27.66%

**F1 Score: 0.7842**
- Harmonic mean of precision and recall
- Balanced measure of performance

### Similarity Distribution

**Duplicates:**
- Mean: 0.832 (most duplicates have high similarity)
- Std: 0.124 (moderate spread)
- Range: [0.412, 0.987]

**Non-Duplicates:**
- Mean: 0.623 (most non-duplicates have lower similarity)
- Std: 0.187 (wider spread)
- Range: [0.234, 0.912]

**Analysis:**
- Good separation between classes (0.832 vs 0.623)
- Some overlap (non-dup max 0.912 > dup min 0.412)
- Threshold 0.850 is between the two means

---

## Advanced Workflows

### 1. Cross-Validation for Threshold Stability

```python
from embeddings import ThresholdTuner, load_train_dataset
import numpy as np

df = load_train_dataset()
n_folds = 5
fold_size = len(df) // n_folds

thresholds = []
for i in range(n_folds):
    start = i * fold_size
    end = (i + 1) * fold_size
    df_fold = df.iloc[start:end]

    tuner = ThresholdTuner(max_samples=len(df_fold))
    result = tuner.tune_model('mpnet-base', df_fold, metric='f1')
    thresholds.append(result['optimal_threshold'])

print(f"Thresholds across folds: {thresholds}")
print(f"Mean: {np.mean(thresholds):.3f}")
print(f"Std: {np.std(thresholds):.3f}")
```

### 2. Threshold Heatmap Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Generate threshold vs metric curves
thresholds = np.arange(0.50, 1.00, 0.01)
metrics_data = []

for threshold in thresholds:
    predictions = similarities >= threshold
    f1 = f1_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)

    metrics_data.append({
        'threshold': threshold,
        'f1': f1,
        'precision': precision,
        'recall': recall
    })

# Plot
df_metrics = pd.DataFrame(metrics_data)
plt.figure(figsize=(10, 6))
plt.plot(df_metrics['threshold'], df_metrics['f1'], label='F1')
plt.plot(df_metrics['threshold'], df_metrics['precision'], label='Precision')
plt.plot(df_metrics['threshold'], df_metrics['recall'], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.title('Metrics vs Threshold')
plt.show()
```

### 3. Multi-Metric Pareto Frontier

```python
# Find thresholds that are Pareto-optimal for precision/recall
from embeddings import ThresholdTuner

tuner = ThresholdTuner(max_samples=5000)
similarities = tuner.compute_similarities('mpnet-base', df)
ground_truth = df['is_duplicate'].values

pareto_points = []
for threshold in np.arange(0.50, 1.00, 0.01):
    predictions = similarities >= threshold
    prec = precision_score(ground_truth, predictions)
    rec = recall_score(ground_truth, predictions)
    pareto_points.append((threshold, prec, rec))

# Find Pareto frontier (no other point has both higher prec and rec)
# ... implementation ...
```

---

## Troubleshooting

### Issue: Unstable Thresholds

**Symptoms:** Threshold varies significantly across runs

**Causes:**
- Sample size too small (<3000)
- Highly imbalanced dataset
- Model not converged

**Solutions:**
```python
# Increase sample size
tuner = ThresholdTuner(max_samples=10000)

# Check class balance
print(f"Duplicate ratio: {df['is_duplicate'].mean():.2%}")
# If <10% or >90%, consider stratified sampling
```

---

### Issue: All Predictions Same Class

**Symptoms:** Precision or recall is 0 or undefined

**Causes:**
- Threshold too high (all predictions 0)
- Threshold too low (all predictions 1)

**Solutions:**
- Model likely not suitable for task
- Try different model
- Check data quality

---

### Issue: Low Separation

**Symptoms:** Duplicate and non-duplicate distributions overlap heavily

**Example:**
```
Duplicates:     mean=0.65, std=0.20
Non-duplicates: mean=0.62, std=0.18
```

**Causes:**
- Model not well-suited for task
- Data quality issues
- Domain mismatch

**Solutions:**
```python
# Try different models
models = ['mpnet-base', 'gte-large', 'instructor-xl']
for model in models:
    tuner.tune_model(model, df, metric='f1')
    # Compare similarity stats
```

---

### Issue: Out of Memory

**Symptoms:** Process killed during embedding generation

**Solutions:**
```python
# Reduce sample size
tuner = ThresholdTuner(max_samples=2000)

# Or process in smaller batches (edit embedding_engine.py)
engine.encode_batch(texts, batch_size=16)  # Default is 32
```

---

## Summary

### Key Takeaways

1. ✅ **Always tune thresholds** - Don't use fixed 0.85
2. ✅ **Use training data** - Never tune on test data
3. ✅ **Choose right metric** - F1 (balanced), Precision (conservative), Recall (aggressive)
4. ✅ **Use enough samples** - 5000+ recommended
5. ✅ **Validate** - Test on separate validation set
6. ✅ **Document** - Save and version your thresholds

### Quick Reference

```python
# Standard workflow
from embeddings import tune_default_models

tuner = tune_default_models(max_samples=5000, metric='f1')
thresholds = tuner.get_optimal_thresholds()

# Use in production
from embeddings import EmbeddingEngine, get_model_info

model_info = get_model_info('mpnet-base')
engine = EmbeddingEngine(model_name=model_info.model_id)
threshold = thresholds['mpnet-base']

# Classify
emb1 = engine.encode("Question 1")
emb2 = engine.encode("Question 2")
similarity = np.dot(emb1, emb2)
is_duplicate = similarity >= threshold
```

---

## References

- [Threshold Tuner Implementation](threshold_tuner.py)
- [Tuning Examples](tune_thresholds_example.py)
- [Model Registry](model_registry.py)
- [Evaluator](evaluator.py)
