# Embeddings Package

Comprehensive embedding evaluation and comparison framework for Privacy Aware Semantic Cache.

## Overview

This package provides:
- **Embedding Engine**: Encode text using Sentence Transformers models
- **Model Registry**: Curated list of embedding models for different use cases
- **Evaluation Framework**: Evaluate model performance with detailed metrics
- **Model Comparison**: Compare multiple models side-by-side

## Quick Start

### 1. Install Dependencies

```bash
pip3 install -r ../requirements.txt
```

### 2. Basic Usage

```python
from embeddings import EmbeddingEngine

# Create embedding engine
engine = EmbeddingEngine(model_name='all-MiniLM-L6-v2')

# Encode text
embedding = engine.encode("How do I reset my password?")
print(embedding.shape)  # (384,)
```

### 3. Evaluate a Single Model

```python
from embeddings import run_evaluation

# Evaluate with default settings (threshold=0.85, max_samples=1000)
metrics = run_evaluation()

# Custom settings
metrics = run_evaluation(threshold=0.80, max_samples=500)
```

### 4. Tune Thresholds and Compare Models

```python
from embeddings.eval.tune_thresholds_utils import run_tune_all_models

# Automatically find optimal thresholds for all models
# Each model is tuned individually for best F1 score
tuner = run_tune_all_models()

# Get optimal thresholds
thresholds = tuner.get_optimal_thresholds()
print(thresholds)  # {'mpnet-base': 0.780, 'roberta-large': 0.770, ...}
```

## Available Models

### Fast Models (Optimized for Speed)
- **minilm-l6**: MiniLM-L6-v2 (384d) - Fast and efficient
- **minilm-l3**: MiniLM-L3-v2 (384d) - Ultra-fast

### Balanced Models (Speed/Quality Trade-off)
- **mpnet-base**: MPNet-Base-v2 (768d) - Best overall quality
- **distilroberta**: DistilRoBERTa-v1 (768d) - Good balance
- **msmarco-distilbert**: MS-MARCO DistilBERT (768d) - Specialized for search

### Quality Models (Optimized for Accuracy)
- **roberta-large**: RoBERTa-Large-v1 (1024d) - Highest quality

### Multilingual Models
- **paraphrase-multilingual**: Multilingual MiniLM (384d)
- **distiluse-multilingual**: Multilingual USE (512d)

View all models:
```python
from embeddings import print_model_registry
print_model_registry()
```

## Module Reference

### embedding_engine.py
Core embedding functionality.

```python
from embeddings import EmbeddingEngine

engine = EmbeddingEngine(model_name='all-mpnet-base-v2')
embeddings = engine.encode_batch(["text1", "text2"])
dim = engine.get_embedding_dimension()
```

### evaluator.py
Evaluate model performance on question similarity dataset.

```python
from embeddings import SimilarityEvaluator, EmbeddingEngine

engine = EmbeddingEngine()
evaluator = SimilarityEvaluator(engine, similarity_threshold=0.85)
metrics = evaluator.evaluate_dataset(df, max_samples=1000)
evaluator.print_evaluation_report(metrics)
```

**Metrics Provided:**
- Accuracy, Precision, Recall, F1 Score, Specificity
- Confusion Matrix (TP, TN, FP, FN)
- Similarity statistics (mean, std, min, max, median)
- Average similarity by category

### model_registry.py
Registry of available embedding models.

```python
from embeddings import (
    get_model_info,
    get_all_model_keys,
    get_models_by_category,
    get_default_comparison_set
)

# Get info about a specific model
info = get_model_info('mpnet-base')
print(f"{info.name}: {info.dimension}d, {info.category}")

# Get all models in a category
fast_models = get_models_by_category('fast')

# Get default comparison set
default_models = get_default_comparison_set()
```

### threshold_tuner.py & tune_thresholds_utils.py
Automatically find optimal thresholds for each model.

```python
from embeddings.eval.tune_thresholds_utils import (
    run_tune_all_models,
    run_tune_default_models,
    run_tune_specific_models
)

# Tune all 21 models (comprehensive)
tuner = run_tune_all_models()
thresholds = tuner.get_optimal_thresholds()

# Tune default comparison set
run_tune_default_models()

# Tune specific models
run_tune_specific_models()  # Edit function to specify models
```

**Why use threshold tuning:**
- Each model performs best at a different threshold (e.g., RoBERTa: 0.770, Instructor: 0.950)
- Automatic optimization for F1, precision, or recall
- More accurate model comparisons than fixed thresholds
- See [MODELS_OVERVIEW.md](eval/MODELS_OVERVIEW.md) for optimized results

### model_comparison.py
Compare models with fixed thresholds (use threshold tuning instead for better results).

```python
from embeddings import ModelComparison

# Create comparison with fixed threshold (not recommended)
comparison = ModelComparison(threshold=0.85, max_samples=1000)
comparison.compare_models(['minilm-l6', 'mpnet-base'])
```

### dataset.py
Load question similarity dataset.

```python
from embeddings import load_dataset, get_csv_file_path

# Load dataset
df = load_dataset()
print(f"Loaded {len(df)} question pairs")

# Get CSV path
path = get_csv_file_path()
```

## Example Scripts

### Run Examples

```bash
# View available models
python3 -m embeddings.eval.model_registry

# Evaluate single model
python3 -m embeddings.eval.evaluator

# Tune thresholds for all models (recommended)
python3 -m embeddings.eval.tune_thresholds_utils

# Basic embedding example
python3 -m embeddings.example_usage
```

Note: Results from threshold tuning and model comparison are automatically saved to `embeddings/eval/results/`

## Dataset Format

The evaluation dataset (`data/questions.csv`) should have these columns:
- `id`: Question pair ID
- `qid1`, `qid2`: Individual question IDs
- `question1`, `question2`: Question text
- `is_duplicate`: Binary label (1 = semantically similar, 0 = different)

## Configuration

### Automatic Threshold Optimization (Recommended)

```python
from embeddings.eval.threshold_tuner import ThresholdTuner

# Automatically find optimal threshold for each model
tuner = ThresholdTuner(max_samples=5000)
tuner.tune_models(['mpnet-base', 'roberta-large'], metric='f1')

# Get optimized thresholds
optimal_thresholds = tuner.get_optimal_thresholds()
# Result: {'mpnet-base': 0.780, 'roberta-large': 0.770}
```

### Adjust Sample Size

```python
# Quick tuning (1000 samples)
ThresholdTuner(max_samples=1000)

# Standard tuning (5000 samples)
ThresholdTuner(max_samples=5000)

# Comprehensive tuning (10000+ samples)
ThresholdTuner(max_samples=10000)
```

## Performance Tips

1. **Start with fast models** for prototyping
2. **Use default comparison set** for balanced coverage
3. **Adjust max_samples** based on available time
4. **Save results** for later analysis
5. **Compare thresholds** to find optimal value

## Output

### Evaluation Report Example
```
======================================================================
EVALUATION REPORT
======================================================================

Threshold: 0.850

--- Classification Metrics ---
Accuracy:    0.7842
Precision:   0.8234
Recall:      0.7156
F1 Score:    0.7658
Specificity: 0.8421

--- Confusion Matrix ---
True Positives:  285
True Negatives:  499
False Positives: 93
False Negatives: 123
```

### Comparison Report Example
```
====================================================================================================
MODEL COMPARISON REPORT
====================================================================================================

Model              Category   Dim  Accuracy  Precision  Recall    F1        Time (s)
MPNet-Base         balanced   768  0.7920    0.8456     0.7234    0.7798    45.23
DistilRoBERTa      balanced   768  0.7856    0.8312     0.7189    0.7705    42.18
MiniLM-L6          fast       384  0.7734    0.8123     0.7012    0.7523    18.45

🏆 Best Model (by F1): MPNet-Base (F1: 0.7798)
```

## Troubleshooting

**Model download issues:** First run downloads models from HuggingFace. Ensure internet connection.

**Memory errors:** Reduce `max_samples` or use smaller models (minilm-l3, minilm-l6).

**Slow evaluation:** Use fast models or reduce `max_samples` for quick tests.

## Directory Structure

```
embeddings/
├── __init__.py                  # Package initialization
├── embedding_engine.py          # Core embedding functionality
├── dataset.py                   # Dataset loading
├── example_usage.py             # Basic usage examples
├── README.md                    # This file
└── eval/                        # Evaluation module
    ├── __init__.py              # Eval module initialization
    ├── evaluator.py             # Single model evaluation
    ├── model_registry.py        # Model registry (25 models)
    ├── model_comparison.py      # Multi-model comparison
    ├── threshold_tuner.py       # Automatic threshold optimization
    ├── tune_thresholds_utils.py # Threshold tuning utilities
    ├── MODELS_OVERVIEW.md       # Complete model documentation
    ├── THRESHOLD_TUNING.md      # Threshold tuning guide
    └── results/                 # Results directory (JSON files saved here)
        └── .gitignore           # Ignore result files in git
```
