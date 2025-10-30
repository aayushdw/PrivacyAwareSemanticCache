# Embedding Models Overview

Complete list of **25 embedding models** available in the model registry, from fast prototyping models to state-of-the-art (SOTA) large models.

## Summary by Category

- **Fast Models**: 3 models (384d) - optimized for speed
- **Balanced Models**: 5 models (384-768d) - good speed/quality trade-off
- **Quality Models**: 9 models (768-1024d) - **LARGE MODELS** with highest accuracy
- **Multilingual Models**: 4 models (384-768d) - support 50-100+ languages
- **Domain-Specific**: 2 models (768d) - specialized for legal/scientific text

**Total: 25 models**

---

## Fast Models (3)
*Optimized for speed, great for prototyping and development*

| Key | Name | Dimensions | Description |
|-----|------|------------|-------------|
| `minilm-l3` | MiniLM-L3 | 384 | Ultra-fast, smallest model |
| `minilm-l6` | MiniLM-L6 | 384 | Fast and efficient baseline |
| `minilm-l12` | MiniLM-L12 | 384 | Balanced MiniLM with 12 layers |

---

## Balanced Models (5)
*Good speed/quality trade-off, recommended for most use cases*

| Key | Name | Dimensions | Description |
|-----|------|------------|-------------|
| `mpnet-base` | MPNet-Base | 768 | **RECOMMENDED** - best balanced performance |
| `distilroberta` | DistilRoBERTa | 768 | Distilled RoBERTa |
| `msmarco-distilbert` | MS-MARCO-DistilBERT | 768 | Trained on MS-MARCO passage ranking |
| `msmarco-minilm` | MS-MARCO-MiniLM | 384 | MS-MARCO trained, optimized for cosine similarity |
| `legal-bert` | Legal-BERT | 768 | Legal domain specialized |
| `scibert` | SciBERT | 768 | Scientific text specialized |

---

## Quality Models (9) - LARGE MODELS ⭐
*Highest accuracy, state-of-the-art performance on benchmarks like MTEB*

### Large Models (1024 dimensions)

| Key | Name | Dimensions | Description |
|-----|------|------------|-------------|
| `roberta-large` | RoBERTa-Large | 1024 | High quality large RoBERTa |
| `gte-large` | **GTE-Large** | 1024 | **Top MTEB performer** - General Text Embeddings |
| `e5-large` | **E5-Large** | 1024 | Trained on diverse datasets |
| `bge-large` | **BGE-Large** | 1024 | **Top MTEB ranking** - BAAI BGE |

### Base Quality Models (768 dimensions)

| Key | Name | Dimensions | Description |
|-----|------|------------|-------------|
| `instructor-large` | **Instructor-Large** | 768 | **SOTA** - Instruction-based embeddings |
| `instructor-xl` | **Instructor-XL** | 768 | **Highest quality** - Extra-large instructor |
| `gte-base` | GTE-Base | 768 | Excellent quality, more efficient than large |
| `e5-base` | E5-Base | 768 | Strong performance |
| `bge-base` | BGE-Base | 768 | Excellent balance of quality and speed |

---

## Multilingual Models (4)
*Support for 50-100+ languages*

| Key | Name | Dimensions | Languages | Description |
|-----|------|------------|-----------|-------------|
| `paraphrase-multilingual` | Paraphrase-Multilingual | 384 | 50+ | Multilingual paraphrase detection |
| `paraphrase-multilingual-mpnet` | Paraphrase-Multilingual-MPNet | 768 | 50+ | Highest quality multilingual |
| `distiluse-multilingual` | DistilUSE-Multilingual | 512 | 50+ | Multilingual USE |
| `labse` | LaBSE | 768 | 100+ | Language-agnostic BERT |

---

## Performance Expectations

### Fast Models (384d)
- **Speed**: ⚡⚡⚡ Very Fast
- **Quality**: ⭐⭐ Good
- **Use Cases**: Prototyping, development, real-time applications
- **Memory**: ~100-200 MB

### Balanced Models (768d)
- **Speed**: ⚡⚡ Fast
- **Quality**: ⭐⭐⭐ Very Good
- **Use Cases**: Production, general-purpose semantic search
- **Memory**: ~200-400 MB

### Quality Models - Base (768d)
- **Speed**: ⚡ Moderate
- **Quality**: ⭐⭐⭐⭐ Excellent
- **Use Cases**: High-accuracy requirements, competitive benchmarks
- **Memory**: ~400-800 MB

### Quality Models - Large (1024d)
- **Speed**: 🐢 Slower
- **Quality**: ⭐⭐⭐⭐⭐ State-of-the-art
- **Use Cases**: Maximum accuracy, research, benchmarking
- **Memory**: ~800-1500 MB

---

## Recommended Models by Use Case

### 🚀 Quick Prototyping
```python
models = ['minilm-l6']
```

### 🎯 Production (Balanced)
```python
models = ['mpnet-base', 'gte-base', 'bge-base']
```

### 🏆 Maximum Quality (Research/Benchmarking)
```python
models = ['instructor-xl', 'gte-large', 'bge-large']
```

### 🌍 Multilingual Applications
```python
models = ['paraphrase-multilingual-mpnet', 'labse']
```

### ⚖️ Legal Domain
```python
models = ['legal-bert']
```

### 🔬 Scientific Domain
```python
models = ['scibert']
```

---

## Model Comparison Sets

### Default Set (4 models)
```python
from embeddings import compare_default_models

compare_default_models()
# Tests: minilm-l6, mpnet-base, gte-base, bge-base
```

### Comprehensive Set (8 models)
```python
from embeddings import get_comprehensive_comparison_set, ModelComparison

models = get_comprehensive_comparison_set()
comparison = ModelComparison()
comparison.compare_models(models)
# Tests: minilm-l6, mpnet-base, gte-base, gte-large,
#        bge-base, bge-large, e5-base, e5-large
```

### All Quality Models (9 models)
```python
from embeddings import compare_category

compare_category('quality')
# Tests all 9 quality models
```

### All Models (25 models)
```python
from embeddings import compare_all_models

compare_all_models(max_samples=500)  # Use fewer samples for speed
```

---

## MTEB Leaderboard Rankings

The **Massive Text Embedding Benchmark (MTEB)** is the standard for evaluating embedding models.

### Top Performers in This Registry (approximate MTEB scores)

1. **Instructor-XL**: ~68-70 (instruction-based, very versatile)
2. **GTE-Large**: ~65-67 (general-purpose, top performer)
3. **BGE-Large**: ~65-67 (consistently high scores)
4. **E5-Large**: ~64-66 (diverse training data)
5. **GTE-Base**: ~63-65 (best base model)
6. **BGE-Base**: ~63-65 (excellent balance)
7. **MPNet-Base**: ~62-64 (classic strong baseline)

*Note: Exact scores vary by task. Check [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for latest rankings.*

---

## Usage Examples

### Test a Single Large Model
```python
from embeddings import EmbeddingEngine, SimilarityEvaluator, load_train_dataset

# Use a large SOTA model
engine = EmbeddingEngine(model_name='BAAI/bge-large-en-v1.5')
evaluator = SimilarityEvaluator(engine, similarity_threshold=0.85)

df = load_train_dataset()
metrics = evaluator.evaluate_dataset(df, max_samples=1000)
evaluator.print_evaluation_report(metrics)
```

### Compare Fast vs Quality Models
```python
from embeddings import ModelComparison

comparison = ModelComparison(threshold=0.85, max_samples=1000)
comparison.compare_models([
    'minilm-l6',      # Fast baseline
    'mpnet-base',     # Balanced
    'gte-large',      # SOTA quality
    'bge-large'       # SOTA quality alternative
])
```

### Find the Best Model for Your Task
```python
from embeddings import compare_all_models

# Compare all 25 models
comparison = compare_all_models(threshold=0.85, max_samples=500)

# Results saved to embeddings/eval/results/
# Check which model has highest F1 score for your data
```

---

## Adding More Models

To add custom models, edit `embeddings/eval/model_registry.py`:

```python
MODEL_REGISTRY = {
    'custom-model': ModelInfo(
        name='Custom Model Name',
        model_id='huggingface/model-id',
        description='Description of model',
        dimension=768,
        category='balanced'  # or 'fast', 'quality', 'multilingual'
    ),
    # ... existing models
}
```

Any model from [HuggingFace Sentence Transformers](https://huggingface.co/models?library=sentence-transformers) can be added!

---

## References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [HuggingFace Models](https://huggingface.co/models?library=sentence-transformers)
