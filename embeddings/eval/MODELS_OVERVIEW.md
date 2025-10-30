# Embedding Models Overview

Complete list of **25 embedding models** available in the model registry, from fast prototyping models to state-of-the-art (SOTA) large models.

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

## Optimized Threshold Performance

Results from comprehensive threshold tuning across 21 available models (evaluated on semantic similarity task).

### Top 10 Performers (by F1 Score)

| Rank | Model | Category | Threshold | Accuracy | Precision | Recall | F1 |
|------|-------|----------|-----------|----------|-----------|--------|-----|
| 1 | **RoBERTa-Large** | quality | 0.770 | 0.8322 | 0.7256 | 0.8866 | **0.7981** |
| 2 | **MPNet-Base** | balanced | 0.780 | 0.8188 | 0.7146 | 0.8583 | **0.7799** |
| 3 | **Instructor-Large** | quality | 0.950 | 0.8152 | 0.7133 | 0.8460 | **0.7740** |
| 4 | **DistilRoBERTa** | balanced | 0.730 | 0.7986 | 0.6706 | 0.9070 | **0.7711** |
| 5 | **Instructor-XL** | quality | 0.880 | 0.8064 | 0.6976 | 0.8513 | **0.7669** |
| 6 | **Paraphrase-Multilingual-MPNet** | multilingual | 0.800 | 0.7920 | 0.6683 | 0.8813 | **0.7601** |
| 7 | **BGE-Large** | quality | 0.800 | 0.7800 | 0.6499 | 0.8925 | **0.7521** |
| 8 | **MiniLM-L12** | fast | 0.780 | 0.7876 | 0.6696 | 0.8529 | **0.7502** |
| 9 | **MiniLM-L6** | fast | 0.740 | 0.7734 | 0.6408 | 0.8968 | **0.7475** |
| 10 | **BGE-Base** | quality | 0.810 | 0.7770 | 0.6490 | 0.8791 | **0.7468** |

### All Models Performance (Sorted by F1)

| Model | Category | Threshold | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|----------|-----------|--------|-----|
| RoBERTa-Large | quality | 0.770 | 0.8322 | 0.7256 | 0.8866 | 0.7981 |
| MPNet-Base | balanced | 0.780 | 0.8188 | 0.7146 | 0.8583 | 0.7799 |
| Instructor-Large | quality | 0.950 | 0.8152 | 0.7133 | 0.8460 | 0.7740 |
| DistilRoBERTa | balanced | 0.730 | 0.7986 | 0.6706 | 0.9070 | 0.7711 |
| Instructor-XL | quality | 0.880 | 0.8064 | 0.6976 | 0.8513 | 0.7669 |
| Paraphrase-Multilingual-MPNet | multilingual | 0.800 | 0.7920 | 0.6683 | 0.8813 | 0.7601 |
| BGE-Large | quality | 0.800 | 0.7800 | 0.6499 | 0.8925 | 0.7521 |
| MiniLM-L12 | fast | 0.780 | 0.7876 | 0.6696 | 0.8529 | 0.7502 |
| MiniLM-L6 | fast | 0.740 | 0.7734 | 0.6408 | 0.8968 | 0.7475 |
| BGE-Base | quality | 0.810 | 0.7770 | 0.6490 | 0.8791 | 0.7468 |
| Paraphrase-Multilingual | multilingual | 0.770 | 0.7724 | 0.6449 | 0.8711 | 0.7411 |
| GTE-Large | quality | 0.910 | 0.7582 | 0.6285 | 0.8642 | 0.7278 |
| E5-Base | quality | 0.900 | 0.7508 | 0.6171 | 0.8791 | 0.7252 |
| DistilUSE-Multilingual | multilingual | 0.750 | 0.7476 | 0.6144 | 0.8733 | 0.7213 |
| GTE-Base | quality | 0.900 | 0.7418 | 0.6055 | 0.8888 | 0.7203 |
| E5-Large | quality | 0.890 | 0.7414 | 0.6092 | 0.8604 | 0.7134 |
| MS-MARCO-DistilBERT | balanced | 0.620 | 0.7366 | 0.6019 | 0.8733 | 0.7126 |
| MS-MARCO-MiniLM | balanced | 0.680 | 0.7312 | 0.5982 | 0.8567 | 0.7045 |
| LaBSE | multilingual | 0.710 | 0.7192 | 0.5842 | 0.8642 | 0.6972 |
| Legal-BERT | balanced | 0.910 | 0.6292 | 0.5026 | 0.8353 | 0.6276 |
| SciBERT | balanced | 0.840 | 0.6102 | 0.4878 | 0.8481 | 0.6194 |

### Key Insights

1. **Best Overall**: RoBERTa-Large achieves the highest F1 score (0.7981) with optimal threshold at 0.770
2. **Best Balanced Model**: MPNet-Base (F1: 0.7799, threshold: 0.780) - excellent performance for production use
3. **Best Fast Model**: MiniLM-L12 (F1: 0.7502, threshold: 0.780) - strong performance with minimal resources
4. **Domain Models**: Legal-BERT and SciBERT show lower performance on general semantic similarity task (specialized for domain-specific use)

---

## MTEB Leaderboard Rankings

Check [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for latest rankings.*


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
