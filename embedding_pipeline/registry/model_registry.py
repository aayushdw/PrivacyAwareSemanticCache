"""
Model Registry for embedding models.

Provides a curated list of embedding models for evaluation and comparison,
enhanced with MPS compatibility flags and memory estimates for M4 MacBook Air.
"""

from typing import Dict, List, Optional
from .model_info import ModelInfo


# Registry of available models for evaluation
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # Fast models - optimized for speed (< 200MB, parallelizable)
    "minilm-l6": ModelInfo(
        name="MiniLM-L6",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        description="Fast and efficient model, good for prototyping",
        dimension=384,
        category="fast",
        estimated_memory_mb=90.0,
        max_seq_length=256,
    ),
    "minilm-l3": ModelInfo(
        name="MiniLM-L3",
        model_id="sentence-transformers/all-MiniLM-L3-v2",
        description="Ultra-fast model with smaller dimension",
        dimension=384,
        category="fast",
        estimated_memory_mb=60.0,
        max_seq_length=256,
    ),
    "minilm-l12": ModelInfo(
        name="MiniLM-L12",
        model_id="sentence-transformers/all-MiniLM-L12-v2",
        description="Balanced MiniLM with 12 layers",
        dimension=384,
        category="fast",
        estimated_memory_mb=120.0,
        max_seq_length=256,
    ),
    # Balanced models - good speed/quality trade-off (200-500MB)
    "mpnet-base": ModelInfo(
        name="MPNet-Base",
        model_id="sentence-transformers/all-mpnet-base-v2",
        description="Best quality for balanced performance (RECOMMENDED)",
        dimension=768,
        category="balanced",
        estimated_memory_mb=420.0,
        max_seq_length=384,
    ),
    "distilroberta": ModelInfo(
        name="DistilRoBERTa",
        model_id="sentence-transformers/all-distilroberta-v1",
        description="Distilled RoBERTa model, good balance",
        dimension=768,
        category="balanced",
        estimated_memory_mb=330.0,
        max_seq_length=512,
    ),
    "msmarco-distilbert": ModelInfo(
        name="MS-MARCO-DistilBERT",
        model_id="sentence-transformers/msmarco-distilbert-base-v4",
        description="Trained on MS-MARCO passage ranking",
        dimension=768,
        category="balanced",
        estimated_memory_mb=260.0,
        max_seq_length=512,
    ),
    "msmarco-minilm": ModelInfo(
        name="MS-MARCO-MiniLM",
        model_id="sentence-transformers/msmarco-MiniLM-L6-cos-v5",
        description="MS-MARCO trained MiniLM, optimized for cosine similarity",
        dimension=384,
        category="balanced",
        estimated_memory_mb=90.0,
        max_seq_length=512,
    ),
    # Quality models - optimized for accuracy (> 500MB, run sequentially)
    "roberta-large": ModelInfo(
        name="RoBERTa-Large",
        model_id="sentence-transformers/all-roberta-large-v1",
        description="High quality large model (1024d)",
        dimension=1024,
        category="quality",
        estimated_memory_mb=1400.0,
        max_seq_length=512,
    ),
    "instructor-large": ModelInfo(
        name="Instructor-Large",
        model_id="hkunlp/instructor-large",
        description="Instruction-based embedding model (768d) - SOTA performance",
        dimension=768,
        category="quality",
        estimated_memory_mb=1340.0,
        requires_instruction=True,
        instruction_template="Represent the question for semantic similarity: ",
        max_seq_length=512,
    ),
    "gte-large": ModelInfo(
        name="GTE-Large",
        model_id="thenlper/gte-large",
        description="General Text Embeddings Large (1024d) - MTEB top performer",
        dimension=1024,
        category="quality",
        estimated_memory_mb=1340.0,
        max_seq_length=512,
    ),
    "gte-base": ModelInfo(
        name="GTE-Base",
        model_id="thenlper/gte-base",
        description="General Text Embeddings Base (768d) - excellent quality",
        dimension=768,
        category="quality",
        estimated_memory_mb=440.0,
        max_seq_length=512,
    ),
    "e5-large": ModelInfo(
        name="E5-Large",
        model_id="intfloat/e5-large-v2",
        description="E5 Large model (1024d) - trained on diverse datasets",
        dimension=1024,
        category="quality",
        estimated_memory_mb=1340.0,
        requires_instruction=True,
        instruction_template="query: ",
        max_seq_length=512,
    ),
    "e5-base": ModelInfo(
        name="E5-Base",
        model_id="intfloat/e5-base-v2",
        description="E5 Base model (768d) - strong performance",
        dimension=768,
        category="quality",
        estimated_memory_mb=440.0,
        requires_instruction=True,
        instruction_template="query: ",
        max_seq_length=512,
    ),
    "bge-large": ModelInfo(
        name="BGE-Large",
        model_id="BAAI/bge-large-en-v1.5",
        description="BGE Large English (1024d) - top MTEB ranking",
        dimension=1024,
        category="quality",
        estimated_memory_mb=1340.0,
        max_seq_length=512,
    ),
    "bge-base": ModelInfo(
        name="BGE-Base",
        model_id="BAAI/bge-base-en-v1.5",
        description="BGE Base English (768d) - excellent balance",
        dimension=768,
        category="quality",
        estimated_memory_mb=440.0,
        max_seq_length=512,
    ),
    "bge-small": ModelInfo(
        name="BGE-Small",
        model_id="BAAI/bge-small-en-v1.5",
        description="BGE Small English (384d) - fast quality",
        dimension=384,
        category="balanced",
        estimated_memory_mb=130.0,
        max_seq_length=512,
    ),
    # Multilingual models
    "paraphrase-multilingual": ModelInfo(
        name="Paraphrase-Multilingual",
        model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Multilingual paraphrase detection (50+ languages)",
        dimension=384,
        category="multilingual",
        estimated_memory_mb=470.0,
        max_seq_length=128,
    ),
    "paraphrase-multilingual-mpnet": ModelInfo(
        name="Paraphrase-Multilingual-MPNet",
        model_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="Multilingual MPNet (768d) - highest quality multilingual",
        dimension=768,
        category="multilingual",
        estimated_memory_mb=1100.0,
        max_seq_length=128,
    ),
    "distiluse-multilingual": ModelInfo(
        name="DistilUSE-Multilingual",
        model_id="sentence-transformers/distiluse-base-multilingual-cased-v2",
        description="Multilingual Universal Sentence Encoder (50+ languages)",
        dimension=512,
        category="multilingual",
        estimated_memory_mb=540.0,
        max_seq_length=128,
    ),
    "labse": ModelInfo(
        name="LaBSE",
        model_id="sentence-transformers/LaBSE",
        description="Language-agnostic BERT (768d) - 100+ languages",
        dimension=768,
        category="multilingual",
        estimated_memory_mb=1880.0,
        max_seq_length=256,
    ),
    # Qwen3 Embedding Models - SOTA multilingual
    "qwen3-0.6b": ModelInfo(
        name="Qwen3-Embedding-0.6B",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        description="Qwen3 0.6B (1024d) - Fast SOTA multilingual, 100+ languages",
        dimension=1024,
        category="quality",
        estimated_memory_mb=1200.0,
        max_seq_length=8192,
        supports_lora=True,
    ),
}


def get_model_info(model_key: str) -> ModelInfo:
    """Get information about a specific model."""
    if model_key not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Model '{model_key}' not found. Available models: {available}"
        )
    return MODEL_REGISTRY[model_key]


def get_all_model_keys() -> List[str]:
    """Get list of all available model keys."""
    return list(MODEL_REGISTRY.keys())


def get_models_by_category(category: str) -> Dict[str, ModelInfo]:
    """Get all models in a specific category."""
    return {
        key: info
        for key, info in MODEL_REGISTRY.items()
        if info.category == category
    }


def get_models_by_size(
    max_memory_mb: float = 500.0, min_memory_mb: float = 0.0
) -> Dict[str, ModelInfo]:
    """Get models within a memory range."""
    return {
        key: info
        for key, info in MODEL_REGISTRY.items()
        if min_memory_mb <= (info.estimated_memory_mb or 0) <= max_memory_mb
    }


def get_small_models(threshold_mb: float = 500.0) -> Dict[str, ModelInfo]:
    """Get models suitable for parallel evaluation."""
    return {
        key: info
        for key, info in MODEL_REGISTRY.items()
        if info.is_small_model(threshold_mb)
    }


def get_large_models(threshold_mb: float = 500.0) -> Dict[str, ModelInfo]:
    """Get models that should be evaluated sequentially."""
    return {
        key: info
        for key, info in MODEL_REGISTRY.items()
        if not info.is_small_model(threshold_mb)
    }


def get_mps_compatible_models() -> Dict[str, ModelInfo]:
    """Get models compatible with Apple Silicon MPS."""
    return {
        key: info
        for key, info in MODEL_REGISTRY.items()
        if info.supports_mps
    }


def get_lora_compatible_models() -> Dict[str, ModelInfo]:
    """Get models that support LoRA fine-tuning."""
    return {
        key: info
        for key, info in MODEL_REGISTRY.items()
        if info.supports_lora
    }


def get_default_comparison_set() -> List[str]:
    """Get a default set of models for quick comparison."""
    return [
        "minilm-l6",  # Fast baseline
        "mpnet-base",  # Balanced - recommended
        "gte-base",  # Quality - MTEB top performer
        "bge-base",  # Quality - alternative
    ]


def get_comprehensive_comparison_set() -> List[str]:
    """Get a comprehensive set including larger/SOTA models."""
    return [
        "minilm-l6",
        "minilm-l12",
        "mpnet-base",
        "distilroberta",
        "gte-base",
        "gte-large",
        "bge-base",
        "bge-large",
        "e5-base",
        "e5-large",
    ]


def get_full_evaluation_set() -> List[str]:
    """Get all models for full evaluation pipeline."""
    return get_all_model_keys()


def get_models_for_evaluation(
    categories: Optional[List[str]] = None,
    max_memory_mb: Optional[float] = None,
    require_lora: bool = True,
    require_mps: bool = True,
) -> List[ModelInfo]:
    """
    Get filtered list of models for evaluation.

    Args:
        categories: Filter by category (fast, balanced, quality, multilingual)
        max_memory_mb: Maximum memory in MB
        require_lora: Only include LoRA-compatible models
        require_mps: Only include MPS-compatible models

    Returns:
        List of ModelInfo objects matching criteria
    """
    models = list(MODEL_REGISTRY.values())

    if categories:
        models = [m for m in models if m.category in categories]

    if max_memory_mb is not None:
        models = [m for m in models if (m.estimated_memory_mb or 0) <= max_memory_mb]

    if require_lora:
        models = [m for m in models if m.supports_lora]

    if require_mps:
        models = [m for m in models if m.supports_mps]

    return models


def print_model_registry():
    """Print all available models in a formatted table."""
    print("\n" + "=" * 90)
    print("AVAILABLE EMBEDDING MODELS")
    print("=" * 90)

    categories = ["fast", "balanced", "quality", "multilingual"]

    for category in categories:
        models = get_models_by_category(category)
        if not models:
            continue

        print(f"\n{category.upper()} Models:")
        print("-" * 90)

        for key, info in models.items():
            mem = info.estimated_memory_mb or 0
            print(f"  [{key}]")
            print(f"    Model ID:  {info.model_id}")
            print(f"    Dimension: {info.dimension}d | Memory: ~{mem:.0f}MB")
            print(f"    MPS: {'Yes' if info.supports_mps else 'No'} | "
                  f"LoRA: {'Yes' if info.supports_lora else 'No'}")
            print()

    print("=" * 90)


if __name__ == "__main__":
    print_model_registry()

    print("\nDefault comparison set:")
    for key in get_default_comparison_set():
        info = get_model_info(key)
        print(f"  - {key}: {info.name} ({info.dimension}d, ~{info.estimated_memory_mb:.0f}MB)")

    print("\nSmall models (parallelizable):")
    for key, info in get_small_models().items():
        print(f"  - {key}: ~{info.estimated_memory_mb:.0f}MB")
