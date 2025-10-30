"""
Model Registry for embedding models.
Provides a curated list of embedding models for evaluation and comparison.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about an embedding model."""
    name: str
    model_id: str
    description: str
    dimension: int
    category: str  # 'fast', 'balanced', 'quality', 'multilingual'


# Registry of available models for evaluation
MODEL_REGISTRY = {
    # Fast models - optimized for speed
    'minilm-l6': ModelInfo(
        name='MiniLM-L6',
        model_id='all-MiniLM-L6-v2',
        description='Fast and efficient model, good for prototyping',
        dimension=384,
        category='fast'
    ),
    'minilm-l3': ModelInfo(
        name='MiniLM-L3',
        model_id='all-MiniLM-L3-v2',
        description='Ultra-fast model with smaller dimension',
        dimension=384,
        category='fast'
    ),
    'minilm-l12': ModelInfo(
        name='MiniLM-L12',
        model_id='all-MiniLM-L12-v2',
        description='Balanced MiniLM with 12 layers',
        dimension=384,
        category='fast'
    ),

    # Balanced models - good speed/quality trade-off
    'mpnet-base': ModelInfo(
        name='MPNet-Base',
        model_id='all-mpnet-base-v2',
        description='Best quality for balanced performance (RECOMMENDED)',
        dimension=768,
        category='balanced'
    ),
    'distilroberta': ModelInfo(
        name='DistilRoBERTa',
        model_id='all-distilroberta-v1',
        description='Distilled RoBERTa model, good balance',
        dimension=768,
        category='balanced'
    ),
    'msmarco-distilbert': ModelInfo(
        name='MS-MARCO-DistilBERT',
        model_id='msmarco-distilbert-base-v4',
        description='Trained on MS-MARCO passage ranking',
        dimension=768,
        category='balanced'
    ),
    'msmarco-minilm': ModelInfo(
        name='MS-MARCO-MiniLM',
        model_id='msmarco-MiniLM-L6-cos-v5',
        description='MS-MARCO trained MiniLM, optimized for cosine similarity',
        dimension=384,
        category='balanced'
    ),

    # Quality models - optimized for accuracy (LARGE MODELS)
    'roberta-large': ModelInfo(
        name='RoBERTa-Large',
        model_id='all-roberta-large-v1',
        description='High quality large model (1024d)',
        dimension=1024,
        category='quality'
    ),
    'instructor-large': ModelInfo(
        name='Instructor-Large',
        model_id='hkunlp/instructor-large',
        description='Instruction-based embedding model (768d) - SOTA performance',
        dimension=768,
        category='quality'
    ),
    'instructor-xl': ModelInfo(
        name='Instructor-XL',
        model_id='hkunlp/instructor-xl',
        description='Extra-large instruction model (768d) - highest quality',
        dimension=768,
        category='quality'
    ),
    'gte-large': ModelInfo(
        name='GTE-Large',
        model_id='thenlper/gte-large',
        description='General Text Embeddings Large (1024d) - MTEB top performer',
        dimension=1024,
        category='quality'
    ),
    'gte-base': ModelInfo(
        name='GTE-Base',
        model_id='thenlper/gte-base',
        description='General Text Embeddings Base (768d) - excellent quality',
        dimension=768,
        category='quality'
    ),
    'e5-large': ModelInfo(
        name='E5-Large',
        model_id='intfloat/e5-large-v2',
        description='E5 Large model (1024d) - trained on diverse datasets',
        dimension=1024,
        category='quality'
    ),
    'e5-base': ModelInfo(
        name='E5-Base',
        model_id='intfloat/e5-base-v2',
        description='E5 Base model (768d) - strong performance',
        dimension=768,
        category='quality'
    ),
    'bge-large': ModelInfo(
        name='BGE-Large',
        model_id='BAAI/bge-large-en-v1.5',
        description='BGE Large English (1024d) - top MTEB ranking',
        dimension=1024,
        category='quality'
    ),
    'bge-base': ModelInfo(
        name='BGE-Base',
        model_id='BAAI/bge-base-en-v1.5',
        description='BGE Base English (768d) - excellent balance',
        dimension=768,
        category='quality'
    ),

    # Multilingual models
    'paraphrase-multilingual': ModelInfo(
        name='Paraphrase-Multilingual',
        model_id='paraphrase-multilingual-MiniLM-L12-v2',
        description='Multilingual paraphrase detection (50+ languages)',
        dimension=384,
        category='multilingual'
    ),
    'paraphrase-multilingual-mpnet': ModelInfo(
        name='Paraphrase-Multilingual-MPNet',
        model_id='paraphrase-multilingual-mpnet-base-v2',
        description='Multilingual MPNet (768d) - highest quality multilingual',
        dimension=768,
        category='multilingual'
    ),
    'distiluse-multilingual': ModelInfo(
        name='DistilUSE-Multilingual',
        model_id='distiluse-base-multilingual-cased-v2',
        description='Multilingual Universal Sentence Encoder (50+ languages)',
        dimension=512,
        category='multilingual'
    ),
    'labse': ModelInfo(
        name='LaBSE',
        model_id='sentence-transformers/LaBSE',
        description='Language-agnostic BERT (768d) - 100+ languages',
        dimension=768,
        category='multilingual'
    ),

    # Domain-specific models
    'legal-bert': ModelInfo(
        name='Legal-BERT',
        model_id='nlpaueb/legal-bert-base-uncased',
        description='Legal domain specialized BERT (768d)',
        dimension=768,
        category='balanced'
    ),
    'scibert': ModelInfo(
        name='SciBERT',
        model_id='allenai/scibert_scivocab_uncased',
        description='Scientific text specialized BERT (768d)',
        dimension=768,
        category='balanced'
    ),
}


def get_model_info(model_key: str) -> ModelInfo:
    """
    Get information about a specific model.

    Args:
        model_key: Key from MODEL_REGISTRY

    Returns:
        ModelInfo object

    Raises:
        KeyError: If model_key not found in registry
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Model '{model_key}' not found in registry. "
                      f"Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_key]


def get_all_model_keys() -> List[str]:
    """Get list of all available model keys."""
    return list(MODEL_REGISTRY.keys())


def get_models_by_category(category: str) -> Dict[str, ModelInfo]:
    """
    Get all models in a specific category.

    Args:
        category: Category name ('fast', 'balanced', 'quality', 'multilingual')

    Returns:
        Dictionary of model_key -> ModelInfo for matching category
    """
    return {
        key: info for key, info in MODEL_REGISTRY.items()
        if info.category == category
    }


def get_default_comparison_set() -> List[str]:
    """
    Get a default set of models for comparison.
    Includes representative models across categories for balanced evaluation.

    Returns:
        List of model keys
    """
    return [
        'minilm-l6',        # Fast baseline
        'mpnet-base',       # Balanced - recommended
        'gte-base',         # Quality - MTEB top performer
        'bge-base',         # Quality - alternative high performer
    ]


def get_comprehensive_comparison_set() -> List[str]:
    """
    Get a comprehensive set including larger/SOTA models.
    Warning: These models are slower and require more memory.

    Returns:
        List of model keys
    """
    return [
        'minilm-l6',        # Fast baseline
        'mpnet-base',       # Balanced
        'gte-base',         # Quality base
        'gte-large',        # Quality large
        'bge-base',         # Quality base
        'bge-large',        # Quality large
        'e5-base',          # Quality base
        'e5-large',         # Quality large
    ]


def print_model_registry():
    """Print all available models in a formatted table."""
    print("\n" + "=" * 80)
    print("AVAILABLE EMBEDDING MODELS")
    print("=" * 80)

    categories = ['fast', 'balanced', 'quality', 'multilingual']

    for category in categories:
        models = get_models_by_category(category)
        if not models:
            continue

        print(f"\n{category.upper()} Models:")
        print("-" * 80)

        for key, info in models.items():
            print(f"  [{key}]")
            print(f"    Name:        {info.name}")
            print(f"    Model ID:    {info.model_id}")
            print(f"    Dimension:   {info.dimension}")
            print(f"    Description: {info.description}")
            print()

    print("=" * 80)


if __name__ == "__main__":
    # Display all available models
    print_model_registry()

    print("\nDefault comparison set:")
    for key in get_default_comparison_set():
        info = get_model_info(key)
        print(f"  - {key}: {info.name} ({info.dimension}d)")
