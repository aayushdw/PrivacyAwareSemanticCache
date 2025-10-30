"""
Embeddings package for Privacy Aware Semantic Cache.
Provides embedding engine functionality, evaluation, and model comparison.
"""

from embeddings.embedding_engine import EmbeddingEngine
from embeddings.dataset import (
    load_dataset,
    get_csv_file_path,
    load_train_dataset,
    load_val_dataset,
    load_test_dataset,
    load_full_dataset
)
from embeddings.eval.evaluator import SimilarityEvaluator, run_evaluation
from embeddings.eval.model_registry import (
    MODEL_REGISTRY,
    get_model_info,
    get_all_model_keys,
    get_models_by_category,
    get_default_comparison_set,
    get_comprehensive_comparison_set,
    print_model_registry
)
from embeddings.eval.model_comparison import (
    ModelComparison,
    compare_default_models,
    compare_all_models,
    compare_category
)
from embeddings.eval.threshold_tuner import (
    ThresholdTuner,
    tune_default_models,
    tune_all_models,
    tune_specific_models
)

__all__ = [
    'EmbeddingEngine',
    'SimilarityEvaluator',
    'run_evaluation',
    'MODEL_REGISTRY',
    'get_model_info',
    'get_all_model_keys',
    'get_models_by_category',
    'get_default_comparison_set',
    'get_comprehensive_comparison_set',
    'print_model_registry',
    'ModelComparison',
    'compare_default_models',
    'compare_all_models',
    'compare_category',
    'ThresholdTuner',
    'tune_default_models',
    'tune_all_models',
    'tune_specific_models',
    'load_dataset',
    'get_csv_file_path',
    'load_train_dataset',
    'load_val_dataset',
    'load_test_dataset',
    'load_full_dataset'
]
