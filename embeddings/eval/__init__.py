"""
Evaluation module for embedding models.
Provides evaluation, comparison, threshold tuning, and model registry functionality.
"""

from eval.evaluator import SimilarityEvaluator, run_evaluation
from eval.model_registry import (
    MODEL_REGISTRY,
    get_model_info,
    get_all_model_keys,
    get_models_by_category,
    get_default_comparison_set,
    get_comprehensive_comparison_set,
    print_model_registry
)
from eval.model_comparison import (
    ModelComparison,
    compare_default_models,
    compare_all_models,
    compare_category
)
from eval.threshold_tuner import (
    ThresholdTuner,
    tune_default_models,
    tune_all_models,
    tune_specific_models
)

__all__ = [
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
]
