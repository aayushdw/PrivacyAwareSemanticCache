"""
Multi-model comparison framework for embedding models.
Evaluates multiple models and compares their performance.
"""

import time
import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_engine import EmbeddingEngine
from eval.evaluator import SimilarityEvaluator
from dataset import load_dataset
from eval.model_registry import (
    get_model_info,
    get_default_comparison_set,
    get_all_model_keys,
    MODEL_REGISTRY
)


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class ModelComparison:
    """
    Compares multiple embedding models on the same evaluation task.
    """

    def __init__(self, threshold: float = 0.85, max_samples: int = 1000):
        """
        Initialize the model comparison framework.

        Args:
            threshold: Similarity threshold for classification
            max_samples: Maximum number of samples to evaluate
        """
        self.threshold = threshold
        self.max_samples = max_samples
        self.results = []

    def evaluate_model(self, model_key: str, df: pd.DataFrame) -> Dict:
        """
        Evaluate a single model.

        Args:
            model_key: Key from model registry
            df: Dataset DataFrame

        Returns:
            Dictionary containing model info, metrics, and timing
        """
        model_info = get_model_info(model_key)

        print(f"\n{'='*80}")
        print(f"Evaluating: {model_info.name} ({model_key})")
        print(f"Model ID: {model_info.model_id}")
        print(f"{'='*80}")

        # Initialize engine and evaluator
        start_time = time.time()
        try:
            engine = EmbeddingEngine(model_name=model_info.model_id)
            evaluator = SimilarityEvaluator(engine, similarity_threshold=self.threshold)

            # Run evaluation
            eval_start = time.time()
            metrics = evaluator.evaluate_dataset(df, max_samples=self.max_samples)
            eval_time = time.time() - eval_start

            total_time = time.time() - start_time

            # Compile results
            result = {
                'model_key': model_key,
                'model_name': model_info.name,
                'model_id': model_info.model_id,
                'category': model_info.category,
                'dimension': model_info.dimension,
                'loading_time': start_time,
                'evaluation_time': eval_time,
                'total_time': total_time,
                'metrics': metrics,
                'success': True,
                'error': None
            }

            print(f"\n✓ Completed in {total_time:.2f}s (evaluation: {eval_time:.2f}s)")

            return result

        except Exception as e:
            print(f"\n✗ Error evaluating model: {e}")
            return {
                'model_key': model_key,
                'model_name': model_info.name,
                'model_id': model_info.model_id,
                'category': model_info.category,
                'dimension': model_info.dimension,
                'success': False,
                'error': str(e)
            }

    def compare_models(self, model_keys: List[str], df: Optional[pd.DataFrame] = None):
        """
        Compare multiple models.

        Args:
            model_keys: List of model keys to compare
            df: Optional DataFrame. If None, loads from default path
        """
        if df is None:
            print("Loading dataset...")
            df = load_dataset()
            print(f"Dataset loaded: {len(df)} rows")

        print(f"\nComparing {len(model_keys)} models on {self.max_samples} samples")
        print(f"Threshold: {self.threshold}")

        self.results = []
        for model_key in model_keys:
            result = self.evaluate_model(model_key, df)
            self.results.append(result)

        self._print_comparison_report()

    def _print_comparison_report(self):
        """Print a comprehensive comparison report."""
        if not self.results:
            print("No results to display")
            return

        successful_results = [r for r in self.results if r.get('success', False)]

        if not successful_results:
            print("\n⚠️  No models evaluated successfully")
            return

        print("\n" + "=" * 100)
        print("MODEL COMPARISON REPORT")
        print("=" * 100)

        # Create comparison table
        comparison_data = []
        for result in successful_results:
            metrics = result['metrics']
            comparison_data.append({
                'Model': result['model_name'],
                'Category': result['category'],
                'Dim': result['dimension'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1': f"{metrics['f1_score']:.4f}",
                'Time (s)': f"{result['evaluation_time']:.2f}"
            })

        df_comparison = pd.DataFrame(comparison_data)

        # Sort by F1 score
        df_comparison['F1_numeric'] = df_comparison['F1'].astype(float)
        df_comparison = df_comparison.sort_values('F1_numeric', ascending=False)
        df_comparison = df_comparison.drop('F1_numeric', axis=1)

        print("\n" + df_comparison.to_string(index=False))

        # Find best model
        best_result = max(successful_results, key=lambda x: x['metrics']['f1_score'])
        print(f"\n🏆 Best Model (by F1): {best_result['model_name']} "
              f"(F1: {best_result['metrics']['f1_score']:.4f})")

        # Performance insights
        print("\n" + "-" * 100)
        print("PERFORMANCE INSIGHTS")
        print("-" * 100)

        fastest = min(successful_results, key=lambda x: x['evaluation_time'])
        print(f"⚡ Fastest: {fastest['model_name']} ({fastest['evaluation_time']:.2f}s)")

        most_accurate = max(successful_results, key=lambda x: x['metrics']['accuracy'])
        print(f"🎯 Most Accurate: {most_accurate['model_name']} "
              f"({most_accurate['metrics']['accuracy']:.4f})")

        best_precision = max(successful_results, key=lambda x: x['metrics']['precision'])
        print(f"📊 Best Precision: {best_precision['model_name']} "
              f"({best_precision['metrics']['precision']:.4f})")

        best_recall = max(successful_results, key=lambda x: x['metrics']['recall'])
        print(f"🔍 Best Recall: {best_recall['model_name']} "
              f"({best_recall['metrics']['recall']:.4f})")

        print("\n" + "=" * 100)

    def save_results(self, output_path: str):
        """
        Save comparison results to JSON file.

        Args:
            output_path: Path to save JSON file (relative to eval/results or absolute path)
        """
        # If path is relative and doesn't contain a directory, save to results folder
        if not os.path.isabs(output_path) and os.path.dirname(output_path) == '':
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(results_dir, output_path)

        # Convert numpy types to Python native types for JSON serialization
        results_converted = convert_numpy_types(self.results)

        with open(output_path, 'w') as f:
            json.dump(results_converted, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame.

        Returns:
            DataFrame with comparison results
        """
        successful_results = [r for r in self.results if r.get('success', False)]

        data = []
        for result in successful_results:
            metrics = result['metrics']
            row = {
                'model_key': result['model_key'],
                'model_name': result['model_name'],
                'category': result['category'],
                'dimension': result['dimension'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'specificity': metrics['specificity'],
                'true_positives': metrics['true_positives'],
                'true_negatives': metrics['true_negatives'],
                'false_positives': metrics['false_positives'],
                'false_negatives': metrics['false_negatives'],
                'evaluation_time': result['evaluation_time'],
                'total_time': result['total_time']
            }
            data.append(row)

        return pd.DataFrame(data)


def compare_default_models(threshold: float = 0.85, max_samples: int = 1000):
    """
    Compare the default set of models.

    Args:
        threshold: Similarity threshold
        max_samples: Maximum samples to evaluate
    """
    model_keys = get_default_comparison_set()
    comparison = ModelComparison(threshold=threshold, max_samples=max_samples)
    comparison.compare_models(model_keys)
    return comparison


def compare_all_models(threshold: float = 0.85, max_samples: int = 1000):
    """
    Compare all available models in the registry.

    Args:
        threshold: Similarity threshold
        max_samples: Maximum samples to evaluate
    """
    model_keys = get_all_model_keys()
    comparison = ModelComparison(threshold=threshold, max_samples=max_samples)
    comparison.compare_models(model_keys)
    return comparison


def compare_category(category: str, threshold: float = 0.85, max_samples: int = 1000):
    """
    Compare models within a specific category.

    Args:
        category: Category name ('fast', 'balanced', 'quality', 'multilingual')
        threshold: Similarity threshold
        max_samples: Maximum samples to evaluate
    """
    model_keys = [k for k, v in MODEL_REGISTRY.items() if v.category == category]
    print(f"\nComparing {category.upper()} models: {model_keys}")

    comparison = ModelComparison(threshold=threshold, max_samples=max_samples)
    comparison.compare_models(model_keys)
    return comparison


if __name__ == "__main__":
    # Compare default set of models
    print("Running model comparison with default set...")
    comparison = compare_default_models(threshold=0.85, max_samples=1000)

    # Optionally save results
    # comparison.save_results('model_comparison_results.json')
