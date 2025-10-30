"""
Example script demonstrating model comparison functionality.
Shows different ways to compare embedding models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.model_comparison import (
    ModelComparison,
    compare_default_models,
    compare_all_models,
    compare_category
)
from eval.model_registry import print_model_registry


def default_comparison():
    """Example 1: Compare default set of models."""
    print("\n" + "=" * 100)
    print("EXAMPLE 1: Compare Default Set of Models")
    print("=" * 100)

    comparison = compare_default_models(threshold=0.85, max_samples=1000)

    # Get results as DataFrame
    df_results = comparison.get_results_dataframe()
    print("\nResults DataFrame:")
    print(df_results[['model_name', 'f1_score', 'accuracy', 'evaluation_time']])

    # Save results
    comparison.save_results('results_default_models.json')


def compare_threshold_sweep():
    """Example 4: Test different thresholds on a single model."""
    print("\n" + "=" * 100)
    print("EXAMPLE 4: Threshold Sweep for MPNet-Base")
    print("=" * 100)

    thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]
    results_summary = []

    for threshold in thresholds:
        print(f"\n--- Testing threshold: {threshold} ---")
        comparison = ModelComparison(threshold=threshold, max_samples=500)
        comparison.compare_models(['mpnet-base'])

        if comparison.results and comparison.results[0].get('success'):
            metrics = comparison.results[0]['metrics']
            results_summary.append({
                'threshold': threshold,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            })

    print("\n" + "=" * 100)
    print("THRESHOLD SWEEP SUMMARY")
    print("=" * 100)
    for result in results_summary:
        print(f"Threshold {result['threshold']:.2f}: "
              f"Acc={result['accuracy']:.4f}, "
              f"P={result['precision']:.4f}, "
              f"R={result['recall']:.4f}, "
              f"F1={result['f1_score']:.4f}")


def compare_all():
    """Example 5: Compare all available models (takes longer)."""
    print("\n" + "=" * 100)
    print("EXAMPLE 5: Compare ALL Models (This will take a while!)")
    print("=" * 100)

    # Use fewer samples to speed up
    comparison = compare_all_models(threshold=0.88, max_samples=500)

    # Save comprehensive results
    comparison.save_results('results_all_models.json')


def main():
    """Run examples."""
    # Display available models first
    print_model_registry()
    compare_all()


if __name__ == "__main__":
    main()
