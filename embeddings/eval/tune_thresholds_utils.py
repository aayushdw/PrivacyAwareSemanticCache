"""
Utility functions for threshold tuning.
Provides convenient workflows for tuning similarity thresholds for models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.threshold_tuner import (
    ThresholdTuner,
    tune_default_models,
    tune_all_models,
    tune_specific_models
)
from eval.model_registry import get_default_comparison_set, get_models_by_category


def run_tune_default_models():
    """Tune thresholds for default comparison set."""
    print("\n" + "="*100)
    print("TUNING: Default Model Set")
    print("="*100)

    # Tune default set with 5000 samples
    tuner = tune_default_models(max_samples=5000, metric='f1')

    # Get optimal thresholds
    thresholds = tuner.get_optimal_thresholds()
    print("\nOptimal thresholds:")
    for model, threshold in thresholds.items():
        print(f"  {model}: {threshold:.3f}")


def run_tune_specific_models():
    """Tune specific models of your choice."""
    print("\n" + "="*100)
    print("TUNING: Specific Models")
    print("="*100)

    # Choose models to tune
    models = ['minilm-l6', 'mpnet-base', 'gte-base', 'bge-large']

    tuner = tune_specific_models(
        model_keys=models,
        max_samples=5000,
        metric='f1'
    )


def run_tune_with_different_metrics():
    """Tune optimizing for different metrics."""
    print("\n" + "="*100)
    print("TUNING: Optimizing Different Metrics")
    print("="*100)

    models = ['minilm-l6', 'mpnet-base']

    # Optimize for F1 score (balanced)
    print("\n--- Optimizing for F1 Score ---")
    tuner_f1 = ThresholdTuner(max_samples=3000)
    tuner_f1.tune_models(models, metric='f1')

    # Optimize for precision (minimize false positives)
    print("\n--- Optimizing for Precision ---")
    tuner_precision = ThresholdTuner(max_samples=3000)
    tuner_precision.tune_models(models, metric='precision')

    # Optimize for recall (minimize false negatives)
    print("\n--- Optimizing for Recall ---")
    tuner_recall = ThresholdTuner(max_samples=3000)
    tuner_recall.tune_models(models, metric='recall')

    # Compare results
    print("\n" + "="*100)
    print("COMPARISON: F1 vs Precision vs Recall Optimization")
    print("="*100)

    for model in models:
        f1_threshold = tuner_f1.get_optimal_thresholds()[model]
        prec_threshold = tuner_precision.get_optimal_thresholds()[model]
        rec_threshold = tuner_recall.get_optimal_thresholds()[model]

        print(f"\n{model}:")
        print(f"  F1-optimized threshold:        {f1_threshold:.3f}")
        print(f"  Precision-optimized threshold: {prec_threshold:.3f}")
        print(f"  Recall-optimized threshold:    {rec_threshold:.3f}")


def run_tune_by_category():
    """Tune all models in a specific category."""
    print("\n" + "="*100)
    print("TUNING: All Quality Models")
    print("="*100)

    # Get all quality models
    quality_models = list(get_models_by_category('quality').keys())
    print(f"Tuning {len(quality_models)} quality models: {quality_models}")

    tuner = ThresholdTuner(max_samples=3000)
    tuner.tune_models(quality_models, metric='f1')
    tuner.save_results('threshold_tuning_quality_models.json')


def run_tune_with_different_sample_sizes():
    """Compare tuning with different sample sizes."""
    print("\n" + "="*100)
    print("TUNING: Different Sample Sizes")
    print("="*100)

    model = 'mpnet-base'
    sample_sizes = [1000, 3000, 5000, 10000]

    results = {}
    for size in sample_sizes:
        print(f"\n--- Tuning with {size} samples ---")
        tuner = ThresholdTuner(max_samples=size)
        tuner.tune_models([model], metric='f1')
        results[size] = tuner.get_optimal_thresholds()[model]

    # Compare results
    print("\n" + "="*100)
    print(f"COMPARISON: Optimal Threshold vs Sample Size for {model}")
    print("="*100)
    for size, threshold in results.items():
        print(f"  {size:5d} samples: {threshold:.3f}")


def run_programmatic_usage():
    """Programmatic usage for custom workflows."""
    print("\n" + "="*100)
    print("TUNING: Programmatic Usage")
    print("="*100)

    from dataset import load_train_dataset

    # Load and prepare data
    df = load_train_dataset()
    df_sample = df.sample(n=2000, random_state=42)

    # Create tuner
    tuner = ThresholdTuner(max_samples=2000)

    # Tune individual models
    print("\nTuning individual models with custom data...")
    models = ['minilm-l6', 'mpnet-base']

    for model in models:
        result = tuner.tune_model(model, df_sample, metric='f1')
        tuner.tuning_results.append(result)

    # Save results
    tuner.save_results('custom_tuning_results.json')

    # Use tuned thresholds in evaluation
    print("\nOptimal thresholds for use in evaluation:")
    thresholds = tuner.get_optimal_thresholds()
    for model_key, threshold in thresholds.items():
        print(f"  {model_key}: {threshold:.3f}")


def run_tune_and_evaluate():
    """Tune on train, then evaluate on validation."""
    print("\n" + "="*100)
    print("WORKFLOW: Tune on Train, Evaluate on Validation")
    print("="*100)

    from dataset import load_train_dataset, load_val_dataset
    from eval.evaluator import SimilarityEvaluator
    from embedding_engine import EmbeddingEngine
    from eval.model_registry import get_model_info

    model_key = 'mpnet-base'

    # Step 1: Tune on training data
    print("\n--- Step 1: Tuning on training data ---")
    train_df = load_train_dataset()
    tuner = ThresholdTuner(max_samples=5000)
    result = tuner.tune_model(model_key, train_df.head(5000), metric='f1')

    optimal_threshold = result['optimal_threshold']
    print(f"\nOptimal threshold found: {optimal_threshold:.3f}")

    # Step 2: Evaluate on validation data with tuned threshold
    print("\n--- Step 2: Evaluating on validation data ---")
    val_df = load_val_dataset()

    model_info = get_model_info(model_key)
    engine = EmbeddingEngine(model_name=model_info.model_id)
    evaluator = SimilarityEvaluator(engine, similarity_threshold=optimal_threshold)

    metrics = evaluator.evaluate_dataset(val_df, max_samples=1000)
    evaluator.print_evaluation_report(metrics)


def run_tune_all_models():
    """Tune all 25 models in the registry (comprehensive)."""
    print("\n" + "="*100)
    print("TUNING: All Models in Registry")
    print("="*100)

    print("\nWARNING: This will tune all 25 models and may take significant time!")
    print("Recommendation: Use fewer samples (3000) for faster completion.")

    # Get all model keys
    from eval.model_registry import get_all_model_keys
    all_models = get_all_model_keys()

    print(f"\nFound {len(all_models)} models to tune:")
    for i, model in enumerate(all_models, 1):
        print(f"  {i}. {model}")

    # Tune all models with reduced samples for speed
    print(f"\nTuning all models with 3000 samples each...")
    tuner = tune_all_models(max_samples=3000, metric='f1')

    # Get and display all thresholds
    print("\n" + "="*100)
    print("OPTIMAL THRESHOLDS FOR ALL MODELS")
    print("="*100)

    thresholds = tuner.get_optimal_thresholds()

    # Group by category
    from eval.model_registry import get_model_info

    by_category = {}
    for model_key, threshold in thresholds.items():
        info = get_model_info(model_key)
        if info.category not in by_category:
            by_category[info.category] = []
        by_category[info.category].append((model_key, info.name, threshold))

    # Print grouped results
    for category in ['fast', 'balanced', 'quality', 'multilingual']:
        if category in by_category:
            print(f"\n{category.upper()} Models:")
            print("-" * 70)
            for model_key, name, threshold in sorted(by_category[category], key=lambda x: x[2]):
                print(f"  {model_key:30} {name:25} {threshold:.3f}")

    # Summary statistics
    threshold_values = list(thresholds.values())
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print(f"Mean threshold:   {sum(threshold_values)/len(threshold_values):.3f}")
    print(f"Median threshold: {sorted(threshold_values)[len(threshold_values)//2]:.3f}")
    print(f"Min threshold:    {min(threshold_values):.3f}")
    print(f"Max threshold:    {max(threshold_values):.3f}")
    print(f"Range:            {max(threshold_values) - min(threshold_values):.3f}")

    print(f"\n✓ Results saved to: embeddings/eval/results/threshold_tuning_all_models.json")


def main():
    print("\nRunning: Tune Default Models")
    # run_tune_default_models()

    # Uncomment to run other utilities:
    # run_tune_specific_models()
    # run_tune_with_different_metrics()
    # run_tune_by_category()
    # run_tune_with_different_sample_sizes()
    # run_programmatic_usage()
    # run_tune_and_evaluate()
    run_tune_all_models()  # Warning: Takes significant time!


if __name__ == "__main__":
    main()
