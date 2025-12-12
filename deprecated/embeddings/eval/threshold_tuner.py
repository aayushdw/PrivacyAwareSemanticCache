"""
Threshold tuning for embedding models.
Finds optimal similarity thresholds for each model on training data.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, confusion_matrix

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_engine import EmbeddingEngine
from dataset import load_train_dataset
from eval.model_registry import get_model_info, get_all_model_keys


class ThresholdTuner:
    """
    Tunes similarity thresholds for embedding models on training data.
    """

    def __init__(self, max_samples: int = 5000):
        """
        Initialize threshold tuner.

        Args:
            max_samples: Maximum samples to use from training set for tuning
        """
        self.max_samples = max_samples
        self.tuning_results = {}

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataset by removing rows with missing, empty, or invalid data.

        Args:
            df: DataFrame with question pairs

        Returns:
            Cleaned DataFrame
        """
        original_len = len(df)

        # Remove rows with missing values in key columns
        df = df.dropna(subset=['question1', 'question2', 'is_duplicate'])

        # Remove rows where is_duplicate is not 0 or 1 (invalid labels)
        df = df[df['is_duplicate'].isin([0, 1])]

        # Remove rows with empty strings
        df = df[df['question1'].str.strip() != '']
        df = df[df['question2'].str.strip() != '']

        # Reset index
        df = df.reset_index(drop=True)

        cleaned_len = len(df)
        if cleaned_len < original_len:
            print(f"ℹ️  Cleaned dataset: removed {original_len - cleaned_len} rows with missing/empty/invalid data")
            print(f"   Valid rows: {cleaned_len}/{original_len} ({cleaned_len/original_len*100:.1f}%)")

        return df

    def compute_and_format_confusion_matrix(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Tuple[Dict, str]:
        """
        Compute confusion matrix and return both dictionary and formatted string.

        Args:
            predictions: Predicted labels (0 or 1)
            ground_truth: True labels (0 or 1)

        Returns:
            Tuple of (confusion_matrix_dict, formatted_string)
        """
        cm = confusion_matrix(ground_truth, predictions, labels=[0, 1])

        # Extract values: [[TN, FP], [FN, TP]]
        tn, fp = cm[0]
        fn, tp = cm[1]

        cm_dict = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total': int(tn + fp + fn + tp)
        }

        # Create formatted string for display
        total = tn + fp + fn + tp
        cm_str = f"""
Confusion Matrix:
                 Predicted
                 Neg    Pos
Actual  Neg      {tn:6d} {fp:6d}  (TN: {tn:6d}, FP: {fp:6d})
        Pos      {fn:6d} {tp:6d}  (FN: {fn:6d}, TP: {tp:6d})

True Negatives  (TN): {tn:6d} ({tn/total*100:5.2f}%)
False Positives (FP): {fp:6d} ({fp/total*100:5.2f}%)
False Negatives (FN): {fn:6d} ({fn/total*100:5.2f}%)
True Positives  (TP): {tp:6d} ({tp/total*100:5.2f}%)
"""

        return cm_dict, cm_str

    def compute_similarities(self, model_key: str, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute similarity scores for all question pairs.

        Args:
            model_key: Model key from registry
            df: DataFrame with question pairs

        Returns:
            Tuple of (similarity_scores, valid_indices) - indices of non-NaN entries
        """
        model_info = get_model_info(model_key)

        print(f"\nLoading model: {model_info.name}")
        engine = EmbeddingEngine(model_name=model_info.model_id)

        # Extract questions
        questions1 = df['question1'].tolist()
        questions2 = df['question2'].tolist()

        # Generate embeddings
        print("Generating embeddings for question1...")
        embeddings1 = engine.encode_batch(
            questions1,
            normalize=True,
            show_progress_bar=True
        )

        print("Generating embeddings for question2...")
        embeddings2 = engine.encode_batch(
            questions2,
            normalize=True,
            show_progress_bar=True
        )

        # Compute similarities
        print("Computing similarity scores...")
        similarities = np.array([
            np.dot(emb1, emb2)
            for emb1, emb2 in zip(embeddings1, embeddings2)
        ])

        # Check for NaN values and filter them out
        valid_mask = ~np.isnan(similarities)
        num_invalid = np.sum(~valid_mask)

        if num_invalid > 0:
            print(f"⚠️  Warning: Found {num_invalid} NaN similarity scores ({num_invalid/len(similarities)*100:.2f}%)")
            print(f"   These data points will be excluded from threshold tuning.")

        valid_indices = np.where(valid_mask)[0]

        return similarities, valid_indices

    def find_optimal_threshold(
        self,
        similarities: np.ndarray,
        ground_truth: np.ndarray,
        metric: str = 'f1',
        min_precision: Optional[float] = None
    ) -> Tuple[float, Dict]:
        """
        Find optimal threshold that maximizes the specified metric.

        Args:
            similarities: Similarity scores
            ground_truth: True labels
            metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
            min_precision: Minimum precision constraint. If set, only considers thresholds
                          where precision >= min_precision, then maximizes the metric.
                          Useful for "precision-first" optimization (e.g., min_precision=0.90)

        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        # Try thresholds from 0.5 to 0.99
        thresholds = np.arange(0.50, 1.00, 0.01)

        best_score = -1
        best_threshold = None
        best_metrics = None

        if min_precision is not None:
            print(f"\nSearching for optimal threshold (optimizing {metric} with precision >= {min_precision:.2f})...")
        else:
            print(f"\nSearching for optimal threshold (optimizing {metric})...")

        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)

            # Compute metrics
            acc = accuracy_score(ground_truth, predictions)

            # Handle edge cases where all predictions are same class
            if len(np.unique(predictions)) == 1:
                prec = 0.0 if predictions[0] == 1 else 1.0
                rec = 0.0 if predictions[0] == 0 else 1.0
                f1 = 0.0
            else:
                from sklearn.metrics import precision_score, recall_score
                prec = precision_score(ground_truth, predictions, zero_division=0)
                rec = recall_score(ground_truth, predictions, zero_division=0)
                f1 = f1_score(ground_truth, predictions, zero_division=0)

            metrics = {
                'threshold': float(threshold),
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1)
            }

            # Check precision constraint if specified
            if min_precision is not None and prec < min_precision:
                continue  # Skip this threshold if it doesn't meet precision requirement

            # Select best based on chosen metric
            if metric == 'f1':
                score = f1
            elif metric == 'accuracy':
                score = acc
            elif metric == 'precision':
                score = prec
            elif metric == 'recall':
                score = rec
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics

        if best_threshold is None:
            if min_precision is not None:
                raise ValueError(
                    f"No threshold found that meets precision >= {min_precision:.2f}. "
                    f"Try lowering min_precision or check your data quality."
                )
            else:
                raise ValueError("No valid threshold found. Check your data quality.")

        print(f"  Best threshold: {best_threshold:.3f}")
        print(f"  Best {metric}: {best_score:.4f}")
        if min_precision is not None:
            print(f"  Precision: {best_metrics['precision']:.4f} (>= {min_precision:.2f})")

        # Compute confusion matrix at optimal threshold
        optimal_predictions = (similarities >= best_threshold).astype(int)
        cm_dict, cm_str = self.compute_and_format_confusion_matrix(optimal_predictions, ground_truth)
        best_metrics['confusion_matrix'] = cm_dict
        best_metrics['confusion_matrix_str'] = cm_str

        return best_threshold, best_metrics

    def tune_model(
        self,
        model_key: str,
        df: pd.DataFrame,
        metric: str = 'f1',
        clean_data: bool = False,
        min_precision: Optional[float] = None
    ) -> Dict:
        """
        Tune threshold for a single model.

        Args:
            model_key: Model key from registry
            df: Training DataFrame
            metric: Metric to optimize
            clean_data: Whether to clean the dataset before tuning (default: False, assumes already cleaned)
            min_precision: Minimum precision constraint (e.g., 0.90 for 90% precision)

        Returns:
            Dictionary with tuning results
        """
        model_info = get_model_info(model_key)

        print(f"\n{'='*80}")
        print(f"Tuning threshold for: {model_info.name} ({model_key})")
        print(f"{'='*80}")

        try:
            # Clean dataset if requested
            if clean_data:
                df = self.clean_dataset(df)

            # Compute similarities and get valid indices
            similarities, valid_indices = self.compute_similarities(model_key, df)

            # Filter out NaN values
            valid_similarities = similarities[valid_indices]
            valid_ground_truth = df['is_duplicate'].values[valid_indices]

            # Check if we have enough valid data
            if len(valid_similarities) == 0:
                raise ValueError("All similarity scores are NaN. Check your dataset for empty or invalid text.")

            if len(valid_similarities) < 100:
                print(f"⚠️  Warning: Only {len(valid_similarities)} valid samples remaining. Results may be unreliable.")

            # Find optimal threshold
            optimal_threshold, metrics = self.find_optimal_threshold(
                valid_similarities,
                valid_ground_truth,
                metric=metric,
                min_precision=min_precision
            )

            # Store additional info
            result = {
                'model_key': model_key,
                'model_name': model_info.name,
                'model_id': model_info.model_id,
                'dimension': model_info.dimension,
                'category': model_info.category,
                'optimal_threshold': optimal_threshold,
                'metrics_at_optimal': metrics,
                'tuning_samples': len(df),
                'valid_samples': len(valid_similarities),
                'invalid_samples': len(df) - len(valid_similarities),
                'optimization_metric': metric,
                'min_precision_constraint': min_precision,
                'success': True,
                'error': None
            }

            # Compute statistics on similarity distribution (using valid data only)
            duplicate_sims = valid_similarities[valid_ground_truth == 1]
            non_duplicate_sims = valid_similarities[valid_ground_truth == 0]

            result['similarity_stats'] = {
                'duplicates': {
                    'mean': float(np.mean(duplicate_sims)),
                    'std': float(np.std(duplicate_sims)),
                    'min': float(np.min(duplicate_sims)),
                    'max': float(np.max(duplicate_sims))
                },
                'non_duplicates': {
                    'mean': float(np.mean(non_duplicate_sims)),
                    'std': float(np.std(non_duplicate_sims)),
                    'min': float(np.min(non_duplicate_sims)),
                    'max': float(np.max(non_duplicate_sims))
                }
            }

            print(f"\n✓ Tuning completed successfully")
            print(f"   Valid samples used: {len(valid_similarities)}/{len(df)}")
            self._print_tuning_summary(result)

            return result

        except Exception as e:
            print(f"\n✗ Error tuning model: {e}")
            return {
                'model_key': model_key,
                'model_name': model_info.name,
                'success': False,
                'error': str(e)
            }

    def _print_tuning_summary(self, result: Dict):
        """Print summary of tuning results."""
        print(f"\n--- Tuning Summary ---")
        print(f"Optimal Threshold: {result['optimal_threshold']:.3f}")
        metrics = result['metrics_at_optimal']
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")

        # Display confusion matrix if available
        if 'confusion_matrix_str' in metrics:
            print(metrics['confusion_matrix_str'])

        if 'similarity_stats' in result:
            stats = result['similarity_stats']
            print(f"Similarity Distribution:")
            print(f"  Duplicates:     mean={stats['duplicates']['mean']:.3f}, "
                  f"std={stats['duplicates']['std']:.3f}")
            print(f"  Non-duplicates: mean={stats['non_duplicates']['mean']:.3f}, "
                  f"std={stats['non_duplicates']['std']:.3f}")

    def tune_models(
        self,
        model_keys: List[str],
        df: Optional[pd.DataFrame] = None,
        metric: str = 'f1',
        min_precision: Optional[float] = None
    ):
        """
        Tune thresholds for multiple models.

        Args:
            model_keys: List of model keys to tune
            df: Training DataFrame. If None, loads from dataset.
            metric: Metric to optimize
            min_precision: Minimum precision constraint (e.g., 0.90 for 90% precision)
        """
        if df is None:
            print(f"Loading training dataset...")
            df = load_train_dataset()
            print(f"Total training samples: {len(df)}")

        # Clean dataset first
        df = self.clean_dataset(df)

        # Check if we have enough data after cleaning
        if len(df) == 0:
            raise ValueError("Dataset is empty after cleaning. Please check your data quality.")

        # Limit samples if needed
        if len(df) > self.max_samples:
            print(f"Using {self.max_samples} samples for tuning (randomly sampled)")
            df = df.sample(n=self.max_samples, random_state=42).reset_index(drop=True)
        else:
            print(f"Using all {len(df)} samples for tuning")

        self.tuning_results = []

        for model_key in model_keys:
            result = self.tune_model(model_key, df, metric=metric, min_precision=min_precision)
            self.tuning_results.append(result)

        self._print_comparison_report()

    def _print_comparison_report(self):
        """Print comparison report of all tuned models."""
        successful_results = [r for r in self.tuning_results if r.get('success', False)]

        if not successful_results:
            print("\n⚠️  No models tuned successfully")
            return

        print("\n" + "="*100)
        print("THRESHOLD TUNING COMPARISON")
        print("="*100)

        # Create comparison table
        comparison_data = []
        for result in successful_results:
            metrics = result['metrics_at_optimal']
            comparison_data.append({
                'Model': result['model_name'],
                'Category': result['category'],
                'Threshold': f"{result['optimal_threshold']:.3f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1': f"{metrics['f1_score']:.4f}"
            })

        df_comparison = pd.DataFrame(comparison_data)
        df_comparison['F1_numeric'] = df_comparison['F1'].astype(float)
        df_comparison = df_comparison.sort_values('F1_numeric', ascending=False)
        df_comparison = df_comparison.drop('F1_numeric', axis=1)

        print("\n" + df_comparison.to_string(index=False))

        # Best model
        best_result = max(successful_results,
                         key=lambda x: x['metrics_at_optimal']['f1_score'])
        print(f"\n🏆 Best Model: {best_result['model_name']} "
              f"(Threshold: {best_result['optimal_threshold']:.3f}, "
              f"F1: {best_result['metrics_at_optimal']['f1_score']:.4f})")

        print("\n" + "="*100)

    def save_results(self, output_path: str = 'threshold_tuning_results.json'):
        """
        Save tuning results to JSON file.

        Args:
            output_path: Path to save results (relative paths go to eval/results/)
        """
        # If path is relative and doesn't contain a directory, save to results folder
        if not os.path.isabs(output_path) and os.path.dirname(output_path) == '':
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(results_dir, output_path)

        with open(output_path, 'w') as f:
            json.dump(self.tuning_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    def get_optimal_thresholds(self) -> Dict[str, float]:
        """
        Get dictionary of model_key -> optimal_threshold.

        Returns:
            Dictionary mapping model keys to their optimal thresholds
        """
        return {
            result['model_key']: result['optimal_threshold']
            for result in self.tuning_results
            if result.get('success', False)
        }


def tune_default_models(max_samples: int = 5000, metric: str = 'f1', min_precision: Optional[float] = None):
    """
    Tune thresholds for default comparison set.

    Args:
        max_samples: Maximum samples to use
        metric: Metric to optimize
        min_precision: Minimum precision constraint (e.g., 0.90 for 90% precision)
    """
    from eval.model_registry import get_default_comparison_set

    model_keys = get_default_comparison_set()
    tuner = ThresholdTuner(max_samples=max_samples)
    tuner.tune_models(model_keys, metric=metric, min_precision=min_precision)
    tuner.save_results('threshold_tuning_default.json')
    return tuner


def tune_all_models(max_samples: int = 3000, metric: str = 'f1', min_precision: Optional[float] = None):
    """
    Tune thresholds for all models in registry.

    Args:
        max_samples: Maximum samples to use (lower default for speed)
        metric: Metric to optimize
        min_precision: Minimum precision constraint (e.g., 0.90 for 90% precision)
    """
    model_keys = get_all_model_keys()
    tuner = ThresholdTuner(max_samples=max_samples)
    tuner.tune_models(model_keys, metric=metric, min_precision=min_precision)
    tuner.save_results('threshold_tuning_all_models.json')
    return tuner


def tune_specific_models(
    model_keys: List[str],
    max_samples: int = 5000,
    metric: str = 'f1',
    min_precision: Optional[float] = None
):
    """
    Tune thresholds for specific models.

    Args:
        model_keys: List of model keys to tune
        max_samples: Maximum samples to use
        metric: Metric to optimize
        min_precision: Minimum precision constraint (e.g., 0.90 for 90% precision)
    """
    tuner = ThresholdTuner(max_samples=max_samples)
    tuner.tune_models(model_keys, metric=metric, min_precision=min_precision)
    tuner.save_results(f'threshold_tuning_custom.json')
    return tuner


if __name__ == "__main__":
    # Example: Tune default models
    print("Tuning thresholds for default model set...")
    tuner = tune_default_models(max_samples=5000, metric='f1')

    # Print optimal thresholds
    print("\n" + "="*80)
    print("OPTIMAL THRESHOLDS")
    print("="*80)
    for model_key, threshold in tuner.get_optimal_thresholds().items():
        print(f"  {model_key}: {threshold:.3f}")
