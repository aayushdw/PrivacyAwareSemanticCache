"""
Threshold tuning for embedding models.

Finds optimal similarity thresholds that maximize F1 while meeting precision constraints.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .metrics_calculator import ClassificationMetrics, MetricsCalculator


@dataclass
class ThresholdResult:
    """Threshold tuning result with metrics and statistics."""

    threshold: float
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    meets_constraint: bool
    constraint_value: Optional[float]  # The precision constraint used

    # Full metrics at optimal threshold
    metrics: Optional[ClassificationMetrics] = None

    # Similarity statistics
    similarity_stats: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert result to dictionary for logging."""
        result = {
            "optimal_threshold": self.threshold,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "meets_constraint": self.meets_constraint,
            "precision_constraint": self.constraint_value,
        }

        if self.metrics:
            result["confusion_matrix"] = {
                "tp": self.metrics.true_positives,
                "tn": self.metrics.true_negatives,
                "fp": self.metrics.false_positives,
                "fn": self.metrics.false_negatives,
            }

        if self.similarity_stats:
            result["similarity_stats"] = self.similarity_stats

        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ThresholdResult":
        """Create instance from dictionary."""
        metrics = None
        if "confusion_matrix" in data:
            cm = data["confusion_matrix"]
            metrics = ClassificationMetrics(
                threshold=data["optimal_threshold"],
                accuracy=data["accuracy"],
                precision=data["precision"],
                recall=data["recall"],
                f1_score=data["f1_score"],
                true_positives=cm["tp"],
                true_negatives=cm["tn"],
                false_positives=cm["fp"],
                false_negatives=cm["fn"],
            )

        return cls(
            threshold=data["optimal_threshold"],
            f1_score=data["f1_score"],
            precision=data["precision"],
            recall=data["recall"],
            accuracy=data["accuracy"],
            meets_constraint=data["meets_constraint"],
            constraint_value=data.get("precision_constraint"),
            metrics=metrics,
            similarity_stats=data.get("similarity_stats"),
        )


class ThresholdTuner:
    """Tunes similarity thresholds for embedding models with constraint-based optimization."""

    def __init__(
        self,
        threshold_range: Tuple[float, float] = (0.50, 0.99),
        threshold_step: float = 0.01,
    ):
        """Initialize threshold tuner with search range and step."""
        self.threshold_range = threshold_range
        self.threshold_step = threshold_step
        self.calculator = MetricsCalculator()

    def find_optimal_threshold_from_triplets(
        self,
        positive_similarities: np.ndarray,
        negative_similarities: np.ndarray,
        min_precision: Optional[float] = None,
        metric: str = "f1",
    ) -> ThresholdResult:
        """Find optimal threshold from triplet similarities with optional precision constraint."""
        # Convert triplets to classification data
        similarities, labels = self.calculator.triplets_to_classification_data(
            positive_similarities, negative_similarities
        )

        # Compute triplet-specific statistics
        similarity_stats = self.calculator.compute_similarity_stats(
            positive_similarities, negative_similarities
        )

        # Find optimal threshold
        result = self.find_optimal_threshold(
            similarities, labels, min_precision, metric
        )

        # Attach triplet-specific stats
        result.similarity_stats = similarity_stats

        return result

    def find_optimal_threshold(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        min_precision: Optional[float] = None,
        metric: str = "f1",
    ) -> ThresholdResult:
        """Find threshold that maximizes metric while meeting optional precision constraint."""
        # Validate inputs
        similarities = np.asarray(similarities)
        labels = np.asarray(labels)

        # Remove NaN values
        valid_mask = ~np.isnan(similarities)
        if not np.all(valid_mask):
            num_invalid = np.sum(~valid_mask)
            print(f"Warning: Removing {num_invalid} NaN similarity scores")
            similarities = similarities[valid_mask]
            labels = labels[valid_mask]

        if len(similarities) == 0:
            raise ValueError("No valid similarity scores to evaluate")

        best_score = -1.0
        best_threshold = None
        best_metrics = None

        # Search through thresholds
        for threshold in np.arange(
            self.threshold_range[0],
            self.threshold_range[1] + self.threshold_step,
            self.threshold_step,
        ):
            metrics = self.calculator.compute_metrics_at_threshold(
                similarities, labels, threshold
            )

            # Check precision constraint
            if min_precision is not None and metrics.precision < min_precision:
                continue

            # Get score for target metric
            if metric == "f1":
                score = metrics.f1_score
            elif metric == "accuracy":
                score = metrics.accuracy
            elif metric == "recall":
                score = metrics.recall
            elif metric == "precision":
                score = metrics.precision
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics

        # Handle case where no threshold meets constraint
        if best_threshold is None:
            if min_precision is not None:
                raise ValueError(
                    f"No threshold found meeting precision >= {min_precision:.2f}. "
                    "Try lowering the constraint or check data quality."
                )
            else:
                raise ValueError("No valid threshold found. Check data quality.")

        return ThresholdResult(
            threshold=best_threshold,
            f1_score=best_metrics.f1_score,
            precision=best_metrics.precision,
            recall=best_metrics.recall,
            accuracy=best_metrics.accuracy,
            meets_constraint=True,
            constraint_value=min_precision,
            metrics=best_metrics,
        )

    def find_threshold_at_recall(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        target_recall: float,
    ) -> Optional[ThresholdResult]:
        """Find threshold that achieves at least target recall."""
        best_threshold = None
        best_metrics = None
        best_precision = -1.0

        for threshold in np.arange(
            self.threshold_range[0],
            self.threshold_range[1] + self.threshold_step,
            self.threshold_step,
        ):
            metrics = self.calculator.compute_metrics_at_threshold(
                similarities, labels, threshold
            )

            if metrics.recall >= target_recall and metrics.precision > best_precision:
                best_threshold = threshold
                best_metrics = metrics
                best_precision = metrics.precision

        if best_threshold is None:
            return None

        return ThresholdResult(
            threshold=best_threshold,
            f1_score=best_metrics.f1_score,
            precision=best_metrics.precision,
            recall=best_metrics.recall,
            accuracy=best_metrics.accuracy,
            meets_constraint=True,
            constraint_value=target_recall,
            metrics=best_metrics,
        )

    def analyze_threshold_sensitivity(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        center_threshold: float,
        delta: float = 0.05,
    ) -> dict:
        """Analyze how metrics change around a threshold."""
        results = {}

        for offset in [-delta, 0, delta]:
            threshold = center_threshold + offset
            if self.threshold_range[0] <= threshold <= self.threshold_range[1]:
                metrics = self.calculator.compute_metrics_at_threshold(
                    similarities, labels, threshold
                )
                results[f"threshold_{threshold:.3f}"] = metrics.to_dict()

        return results

    def print_tuning_summary(self, result: ThresholdResult):
        """Print summary of tuning results."""
        print(f"\n{'='*60}")
        print("THRESHOLD TUNING RESULTS")
        print(f"{'='*60}")
        print(f"Optimal Threshold: {result.threshold:.3f}")
        print(f"{'─'*60}")
        print(f"  F1 Score:   {result.f1_score:.4f}")
        print(f"  Precision:  {result.precision:.4f}")
        print(f"  Recall:     {result.recall:.4f}")
        print(f"  Accuracy:   {result.accuracy:.4f}")

        if result.constraint_value:
            status = "MET" if result.meets_constraint else "NOT MET"
            print(f"\nPrecision Constraint (>= {result.constraint_value:.2f}): {status}")

        if result.metrics:
            print(self.calculator.format_confusion_matrix(result.metrics))

        if result.similarity_stats:
            stats = result.similarity_stats
            if "positive_pairs" in stats and "negative_pairs" in stats:
                print("Similarity Distribution:")
                print(
                    f"  Positive pairs: mean={stats['positive_pairs']['mean']:.3f}, "
                    f"std={stats['positive_pairs']['std']:.3f}"
                )
                print(
                    f"  Negative pairs: mean={stats['negative_pairs']['mean']:.3f}, "
                    f"std={stats['negative_pairs']['std']:.3f}"
                )
                if "separability" in stats:
                    print(f"  Separability:   {stats['separability']:.3f}")

        print(f"{'='*60}\n")
