"""Metrics calculation for embedding model evaluation with triplet data."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class TripletMetrics:
    """Metrics for triplet-based evaluation."""

    triplet_accuracy: float  # % of triplets where sim(a,p) > sim(a,n)
    mean_positive_sim: float
    mean_negative_sim: float
    mean_margin: float  # avg(sim(a,p) - sim(a,n))
    margin_std: float
    positive_sim_std: float
    negative_sim_std: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "triplet_accuracy": self.triplet_accuracy,
            "mean_positive_sim": self.mean_positive_sim,
            "mean_negative_sim": self.mean_negative_sim,
            "mean_margin": self.mean_margin,
            "margin_std": self.margin_std,
            "positive_sim_std": self.positive_sim_std,
            "negative_sim_std": self.negative_sim_std,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TripletMetrics":
        """Create instance from dictionary."""
        return cls(
            triplet_accuracy=data["triplet_accuracy"],
            mean_positive_sim=data["mean_positive_sim"],
            mean_negative_sim=data["mean_negative_sim"],
            mean_margin=data["mean_margin"],
            margin_std=data["margin_std"],
            positive_sim_std=data["positive_sim_std"],
            negative_sim_std=data["negative_sim_std"],
        )


@dataclass
class ClassificationMetrics:
    """Classification metrics for threshold-based evaluation."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    @property
    def specificity(self) -> float:
        """True negative rate."""
        total_negatives = self.true_negatives + self.false_positives
        if total_negatives == 0:
            return 0.0
        return self.true_negatives / total_negatives

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "specificity": self.specificity,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ClassificationMetrics":
        """Create instance from dictionary."""
        return cls(
            threshold=data["threshold"],
            accuracy=data["accuracy"],
            precision=data["precision"],
            recall=data["recall"],
            f1_score=data["f1_score"],
            true_positives=data["true_positives"],
            true_negatives=data["true_negatives"],
            false_positives=data["false_positives"],
            false_negatives=data["false_negatives"],
        )


class MetricsCalculator:
    """Calculator for evaluation metrics with triplet data."""

    @staticmethod
    def compute_triplet_metrics(
        positive_similarities: np.ndarray,
        negative_similarities: np.ndarray,
    ) -> TripletMetrics:
        """
        Compute triplet-based metrics.

        Args:
            positive_similarities: Cosine similarities between anchor and positive
            negative_similarities: Cosine similarities between anchor and negative
        """
        # Triplet accuracy: how often positive similarity > negative similarity
        correct = positive_similarities > negative_similarities
        triplet_accuracy = float(np.mean(correct))

        # Margin analysis
        margins = positive_similarities - negative_similarities
        mean_margin = float(np.mean(margins))
        margin_std = float(np.std(margins))

        return TripletMetrics(
            triplet_accuracy=triplet_accuracy,
            mean_positive_sim=float(np.mean(positive_similarities)),
            mean_negative_sim=float(np.mean(negative_similarities)),
            mean_margin=mean_margin,
            margin_std=margin_std,
            positive_sim_std=float(np.std(positive_similarities)),
            negative_sim_std=float(np.std(negative_similarities)),
        )

    @staticmethod
    def compute_metrics_at_threshold(
        similarities: np.ndarray,
        labels: np.ndarray,
        threshold: float,
    ) -> ClassificationMetrics:
        """
        Compute classification metrics at a given threshold.

        Args:
            similarities: Cosine similarity scores
            labels: Ground truth labels (0 or 1)
            threshold: Similarity threshold for classification
        """
        predictions = (similarities >= threshold).astype(int)

        # Handle edge cases
        if len(np.unique(predictions)) == 1:
            if predictions[0] == 1:
                precision = float(np.mean(labels))
                recall = 1.0
            else:
                precision = 1.0 if np.sum(labels) == 0 else 0.0
                recall = 0.0
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        else:
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)

        accuracy = accuracy_score(labels, predictions)
        cm = confusion_matrix(labels, predictions, labels=[0, 1])

        tn, fp = cm[0]
        fn, tp = cm[1]

        return ClassificationMetrics(
            threshold=threshold,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
        )

    @staticmethod
    def compute_roc_auc(
        similarities: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute ROC AUC score."""
        try:
            return roc_auc_score(labels, similarities)
        except ValueError:
            return 0.0

    @staticmethod
    def compute_precision_recall_curve(
        similarities: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute precision-recall curve."""
        precision, recall, thresholds = precision_recall_curve(labels, similarities)
        return precision, recall, thresholds

    @staticmethod
    def find_threshold_at_precision(
        similarities: np.ndarray,
        labels: np.ndarray,
        target_precision: float,
    ) -> Optional[float]:
        """Find threshold that achieves at least target precision."""
        precision, recall, thresholds = precision_recall_curve(labels, similarities)

        valid_indices = np.where(precision[:-1] >= target_precision)[0]

        if len(valid_indices) == 0:
            return None

        return float(thresholds[valid_indices[0]])

    @staticmethod
    def compute_similarity_stats(
        positive_similarities: np.ndarray,
        negative_similarities: np.ndarray,
    ) -> Dict:
        """Compute statistics about similarity distributions."""
        all_sims = np.concatenate([positive_similarities, negative_similarities])

        stats = {
            "overall": {
                "mean": float(np.mean(all_sims)),
                "std": float(np.std(all_sims)),
                "min": float(np.min(all_sims)),
                "max": float(np.max(all_sims)),
            },
            "positive_pairs": {
                "count": len(positive_similarities),
                "mean": float(np.mean(positive_similarities)),
                "std": float(np.std(positive_similarities)),
                "min": float(np.min(positive_similarities)),
                "max": float(np.max(positive_similarities)),
            },
            "negative_pairs": {
                "count": len(negative_similarities),
                "mean": float(np.mean(negative_similarities)),
                "std": float(np.std(negative_similarities)),
                "min": float(np.min(negative_similarities)),
                "max": float(np.max(negative_similarities)),
            },
            "separability": float(
                np.mean(positive_similarities) - np.mean(negative_similarities)
            ),
        }

        return stats

    @staticmethod
    def format_confusion_matrix(metrics: ClassificationMetrics) -> str:
        """Format confusion matrix for display."""
        tp, tn = metrics.true_positives, metrics.true_negatives
        fp, fn = metrics.false_positives, metrics.false_negatives
        total = tp + tn + fp + fn

        return f"""
Confusion Matrix (threshold={metrics.threshold:.3f}):
                 Predicted
                 Neg    Pos
Actual  Neg      {tn:6d} {fp:6d}
        Pos      {fn:6d} {tp:6d}

TN: {tn:6d} ({tn/total*100:5.2f}%)  FP: {fp:6d} ({fp/total*100:5.2f}%)
FN: {fn:6d} ({fn/total*100:5.2f}%)  TP: {tp:6d} ({tp/total*100:5.2f}%)
"""

    @staticmethod
    def sweep_thresholds(
        similarities: np.ndarray,
        labels: np.ndarray,
        threshold_range: Tuple[float, float] = (0.50, 0.99),
        step: float = 0.01,
    ) -> Dict[float, ClassificationMetrics]:
        """Compute metrics across a range of thresholds."""
        results = {}
        for threshold in np.arange(threshold_range[0], threshold_range[1] + step, step):
            metrics = MetricsCalculator.compute_metrics_at_threshold(
                similarities, labels, threshold
            )
            results[round(threshold, 3)] = metrics
        return results

    @staticmethod
    def triplets_to_classification_data(
        positive_similarities: np.ndarray,
        negative_similarities: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert triplet similarities to classification format.

        Args:
            positive_similarities: sim(anchor, positive) scores
            negative_similarities: sim(anchor, negative) scores

        Returns:
            Tuple of (all_similarities, labels)
        """
        all_sims = np.concatenate([positive_similarities, negative_similarities])
        labels = np.concatenate([
            np.ones(len(positive_similarities)),
            np.zeros(len(negative_similarities)),
        ])
        return all_sims, labels
