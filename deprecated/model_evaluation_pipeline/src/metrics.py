from typing import List
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    accuracy_score
)
import numpy as np
import warnings

def calculate_cache_metrics(labels: List[int], scores: np.ndarray, min_precision: float = 0.95):
    """
    Finds the optimal threshold where Precision >= min_precision.
    Returns a dict of metrics including classification metrics, confusion matrix components,
    and arrays needed for visualization.
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)

    # Calculate Average Precision (AP) - a good overall summary score
    ap_score = average_precision_score(labels, scores)

    # Calculate ROC-AUC score
    roc_auc = roc_auc_score(labels, scores)

    # Find valid thresholds that meet our precision safety guardrail
    valid_indices = np.where(precisions[:-1] >= min_precision)[0]

    if len(valid_indices) > 0:
        # We want the index with the highest Recall among valid precisions
        # Since precision/recall trade off, the first valid index (lowest threshold)
        # usually gives the highest recall.
        best_idx = valid_indices[0]

        optimal_threshold = float(thresholds[best_idx])
        recall_at_threshold = float(recalls[best_idx])
        precision_at_threshold = float(precisions[best_idx])
    else:
        # Fallback if model is terrible and never reaches target precision
        max_precision = float(precisions[:-1].max()) if len(precisions) > 1 else 0.0
        warning_msg = (
            f"WARNING: Model failed to achieve target precision of {min_precision:.2%}. "
            f"Maximum precision achieved: {max_precision:.2%}. "
            f"Setting threshold to 0.99 with recall=0.0"
        )
        warnings.warn(warning_msg, UserWarning)
        print(f"⚠️  {warning_msg}")

        optimal_threshold = 0.99
        recall_at_threshold = 0.0
        precision_at_threshold = max_precision

    # Make binary predictions at optimal threshold
    predictions = (scores >= optimal_threshold).astype(int)

    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    # Calculate additional classification metrics at optimal threshold
    f1 = f1_score(labels, predictions)
    acc = accuracy_score(labels, predictions)

    return {
        # Existing metrics
        "ap_score": ap_score,
        "optimal_threshold": optimal_threshold,
        "recall_at_target": recall_at_threshold,
        "precision_at_target": precision_at_threshold,

        # New classification metrics
        "f1_score": f1,
        "accuracy": acc,
        "roc_auc": roc_auc,

        # Confusion matrix components
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),

        # Arrays for visualization (not logged to MLflow)
        "all_precisions": precisions,
        "all_recalls": recalls,
        "pr_thresholds": thresholds,
        "labels": labels,
        "scores": scores,
    }