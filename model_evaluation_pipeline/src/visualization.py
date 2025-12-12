import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve
from typing import Dict
import tempfile
import os

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_precision_recall_curve(
    precisions: np.ndarray,
    recalls: np.ndarray,
    ap_score: float,
    optimal_threshold: float,
    thresholds: np.ndarray,
    model_name: str
) -> str:
    """
    Generate Precision-Recall curve plot.

    Args:
        precisions: Array of precision values at different thresholds
        recalls: Array of recall values at different thresholds
        ap_score: Average Precision score
        optimal_threshold: The selected optimal threshold
        thresholds: Array of threshold values
        model_name: Name of model (for title)

    Returns:
        Path to saved PNG file
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot PR curve
    ax.plot(recalls, precisions, linewidth=2, label=f'PR Curve (AP={ap_score:.3f})')

    # Mark optimal threshold point
    # Find index of optimal threshold in thresholds array
    opt_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    ax.scatter([recalls[opt_idx]], [precisions[opt_idx]],
               color='red', s=100, zorder=5,
               label=f'Optimal Threshold={optimal_threshold:.3f}')

    # Styling
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # Save to temp file
    temp_path = os.path.join(tempfile.gettempdir(), 'pr_curve.png')
    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return temp_path


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    roc_auc: float,
    model_name: str
) -> str:
    """
    Generate ROC curve plot.

    Args:
        labels: True binary labels
        scores: Predicted probability scores
        roc_auc: ROC AUC score
        model_name: Name of model (for title)

    Returns:
        Path to saved PNG file
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)

    # Plot ROC curve
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC={roc_auc:.3f})')

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    # Styling
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # Save to temp file
    temp_path = os.path.join(tempfile.gettempdir(), 'roc_curve.png')
    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return temp_path


def plot_confusion_matrix(
    tp: int, tn: int, fp: int, fn: int,
    precision: float, recall: float, f1: float,
    model_name: str
) -> str:
    """
    Generate confusion matrix heatmap.

    Args:
        tp, tn, fp, fn: Confusion matrix components
        precision, recall, f1: Metrics at optimal threshold
        model_name: Name of model (for title)

    Returns:
        Path to saved PNG file
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create confusion matrix array
    cm = np.array([[tn, fp], [fn, tp]])

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar=True, ax=ax,
                xticklabels=['Not Duplicate', 'Duplicate'],
                yticklabels=['Not Duplicate', 'Duplicate'],
                cbar_kws={'label': 'Count'})

    # Add metrics as subtitle
    metrics_text = f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}"
    ax.set_title(f"Confusion Matrix - {model_name}\n{metrics_text}",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    plt.tight_layout()

    # Save to temp file
    temp_path = os.path.join(tempfile.gettempdir(), 'confusion_matrix.png')
    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return temp_path


def generate_all_plots(metrics: Dict, model_name: str) -> Dict[str, str]:
    """
    Generate all visualization plots and return paths.

    Args:
        metrics: Dictionary containing all metrics and arrays from calculate_cache_metrics()
        model_name: Name of the model being evaluated

    Returns:
        Dictionary mapping plot names to file paths
    """
    plot_paths = {}

    # Precision-Recall Curve
    plot_paths['pr_curve'] = plot_precision_recall_curve(
        precisions=metrics['all_precisions'],
        recalls=metrics['all_recalls'],
        ap_score=metrics['ap_score'],
        optimal_threshold=metrics['optimal_threshold'],
        thresholds=metrics['pr_thresholds'],
        model_name=model_name
    )

    # ROC Curve
    plot_paths['roc_curve'] = plot_roc_curve(
        labels=metrics['labels'],
        scores=metrics['scores'],
        roc_auc=metrics['roc_auc'],
        model_name=model_name
    )

    # Confusion Matrix
    plot_paths['confusion_matrix'] = plot_confusion_matrix(
        tp=metrics['true_positives'],
        tn=metrics['true_negatives'],
        fp=metrics['false_positives'],
        fn=metrics['false_negatives'],
        precision=metrics['precision_at_target'],
        recall=metrics['recall_at_target'],
        f1=metrics['f1_score'],
        model_name=model_name
    )

    return plot_paths
