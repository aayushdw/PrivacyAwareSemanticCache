"""
Generate comprehensive evaluation report for all embedding models.
Runs threshold tuning on all models and creates visualizations + markdown report.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from eval.threshold_tuner import tune_all_models
from eval.model_registry import get_model_info

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class EvaluationReportGenerator:
    """Generates evaluation report with plots and markdown documentation."""

    def __init__(self, output_dir: str = None):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save results (default: eval/results)
        """
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'results'
            )

        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)

        self.tuning_results = None
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def run_tuning(self, max_samples: int = 1000, metric: str = 'f1', min_precision: float = 0.90):
        """
        Run threshold tuning on all models.

        Args:
            max_samples: Maximum samples to use for tuning
            metric: Metric to optimize
            min_precision: Minimum precision constraint
        """
        print("="*100)
        print("RUNNING THRESHOLD TUNING FOR ALL MODELS")
        print("="*100)
        print(f"Configuration:")
        print(f"  Max samples: {max_samples}")
        print(f"  Optimization metric: {metric}")
        print(f"  Min precision constraint: {min_precision}")
        print("="*100)

        tuner = tune_all_models(
            max_samples=max_samples,
            metric=metric,
            min_precision=min_precision
        )

        self.tuning_results = tuner.tuning_results
        return tuner

    def _prepare_dataframe(self):
        """Prepare DataFrame from tuning results."""
        successful_results = [r for r in self.tuning_results if r.get('success', False)]

        data = []
        for result in successful_results:
            metrics = result['metrics_at_optimal']
            cm = metrics.get('confusion_matrix', {})

            data.append({
                'Model': result['model_name'],
                'Key': result['model_key'],
                'Category': result['category'],
                'Dimension': result['dimension'],
                'Threshold': result['optimal_threshold'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1_score'],
                'TN': cm.get('true_negatives', 0),
                'FP': cm.get('false_positives', 0),
                'FN': cm.get('false_negatives', 0),
                'TP': cm.get('true_positives', 0),
            })

        return pd.DataFrame(data)

    def plot_thresholds_by_category(self):
        """Plot optimal thresholds grouped by model category."""
        df = self._prepare_dataframe()

        fig, ax = plt.subplots(figsize=(14, 8))

        # Sort by category and threshold
        df_sorted = df.sort_values(['Category', 'Threshold'])

        # Create color map for categories
        categories = df_sorted['Category'].unique()
        colors = sns.color_palette("husl", len(categories))
        category_colors = {cat: colors[i] for i, cat in enumerate(categories)}

        # Plot bars
        bars = ax.barh(
            range(len(df_sorted)),
            df_sorted['Threshold'],
            color=[category_colors[cat] for cat in df_sorted['Category']]
        )

        # Customize
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['Key'], fontsize=9)
        ax.set_xlabel('Optimal Threshold', fontsize=12)
        ax.set_title('Optimal Thresholds by Model (Grouped by Category)', fontsize=14, fontweight='bold')
        ax.set_xlim(0.5, 1.0)
        ax.grid(axis='x', alpha=0.3)

        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, fc=category_colors[cat], label=cat.capitalize())
                          for cat in categories]
        ax.legend(handles=legend_elements, loc='lower right', title='Category')

        # Add value labels
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            ax.text(row['Threshold'] + 0.005, i, f"{row['Threshold']:.3f}",
                   va='center', fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'thresholds_by_category.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return 'plots/thresholds_by_category.png'

    def plot_metrics_comparison(self):
        """Plot comparison of precision, recall, and F1 for all models."""
        df = self._prepare_dataframe()

        # Sort by F1 score
        df_sorted = df.sort_values('F1', ascending=True)

        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot metrics
        x = range(len(df_sorted))
        width = 0.25

        bars1 = ax.barh([i - width for i in x], df_sorted['Precision'],
                       width, label='Precision', color='#2ecc71', alpha=0.8)
        bars2 = ax.barh(x, df_sorted['Recall'],
                       width, label='Recall', color='#3498db', alpha=0.8)
        bars3 = ax.barh([i + width for i in x], df_sorted['F1'],
                       width, label='F1 Score', color='#e74c3c', alpha=0.8)

        # Customize
        ax.set_yticks(x)
        ax.set_yticklabels(df_sorted['Key'], fontsize=9)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison (Precision, Recall, F1)',
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'metrics_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return 'plots/metrics_comparison.png'

    def plot_precision_vs_recall(self):
        """Plot precision vs recall scatter plot."""
        df = self._prepare_dataframe()

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create color map for categories
        categories = df['Category'].unique()
        colors = sns.color_palette("husl", len(categories))
        category_colors = {cat: colors[i] for i, cat in enumerate(categories)}

        # Plot scatter
        for category in categories:
            cat_df = df[df['Category'] == category]
            ax.scatter(cat_df['Recall'], cat_df['Precision'],
                      s=100, alpha=0.6,
                      color=category_colors[category],
                      label=category.capitalize(),
                      edgecolors='black', linewidth=0.5)

        # Add model labels for top performers
        top_models = df.nlargest(5, 'F1')
        for _, row in top_models.iterrows():
            ax.annotate(row['Key'],
                       (row['Recall'], row['Precision']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)

        # Add diagonal reference line (F1 contours)
        x_line = np.linspace(0, 1, 100)
        for f1 in [0.5, 0.6, 0.7, 0.8, 0.9]:
            y_line = (f1 * x_line) / (2 * x_line - f1)
            y_line = np.clip(y_line, 0, 1)
            ax.plot(x_line, y_line, '--', alpha=0.3, color='gray', linewidth=0.5)
            ax.text(0.9, (f1 * 0.9) / (2 * 0.9 - f1), f'F1={f1}',
                   fontsize=7, alpha=0.5, color='gray')

        # Customize
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'precision_vs_recall.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return 'plots/precision_vs_recall.png'

    def plot_confusion_matrices_top_models(self, top_n: int = 6):
        """Plot confusion matrices for top N models."""
        df = self._prepare_dataframe()

        # Get top models by F1 score
        top_models = df.nlargest(top_n, 'F1')

        # Create subplots
        n_cols = 3
        n_rows = (top_n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if top_n > 1 else [axes]

        for idx, (_, model) in enumerate(top_models.iterrows()):
            ax = axes[idx]

            # Create confusion matrix array
            cm = np.array([[model['TN'], model['FP']],
                          [model['FN'], model['TP']]])

            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       cbar=False, ax=ax,
                       xticklabels=['Neg', 'Pos'],
                       yticklabels=['Neg', 'Pos'])

            # Add metrics text
            metrics_text = (f"Prec: {model['Precision']:.3f} | "
                          f"Rec: {model['Recall']:.3f} | "
                          f"F1: {model['F1']:.3f}")

            ax.set_title(f"{model['Key']}\n{metrics_text}",
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('Actual', fontsize=9)

        # Hide unused subplots
        for idx in range(len(top_models), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Confusion Matrices - Top {top_n} Models by F1 Score',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        plot_path = os.path.join(self.plots_dir, 'confusion_matrices_top_models.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return 'plots/confusion_matrices_top_models.png'

    def plot_category_performance(self):
        """Plot average performance by category."""
        df = self._prepare_dataframe()

        # Calculate average metrics by category
        category_stats = df.groupby('Category').agg({
            'Precision': 'mean',
            'Recall': 'mean',
            'F1': 'mean',
            'Threshold': 'mean'
        }).reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Average metrics by category
        x = range(len(category_stats))
        width = 0.25

        ax1.bar([i - width for i in x], category_stats['Precision'],
               width, label='Precision', color='#2ecc71', alpha=0.8)
        ax1.bar(x, category_stats['Recall'],
               width, label='Recall', color='#3498db', alpha=0.8)
        ax1.bar([i + width for i in x], category_stats['F1'],
               width, label='F1 Score', color='#e74c3c', alpha=0.8)

        ax1.set_xticks(x)
        ax1.set_xticklabels([cat.capitalize() for cat in category_stats['Category']])
        ax1.set_ylabel('Average Score', fontsize=11)
        ax1.set_title('Average Performance by Category', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1)

        # Plot 2: Average threshold by category
        bars = ax2.bar(x, category_stats['Threshold'],
                      color=sns.color_palette("husl", len(category_stats)), alpha=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels([cat.capitalize() for cat in category_stats['Category']])
        ax2.set_ylabel('Average Threshold', fontsize=11)
        ax2.set_title('Average Optimal Threshold by Category', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0.5, 1.0)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, category_stats['Threshold'])):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'category_performance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return 'plots/category_performance.png'

    def generate_markdown_report(self):
        """Generate comprehensive markdown report."""
        df = self._prepare_dataframe()

        # Get statistics
        best_f1 = df.loc[df['F1'].idxmax()]
        best_precision = df.loc[df['Precision'].idxmax()]
        best_recall = df.loc[df['Recall'].idxmax()]

        # Load the full results JSON for additional details
        results_json_path = os.path.join(self.output_dir, 'threshold_tuning_all_models.json')
        with open(results_json_path, 'r') as f:
            full_results = json.load(f)

        # Extract configuration from first successful result
        config = next((r for r in full_results if r.get('success', False)), {})

        md_content = f"""# Embedding Models Evaluation Report

**Generated:** {self.timestamp}

## Configuration

- **Optimization Metric:** {config.get('optimization_metric', 'f1')}
- **Minimum Precision Constraint:** {config.get('min_precision_constraint', 'N/A')}
- **Training Samples:** {config.get('tuning_samples', 'N/A')}
- **Total Models Evaluated:** {len(df)}

## Executive Summary

### Best Performing Models

- **Best F1 Score:** `{best_f1['Key']}` - F1: {best_f1['F1']:.4f} (Precision: {best_f1['Precision']:.4f}, Recall: {best_f1['Recall']:.4f})
- **Best Precision:** `{best_precision['Key']}` - Precision: {best_precision['Precision']:.4f} (F1: {best_precision['F1']:.4f})
- **Best Recall:** `{best_recall['Key']}` - Recall: {best_recall['Recall']:.4f} (F1: {best_recall['F1']:.4f})

### Overall Statistics

- **Average F1 Score:** {df['F1'].mean():.4f} ± {df['F1'].std():.4f}
- **Average Precision:** {df['Precision'].mean():.4f} ± {df['Precision'].std():.4f}
- **Average Recall:** {df['Recall'].mean():.4f} ± {df['Recall'].std():.4f}
- **Average Optimal Threshold:** {df['Threshold'].mean():.4f} ± {df['Threshold'].std():.4f}

## Visualizations

### 1. Optimal Thresholds by Model Category

![Thresholds by Category](thresholds_by_category.png)

This chart shows the optimal similarity thresholds for each model, grouped by category (fast, balanced, quality, multilingual). Higher thresholds indicate the model requires stronger similarity before marking questions as duplicates.

### 2. Model Performance Comparison

![Metrics Comparison](metrics_comparison.png)

Comparison of precision, recall, and F1 scores across all models. Models are sorted by F1 score (ascending).

### 3. Precision vs Recall Trade-off

![Precision vs Recall](precision_vs_recall.png)

This scatter plot visualizes the precision-recall trade-off for each model. The diagonal lines represent constant F1 scores. Models in the upper-right corner achieve the best balance.

### 4. Confusion Matrices - Top Performers

![Confusion Matrices](confusion_matrices_top_models.png)

Confusion matrices for the top 6 models by F1 score, showing:
- **TN (True Negatives):** Correctly identified non-duplicates
- **FP (False Positives):** Non-duplicates incorrectly marked as duplicates (cache false hits)
- **FN (False Negatives):** Duplicates incorrectly marked as non-duplicates (cache misses)
- **TP (True Positives):** Correctly identified duplicates

### 5. Performance by Category

![Category Performance](category_performance.png)

Average performance metrics and optimal thresholds grouped by model category.

## Detailed Results

### Top 10 Models by F1 Score

"""

        # Add top 10 table
        top_10 = df.nlargest(10, 'F1')[['Key', 'Category', 'Dimension', 'Threshold',
                                         'Precision', 'Recall', 'F1']]

        md_content += "| Rank | Model | Category | Dim | Threshold | Precision | Recall | F1 Score |\n"
        md_content += "|------|-------|----------|-----|-----------|-----------|--------|----------|\n"

        for idx, (_, row) in enumerate(top_10.iterrows(), 1):
            md_content += (f"| {idx} | `{row['Key']}` | {row['Category'].capitalize()} | "
                          f"{row['Dimension']} | {row['Threshold']:.3f} | "
                          f"{row['Precision']:.4f} | {row['Recall']:.4f} | "
                          f"**{row['F1']:.4f}** |\n")

        # Add category breakdown
        md_content += "\n### Performance by Category\n\n"

        for category in df['Category'].unique():
            cat_df = df[df['Category'] == category].sort_values('F1', ascending=False)

            md_content += f"\n#### {category.capitalize()} Models\n\n"
            md_content += "| Model | Dimension | Threshold | Precision | Recall | F1 Score |\n"
            md_content += "|-------|-----------|-----------|-----------|--------|----------|\n"

            for _, row in cat_df.iterrows():
                md_content += (f"| `{row['Key']}` | {row['Dimension']} | "
                              f"{row['Threshold']:.3f} | {row['Precision']:.4f} | "
                              f"{row['Recall']:.4f} | {row['F1']:.4f} |\n")

        # Add recommendations
        md_content += """
## Recommendations

### For Production Use

Based on the evaluation results:

1. **Best Overall Balance:** `{best_overall}` offers the best F1 score ({f1:.4f}) with strong precision ({prec:.4f}) and recall ({rec:.4f}).

2. **For Speed-Critical Applications:** Consider models from the "fast" category if latency is a primary concern.

3. **For High-Precision Requirements:** If minimizing false positives (incorrect cache hits) is critical, consider `{best_prec}` with precision of {prec_val:.4f}.

4. **For High-Recall Requirements:** If minimizing false negatives (cache misses) is critical, consider `{best_rec}` with recall of {rec_val:.4f}.

### Threshold Tuning

- The optimal thresholds range from {min_thresh:.3f} to {max_thresh:.3f}
- Models with higher thresholds are more conservative (fewer false positives, more false negatives)
- Models with lower thresholds are more aggressive (more false positives, fewer false negatives)

### Next Steps

1. **Validation Testing:** Validate top models on a held-out test set
2. **Latency Benchmarking:** Measure inference time for production workload
3. **A/B Testing:** Deploy top 2-3 candidates for real-world testing
4. **Fine-tuning:** Consider fine-tuning top models on domain-specific data

## Raw Data

Complete results are available in `threshold_tuning_all_models.json`.

---

*Report generated by Privacy-Aware Semantic Cache Evaluation Framework*
""".format(
            best_overall=best_f1['Key'],
            f1=best_f1['F1'],
            prec=best_f1['Precision'],
            rec=best_f1['Recall'],
            best_prec=best_precision['Key'],
            prec_val=best_precision['Precision'],
            best_rec=best_recall['Key'],
            rec_val=best_recall['Recall'],
            min_thresh=df['Threshold'].min(),
            max_thresh=df['Threshold'].max()
        )

        # Write report
        report_path = os.path.join(self.output_dir, 'results.md')
        with open(report_path, 'w') as f:
            f.write(md_content)

        print(f"\n✓ Markdown report saved to: {report_path}")
        return report_path

    def generate_full_report(self, max_samples: int = 1000, metric: str = 'f1',
                            min_precision: float = 0.90):
        """
        Run complete evaluation and generate report with all visualizations.

        Args:
            max_samples: Maximum samples for tuning
            metric: Metric to optimize
            min_precision: Minimum precision constraint
        """
        # Run tuning
        self.run_tuning(max_samples, metric, min_precision)

        print("\n" + "="*100)
        print("GENERATING VISUALIZATIONS")
        print("="*100)

        # Generate all plots
        print("\n[1/5] Generating thresholds by category plot...")
        self.plot_thresholds_by_category()

        print("[2/5] Generating metrics comparison plot...")
        self.plot_metrics_comparison()

        print("[3/5] Generating precision vs recall plot...")
        self.plot_precision_vs_recall()

        print("[4/5] Generating confusion matrices for top models...")
        self.plot_confusion_matrices_top_models()

        print("[5/5] Generating category performance plot...")
        self.plot_category_performance()

        print("\n" + "="*100)
        print("GENERATING MARKDOWN REPORT")
        print("="*100)

        # Generate markdown report
        report_path = self.generate_markdown_report()

        print("\n" + "="*100)
        print("REPORT GENERATION COMPLETE")
        print("="*100)
        print(f"\nReport location: {report_path}")
        print(f"Plots location: {self.plots_dir}")
        print(f"\nYou can view the report by opening: {report_path}")

        return report_path


def main():
    """Main execution function."""
    generator = EvaluationReportGenerator()

    generator.generate_full_report(
        max_samples=2000,
        metric='f1',
        min_precision=0.80
    )


if __name__ == "__main__":
    main()
