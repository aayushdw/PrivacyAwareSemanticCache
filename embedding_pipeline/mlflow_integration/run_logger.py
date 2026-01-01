"""MLFlow run logging utilities."""

import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow

from ..evaluation.embedding_evaluator import EvaluationResult
from ..registry.model_info import ModelInfo
from ..utils.data_loader import get_dvc_dataset_version
from .experiment_manager import ExperimentManager


class RunLogger:
    """
    Handles MLFlow run logging for the evaluation pipeline.

    Provides structured logging of parameters, metrics, tags, and artifacts.
    """

    def __init__(self):
        """Initialize run logger."""
        self.experiment_manager = ExperimentManager()

    def start_evaluation_run(
        self,
        model_info: ModelInfo,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Start a new MLFlow run for model evaluation.

        Args:
            model_info: Model being evaluated
            run_name: Optional run name
            tags: Additional tags

        Returns:
            Run ID
        """
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{model_info.name.lower().replace(' ', '-')}_{timestamp}"

        run = mlflow.start_run(run_name=run_name)

        # Log model parameters
        mlflow.log_params(
            {
                "model_key": model_info.name.lower().replace(" ", "-"),
                "model_name": model_info.name,
                "model_id": model_info.model_id,
                "dimension": model_info.dimension,
                "category": model_info.category,
                "max_seq_length": model_info.max_seq_length,
                "requires_instruction": model_info.requires_instruction,
            }
        )

        # Log standard tags
        self._log_standard_tags(model_info)

        # Log custom tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        return run.info.run_id

    def _log_standard_tags(self, model_info: ModelInfo):
        """Log standard tags for filtering and organization."""
        # Model classification tags
        mlflow.set_tag("model_family", "sentence-transformers")
        mlflow.set_tag("model_size", model_info.get_size_category())
        mlflow.set_tag("category", model_info.category)

        # Git info for reproducibility
        git_info = self.experiment_manager.get_git_info()
        for key, value in git_info.items():
            mlflow.set_tag(key, str(value))

        # Dataset version
        dataset_version = get_dvc_dataset_version()
        mlflow.set_tag("dataset_version", dataset_version)

        # Timestamp
        mlflow.set_tag("evaluation_timestamp", datetime.now().isoformat())

    def log_threshold_tuning_results(
        self,
        result: EvaluationResult,
        min_precision: float,
    ):
        """
        Log threshold tuning results to the current run.

        Args:
            result: Evaluation result with threshold tuning data
            min_precision: Precision constraint used
        """
        # Log constraint parameters
        mlflow.log_param("min_precision_constraint", min_precision)

        if result.threshold_result:
            tr = result.threshold_result

            # Log threshold tuning metrics
            mlflow.log_metrics(
                {
                    "optimal_threshold": tr.threshold,
                    "tuning_f1_score": tr.f1_score,
                    "tuning_precision": tr.precision,
                    "tuning_recall": tr.recall,
                    "tuning_accuracy": tr.accuracy,
                }
            )

            # Log similarity statistics
            if tr.similarity_stats:
                stats = tr.similarity_stats
                if "duplicates" in stats:
                    mlflow.log_metrics(
                        {
                            "sim_duplicate_mean": stats["duplicates"]["mean"],
                            "sim_duplicate_std": stats["duplicates"]["std"],
                        }
                    )
                if "non_duplicates" in stats:
                    mlflow.log_metrics(
                        {
                            "sim_nonduplicate_mean": stats["non_duplicates"]["mean"],
                            "sim_nonduplicate_std": stats["non_duplicates"]["std"],
                        }
                    )
                if "separability" in stats:
                    mlflow.log_metric("sim_separability", stats["separability"])

        # Log constraint satisfaction tag
        mlflow.set_tag(
            "precision_constraint_met",
            str(result.meets_precision_constraint).lower(),
        )

    def log_test_metrics(self, result: EvaluationResult):
        """
        Log test set evaluation metrics.

        Args:
            result: Evaluation result with test metrics
        """
        if result.test_metrics:
            tm = result.test_metrics

            mlflow.log_metrics(
                {
                    "f1_score": tm.f1_score,
                    "precision": tm.precision,
                    "recall": tm.recall,
                    "accuracy": tm.accuracy,
                    "specificity": tm.specificity,
                    "true_positives": tm.true_positives,
                    "true_negatives": tm.true_negatives,
                    "false_positives": tm.false_positives,
                    "false_negatives": tm.false_negatives,
                }
            )

        # Log timing metrics
        mlflow.log_metrics(
            {
                "model_load_time_seconds": result.model_load_time_seconds,
                "embedding_time_seconds": result.embedding_time_seconds,
                "total_samples": result.total_samples,
            }
        )

        # Calculate and log latency per sample
        if result.total_samples > 0:
            latency_ms = (result.embedding_time_seconds / result.total_samples) * 1000
            mlflow.log_metric("latency_ms_per_pair", latency_ms)

    def log_evaluation_result(
        self,
        result: EvaluationResult,
        min_precision: float,
    ):
        """
        Log complete evaluation result.

        Args:
            result: Complete evaluation result
            min_precision: Precision constraint used
        """
        self.log_threshold_tuning_results(result, min_precision)
        self.log_test_metrics(result)

        # Log success/failure
        mlflow.set_tag("evaluation_success", str(result.success).lower())
        if result.error:
            mlflow.set_tag("error_message", result.error[:250])  # Truncate long errors

        # Log full result as artifact
        self.log_result_artifact(result)

    def log_result_artifact(self, result: EvaluationResult):
        """Log full evaluation result as JSON artifact."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(result.to_dict(), f, indent=2)
            temp_path = f.name

        try:
            mlflow.log_artifact(temp_path, "results")
        finally:
            os.unlink(temp_path)

    def log_comparison_report(self, report_content: str, filename: str = "comparison_report.md"):
        """Log markdown comparison report as artifact."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(report_content)
            temp_path = f.name

        try:
            mlflow.log_artifact(temp_path, "reports")
        finally:
            os.unlink(temp_path)

    def log_plot(self, figure, filename: str, artifact_path: str = "plots"):
        """
        Log matplotlib figure as artifact.

        Args:
            figure: Matplotlib figure
            filename: Filename for the plot
            artifact_path: Artifact subdirectory
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            figure.savefig(f.name, dpi=150, bbox_inches="tight")
            temp_path = f.name

        try:
            mlflow.log_artifact(temp_path, artifact_path)
        finally:
            os.unlink(temp_path)

    def end_run(self, status: str = "FINISHED"):
        """End the current MLFlow run."""
        mlflow.end_run(status=status)

    @staticmethod
    def log_ranking_results(
        selected_models: List[Dict],
        min_precision: float,
        top_n: int,
    ):
        """
        Log model ranking results.

        Args:
            selected_models: List of selected model dictionaries
            min_precision: Precision constraint used
            top_n: Number of models selected
        """
        mlflow.log_params(
            {
                "min_precision_constraint": min_precision,
                "top_n": top_n,
                "num_selected": len(selected_models),
            }
        )

        # Log selected model keys as tag
        model_keys = [m.get("model_key", "unknown") for m in selected_models]
        mlflow.set_tag("selected_models", ",".join(model_keys))

        # Log metrics for each selected model
        for i, model in enumerate(selected_models):
            prefix = f"rank_{i+1}_"
            mlflow.log_param(f"{prefix}model_key", model.get("model_key"))
            if "metrics" in model:
                for key, value in model["metrics"].items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{prefix}{key}", value)
