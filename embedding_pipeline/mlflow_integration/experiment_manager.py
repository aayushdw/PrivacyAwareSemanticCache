"""MLFlow experiment management."""

import json
import os
import subprocess
import tempfile
from typing import List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from ..config.pipeline_config import get_config, get_experiment_name


class ExperimentManager:
    """
    Manages MLFlow experiments for the evaluation pipeline.

    Handles experiment creation, hierarchy, and configuration.
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize experiment manager.

        Args:
            tracking_uri: MLFlow tracking server URI (defaults to config)
        """
        config = get_config()
        self.tracking_uri = tracking_uri or config.mlflow.tracking_uri
        self.experiment_prefix = config.mlflow.experiment_prefix

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()

    def get_or_create_experiment(self, experiment_name: str) -> str:
        """
        Get or create an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Experiment ID
        """
        experiment = self.client.get_experiment_by_name(experiment_name)

        if experiment is None:
            experiment_id = self.client.create_experiment(experiment_name)
            print(f"Created experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

        return experiment_id

    def setup_threshold_tuning_experiment(self) -> str:
        """Set up experiment for threshold tuning stage."""
        experiment_name = get_experiment_name("threshold_tuning")
        return self.get_or_create_experiment(experiment_name)

    def setup_ranking_experiment(self) -> str:
        """Set up experiment for ranking stage."""
        experiment_name = get_experiment_name("ranking")
        return self.get_or_create_experiment(experiment_name)

    def setup_lora_training_experiment(self) -> str:
        """Set up experiment for LoRA training stage."""
        experiment_name = get_experiment_name("lora_training")
        return self.get_or_create_experiment(experiment_name)

    def set_experiment(self, stage: str) -> str:
        """
        Set the active experiment for a pipeline stage.

        Args:
            stage: Pipeline stage ('threshold_tuning', 'ranking', 'lora_training')

        Returns:
            Experiment ID
        """
        experiment_name = get_experiment_name(stage)
        experiment_id = self.get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        return experiment_id

    def get_completed_model_evaluations(self, experiment_name: str) -> set:
        """
        Get set of model keys that have already been evaluated.

        Useful for checkpoint/resume functionality.

        Args:
            experiment_name: Experiment to search

        Returns:
            Set of model keys with completed evaluations
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return set()

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
        )

        completed = set()
        for run in runs:
            model_key = run.data.params.get("model_key")
            if model_key:
                completed.add(model_key)

        return completed

    def load_cached_evaluation_results(
        self,
        experiment_name: str,
    ) -> List["EvaluationResult"]:
        """
        Load completed evaluation results from MLFlow artifacts.

        Args:
            experiment_name: Experiment to load results from

        Returns:
            List of EvaluationResult objects from completed runs
        """
        from ..evaluation.embedding_evaluator import EvaluationResult

        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
        )

        results = []
        for run in runs:
            try:
                # Download the result artifact
                artifact_uri = f"runs:/{run.info.run_id}/results"
                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_path = mlflow.artifacts.download_artifacts(
                        artifact_uri=artifact_uri,
                        dst_path=tmp_dir,
                    )
                    # The artifact is saved as {model_key}.json in the results folder
                    for filename in os.listdir(local_path):
                        if filename.endswith(".json"):
                            json_path = os.path.join(local_path, filename)
                            with open(json_path, "r") as f:
                                data = json.load(f)
                            result = EvaluationResult.from_dict(data)
                            if result.success:
                                results.append(result)
                            break
            except Exception as e:
                # Skip runs where we can't load the artifact
                model_key = run.data.params.get("model_key", "unknown")
                print(f"Warning: Could not load cached result for {model_key}: {e}")
                continue

        return results

    def get_best_models_from_experiment(
        self,
        experiment_name: str,
        metric: str = "f1_score",
        top_n: int = 3,
        min_precision: Optional[float] = None,
    ) -> list:
        """
        Get best performing models from an experiment.

        Args:
            experiment_name: Experiment to search
            metric: Metric to rank by
            top_n: Number of models to return
            min_precision: Minimum precision filter

        Returns:
            List of (model_key, run_id, metrics) tuples
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []

        filter_string = "attributes.status = 'FINISHED'"
        if min_precision is not None:
            filter_string += f" and metrics.precision >= {min_precision}"

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=[f"metrics.{metric} DESC"],
            max_results=top_n,
        )

        results = []
        for run in runs:
            model_key = run.data.params.get("model_key")
            if model_key:
                results.append(
                    {
                        "model_key": model_key,
                        "run_id": run.info.run_id,
                        "metrics": dict(run.data.metrics),
                        "params": dict(run.data.params),
                    }
                )

        return results

    @staticmethod
    def get_git_info() -> dict:
        """Get current Git commit info for reproducibility."""
        info = {}

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )
            if result.returncode == 0:
                info["git_commit"] = result.stdout.strip()

            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )
            if result.returncode == 0:
                info["git_branch"] = result.stdout.strip()

            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )
            if result.returncode == 0:
                info["git_dirty"] = len(result.stdout.strip()) > 0

        except Exception:
            pass

        return info

    def list_experiments(self) -> list:
        """List all experiments with the configured prefix."""
        experiments = self.client.search_experiments(
            filter_string=f"name LIKE '{self.experiment_prefix}%'"
        )
        return [
            {"name": exp.name, "id": exp.experiment_id, "lifecycle_stage": exp.lifecycle_stage}
            for exp in experiments
        ]

    def delete_experiment(self, experiment_name: str):
        """Delete an experiment (moves to deleted state)."""
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment:
            self.client.delete_experiment(experiment.experiment_id)
            print(f"Deleted experiment: {experiment_name}")
        else:
            print(f"Experiment not found: {experiment_name}")
