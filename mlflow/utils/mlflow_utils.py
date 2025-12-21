"""MLFlow utility functions for experiment tracking."""

import os
from typing import Any

import mlflow
from dotenv import load_dotenv

load_dotenv()


def get_or_create_experiment(experiment_name: str) -> str:
    """Get existing experiment or create a new one, returning the experiment ID."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    return experiment_id


def log_model_params(params: dict[str, Any]) -> None:
    """Log model hyperparameters to the active MLFlow run."""
    mlflow.log_params(params)


def log_embedding_metrics(
    metrics: dict[str, float],
    step: int | None = None,
) -> None:
    """Log embedding evaluation metrics to the active MLFlow run."""
    mlflow.log_metrics(metrics, step=step)
