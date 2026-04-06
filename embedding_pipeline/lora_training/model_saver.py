"""LoRA model saving and MLFlow registration utilities."""

import json
import os
import shutil
import tempfile
from datetime import datetime
from typing import Dict, Optional

import mlflow

from ..mlflow_integration.model_registry_client import ModelRegistryClient
from .lora_trainer import TrainingResult


class LoRAModelSaver:
    """Save and register LoRA adapters with MLFlow Model Registry."""

    def __init__(self):
        """Initialize model saver."""
        self.registry_client = ModelRegistryClient()

    def save_training_metadata(
        self,
        adapter_path: str,
        result: TrainingResult,
        base_model_id: str,
        config: dict,
    ):
        """
        Save training metadata alongside the adapter.

        Args:
            adapter_path: Path to saved adapter
            result: Training result
            base_model_id: HuggingFace model ID
            config: Training configuration
        """
        metadata = {
            "model_key": result.model_key,
            "base_model_id": base_model_id,
            "training_date": datetime.now().isoformat(),
            "epochs_trained": result.epochs_trained,
            "early_stopped": result.early_stopped,
            "best_val_loss": result.best_val_loss,
            "optimal_threshold": result.optimal_threshold,
            "metrics": {
                "f1_score": result.final_f1,
                "precision": result.final_precision,
                "recall": result.final_recall,
                "accuracy": result.final_accuracy,
            },
            "config": config,
        }

        metadata_path = os.path.join(adapter_path, "training_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved training metadata to: {metadata_path}")

    def log_adapter_to_mlflow(
        self,
        adapter_path: str,
        result: TrainingResult,
        run_id: Optional[str] = None,
    ) -> str:
        """Log adapter directory to MLFlow as artifact."""
        artifact_name = f"lora_adapter_{result.model_key}"

        # Log the entire adapter directory
        mlflow.log_artifacts(adapter_path, artifact_name)

        print(f"Logged adapter to MLFlow: {artifact_name}")

        return artifact_name

    def register_adapter(
        self,
        result: TrainingResult,
        run_id: str,
        artifact_path: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> dict:
        """Register adapter in MLFlow Model Registry with metrics and tags."""
        metrics = {
            "f1_score": result.final_f1,
            "precision": result.final_precision,
            "recall": result.final_recall,
            "accuracy": result.final_accuracy,
            "optimal_threshold": result.optimal_threshold,
        }

        all_tags = {
            "epochs_trained": str(result.epochs_trained),
            "early_stopped": str(result.early_stopped).lower(),
            "best_val_loss": f"{result.best_val_loss:.4f}",
        }

        if tags:
            all_tags.update(tags)

        return self.registry_client.register_lora_adapter(
            model_key=result.model_key,
            adapter_path=artifact_path,
            run_id=run_id,
            metrics=metrics,
            tags=all_tags,
        )

    def promote_to_staging(self, model_key: str, version: Optional[int] = None):
        """Promote adapter to staging."""
        self.registry_client.promote_to_staging(model_key, version)

    def promote_to_production(self, model_key: str, version: Optional[int] = None):
        """Promote adapter to production."""
        self.registry_client.promote_to_production(model_key, version)

    def get_production_adapter_uri(self, model_key: str) -> Optional[str]:
        """Get URI for production adapter."""
        return self.registry_client.get_production_model_uri(model_key)

    def cleanup_local_adapter(self, adapter_path: str):
        """Remove local adapter directory after registration."""
        if os.path.exists(adapter_path):
            shutil.rmtree(adapter_path)
            print(f"Cleaned up local adapter: {adapter_path}")


def save_and_register_adapter(
    result: TrainingResult,
    base_model_id: str,
    config: dict,
    run_id: str,
    cleanup_local: bool = True,
) -> Optional[dict]:
    """
    Convenience function to save and register an adapter.

    Args:
        result: Training result
        base_model_id: HuggingFace model ID
        config: Training configuration
        run_id: MLFlow run ID
        cleanup_local: Remove local files after registration

    Returns:
        Registration info or None if failed
    """
    if not result.success or not result.adapter_path:
        print(f"Cannot register failed training: {result.error}")
        return None

    saver = LoRAModelSaver()

    try:
        # Save metadata
        saver.save_training_metadata(
            result.adapter_path, result, base_model_id, config
        )

        # Log to MLFlow
        artifact_path = saver.log_adapter_to_mlflow(result.adapter_path, result)

        # Register in model registry
        registration = saver.register_adapter(result, run_id, artifact_path)

        # Cleanup
        if cleanup_local:
            saver.cleanup_local_adapter(result.adapter_path)

        return registration

    except Exception as e:
        print(f"Error registering adapter: {e}")
        return None
