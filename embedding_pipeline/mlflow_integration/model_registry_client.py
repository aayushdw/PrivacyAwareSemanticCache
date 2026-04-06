"""MLFlow Model Registry client for LoRA adapter versioning."""

from typing import Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from ..config.pipeline_config import get_config


class ModelRegistryClient:
    """Register and manage LoRA adapters with versioning and stage transitions."""

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize model registry client with optional tracking URI."""
        config = get_config()
        self.tracking_uri = tracking_uri or config.mlflow.tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()

    @staticmethod
    def get_model_name(model_key: str) -> str:
        """Get standardized model name for registry."""
        clean_key = model_key.lower().replace(" ", "-").replace("_", "-")
        return f"semantic-cache-{clean_key}-lora"

    def register_lora_adapter(
        self,
        model_key: str,
        adapter_path: str,
        run_id: str,
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
    ) -> dict:
        """
        Register a LoRA adapter in the model registry.

        Args:
            model_key: Model key identifier
            adapter_path: Path to saved LoRA adapter
            run_id: MLFlow run ID that created the adapter
            metrics: Performance metrics
            tags: Additional tags

        Returns:
            Dictionary with model_name and version
        """
        model_name = self.get_model_name(model_key)

        # Log the model to the registry
        # Using the artifact path from the run
        model_uri = f"runs:/{run_id}/{adapter_path}"

        try:
            # Register the model
            result = mlflow.register_model(model_uri, model_name)
            version = result.version

            # Add version tags
            all_tags = {
                "base_model": model_key,
                "f1_score": str(metrics.get("f1_score", "")),
                "precision": str(metrics.get("precision", "")),
                "recall": str(metrics.get("recall", "")),
                "training_run_id": run_id,
            }

            if tags:
                all_tags.update(tags)

            for key, value in all_tags.items():
                self.client.set_model_version_tag(model_name, version, key, value)

            print(f"Registered {model_name} version {version}")

            return {"model_name": model_name, "version": version}

        except Exception as e:
            print(f"Error registering model: {e}")
            raise

    def promote_to_staging(self, model_key: str, version: Optional[int] = None):
        """Promote model version to Staging stage."""
        model_name = self.get_model_name(model_key)

        if version is None:
            version = self._get_latest_version(model_name)

        if version is None:
            raise ValueError(f"No versions found for model: {model_name}")

        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
        )

        print(f"Promoted {model_name} v{version} to Staging")

    def promote_to_production(self, model_key: str, version: Optional[int] = None):
        """
        Promote a model version to Production.

        Args:
            model_key: Model key identifier
            version: Specific version (defaults to current staging)
        """
        model_name = self.get_model_name(model_key)

        if version is None:
            # Get current staging version
            staging_versions = self.client.get_latest_versions(model_name, stages=["Staging"])
            if staging_versions:
                version = staging_versions[0].version
            else:
                raise ValueError(f"No staging version found for: {model_name}")

        # Archive current production version if exists
        prod_versions = self.client.get_latest_versions(model_name, stages=["Production"])
        for pv in prod_versions:
            self.client.transition_model_version_stage(
                name=model_name,
                version=pv.version,
                stage="Archived",
            )
            print(f"Archived previous production version: {model_name} v{pv.version}")

        # Promote to production
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
        )

        print(f"Promoted {model_name} v{version} to Production")

    def get_production_model_uri(self, model_key: str) -> Optional[str]:
        """Get URI for production version of model."""
        model_name = self.get_model_name(model_key)
        prod_versions = self.client.get_latest_versions(model_name, stages=["Production"])

        if prod_versions:
            return f"models:/{model_name}/Production"
        return None

    def get_staging_model_uri(self, model_key: str) -> Optional[str]:
        """Get URI for staging version of model."""
        model_name = self.get_model_name(model_key)
        staging_versions = self.client.get_latest_versions(model_name, stages=["Staging"])

        if staging_versions:
            return f"models:/{model_name}/Staging"
        return None

    def list_model_versions(self, model_key: str) -> List[dict]:
        """List all versions of a model with metadata."""
        model_name = self.get_model_name(model_key)

        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "run_id": v.run_id,
                    "tags": dict(v.tags) if v.tags else {},
                }
                for v in versions
            ]
        except Exception:
            return []

    def get_model_info(self, model_key: str, stage: str = "Production") -> Optional[dict]:
        """Get information about model at specific stage."""
        model_name = self.get_model_name(model_key)

        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if versions:
                v = versions[0]
                return {
                    "model_name": model_name,
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "source": v.source,
                    "tags": dict(v.tags) if v.tags else {},
                }
        except Exception:
            pass

        return None

    def delete_model_version(self, model_key: str, version: int):
        """Delete a specific model version."""
        model_name = self.get_model_name(model_key)
        self.client.delete_model_version(model_name, version)
        print(f"Deleted {model_name} v{version}")

    def _get_latest_version(self, model_name: str) -> Optional[int]:
        """Get the latest version number for a model."""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if versions:
                return max(int(v.version) for v in versions)
        except Exception:
            pass
        return None

    def list_registered_models(self) -> List[str]:
        """List all registered semantic cache models."""
        models = []
        for rm in self.client.search_registered_models():
            if rm.name.startswith("semantic-cache-"):
                models.append(rm.name)
        return models
