from prefect import flow, task
import mlflow
from mlflow.tracking import MlflowClient
import os
import sys
import time
import hashlib
import numpy as np
from pathlib import Path
from data_loader import load_data
from engine import get_model, get_similarity_scores, get_model_info
from metrics import calculate_cache_metrics
from visualization import generate_all_plots

import constants

from model_evaluation_pipeline.model_registry import MODEL_REGISTRY

# --- Cache Key Helper Functions ---

def hash_array(arr) -> str:
    """Hash numpy array or list for cache key."""
    if isinstance(arr, np.ndarray):
        return hashlib.md5(arr.tobytes()).hexdigest()[:16]
    return hashlib.md5(str(arr).encode()).hexdigest()[:16]

def get_model_cache_identifier(model_name: str) -> str:
    """
    Get model identifier for cache invalidation.
    Returns model name + modification time of cached weights.
    This detects when model weights are re-downloaded/updated.
    """
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_path = cache_dir / f"models--{model_name.replace('/', '--')}"

    if model_cache_path.exists():
        # Use most recent modification time in model directory
        mtimes = [p.stat().st_mtime for p in model_cache_path.rglob('*') if p.is_file()]
        if mtimes:
            return f"{model_name}:{max(mtimes):.0f}"

    # Fallback if model not cached yet
    return model_name

def inference_cache_key(context, parameters):
    """
    Cache key for inference: model identifier + data hash.
    Invalidates when:
    - Model weights change (detected via HF cache mtime)
    - Input data changes (detected via array hash)
    """
    model_name = parameters.get('model_name')
    q1 = parameters.get('q1')
    q2 = parameters.get('q2')

    model_id = get_model_cache_identifier(model_name)
    data_hash = hash_array(q1) + hash_array(q2)

    return f"inference:{model_id}:{data_hash}"

# --- Prefect Tasks ---

@task(name="Load Data", retries=2)
def load_data_task(filepath):
    return load_data(filepath)

@task(name="Run Inference", cache_key_fn=inference_cache_key, persist_result=True)
def inference_task(model_name, q1, q2):
    model = get_model(model_name)

    # Time the inference
    start_time = time.perf_counter()
    scores = get_similarity_scores(model, q1, q2)
    end_time = time.perf_counter()

    # Calculate timing metrics
    total_time = end_time - start_time
    latency_ms = (total_time / len(q1)) * 1000

    # Get model info
    model_info = get_model_info(model)

    perf_metrics = {
        "inference_total_time_sec": total_time,
        "inference_latency_ms_per_pair": latency_ms,
        "total_pairs": len(q1),
        **model_info
    }

    return scores, perf_metrics

@task(name="Calculate Metrics")
def eval_task(labels, scores, min_precision):
    return calculate_cache_metrics(labels, scores, min_precision=min_precision)

@task(name="Register Models in MLflow")
def register_models_task(stage="Development"):
    """
    Register all models from MODEL_REGISTRY in MLflow Model Registry.
    Sets all models to specified stage (default: 'Development').

    Args:
        stage: Stage tag to assign ('Development', 'Production', etc.)

    Returns:
        dict: Summary of registration (registered, already_exists, errors)
    """
    client = MlflowClient()

    summary = {
        "registered": [],
        "already_exists": [],
        "errors": []
    }

    for model_key, model_info in MODEL_REGISTRY.items():
        try:
            # Use model_id as the registered model name
            registered_model_name = model_info.model_id

            # Check if model already exists
            try:
                client.get_registered_model(registered_model_name)
                print(f"Model already registered: {registered_model_name}")
                summary["already_exists"].append(model_key)

                # Update tags/description even if already exists
                client.update_registered_model(
                    name=registered_model_name,
                    description=model_info.description
                )

                # Set/update tags
                client.set_registered_model_tag(registered_model_name, "dimension", str(model_info.dimension))
                client.set_registered_model_tag(registered_model_name, "category", model_info.category)
                client.set_registered_model_tag(registered_model_name, "model_key", model_key)
                client.set_registered_model_tag(registered_model_name, "display_name", model_info.name)
                client.set_registered_model_tag(registered_model_name, "stage", stage)

            except mlflow.exceptions.RestException:
                # Model doesn't exist, create it
                print(f"Registering new model: {registered_model_name}")

                # Create registered model with tags
                client.create_registered_model(
                    name=registered_model_name,
                    tags={
                        "dimension": str(model_info.dimension),
                        "category": model_info.category,
                        "model_key": model_key,
                        "display_name": model_info.name,
                        "stage": stage
                    },
                    description=model_info.description
                )

                summary["registered"].append(model_key)

        except Exception as e:
            print(f"Error registering {model_key}: {e}")
            summary["errors"].append((model_key, str(e)))

    print(f"\nRegistration Summary:")
    print(f"  Newly registered: {len(summary['registered'])}")
    print(f"  Already exists: {len(summary['already_exists'])}")
    print(f"  Errors: {len(summary['errors'])}")
    print(f"  Stage: {stage}")

    return summary

@task(name="Get Models by Stage")
def get_models_by_stage_task(stage="Development"):
    """
    Fetch models from MLflow Model Registry filtered by stage.

    Args:
        stage: Stage tag to filter by ('Development', 'Production', etc.)

    Returns:
        list: List of model IDs (model_id from registry) that match the stage
    """
    client = MlflowClient()

    matching_models = []

    try:
        # Get all registered models
        registered_models = client.search_registered_models()

        for rm in registered_models:
            # Get tags for this model
            model_tags = {tag.key: tag.value for tag in rm.tags}

            # Filter by stage
            if model_tags.get("stage") == stage:
                matching_models.append(rm.name)

        print(f"Found {len(matching_models)} models with stage='{stage}'")

        return matching_models

    except Exception as e:
        print(f"Error fetching models from MLflow: {e}")
        raise

# --- The Main Flow ---

@flow(name="Semantic Cache Eval Pipeline", log_prints=True)
def run_evaluation(model_name: str, dataset_path: str, min_precision: float = 0.75):

    try:
        # 1. Setup MLflow Connection
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(mlflow_uri)

        # Get or create experiment
        experiment_name = "Semantic_Cache_Evaluation_v2"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        print(f"Starting Run for: {model_name}")
        print(f"MLflow URI: {mlflow_uri}")

        # Validate dataset path exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

        # 2. Start MLflow Run
        with mlflow.start_run(run_name=f"eval-{model_name.replace('/', '-')}"):

            # Log Parameters (Configuration)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("dataset", dataset_path)
            mlflow.log_param("target_precision", min_precision)

            # 3. Execute Pipeline Steps
            q1, q2, labels = load_data_task(dataset_path)
            print(f"Loaded {len(labels)} question pairs")

            scores, perf_metrics = inference_task(model_name, q1, q2)
            print(f"Computed similarity scores")

            metrics = eval_task(labels, scores, min_precision)

            # 4. Log Performance Metrics
            mlflow.log_metric("inference_time_sec", perf_metrics["inference_total_time_sec"])
            mlflow.log_metric("latency_ms_per_pair", perf_metrics["inference_latency_ms_per_pair"])
            mlflow.log_metric("model_size_mb", perf_metrics["model_size_mb"])
            mlflow.log_metric("model_parameters_m", perf_metrics["model_parameters_millions"])

            # 5. Log Extended Classification Metrics
            mlflow.log_metric("optimal_threshold", metrics["optimal_threshold"])
            mlflow.log_metric("recall_at_target_precision", metrics["recall_at_target"])
            mlflow.log_metric("precision_at_target", metrics["precision_at_target"])
            mlflow.log_metric("ap_score", metrics["ap_score"])
            mlflow.log_metric("roc_auc", metrics["roc_auc"])
            mlflow.log_metric("f1_score", metrics["f1_score"])
            mlflow.log_metric("accuracy", metrics["accuracy"])

            # 6. Log Confusion Matrix Components
            mlflow.log_metric("true_positives", metrics["true_positives"])
            mlflow.log_metric("true_negatives", metrics["true_negatives"])
            mlflow.log_metric("false_positives", metrics["false_positives"])
            mlflow.log_metric("false_negatives", metrics["false_negatives"])

            # 7. Generate and Log Visualizations
            print("Generating visualizations...")
            plot_paths = generate_all_plots(metrics, model_name)

            mlflow.log_artifact(plot_paths['pr_curve'], "plots")
            mlflow.log_artifact(plot_paths['roc_curve'], "plots")
            mlflow.log_artifact(plot_paths['confusion_matrix'], "plots")

            print(f"Logged 3 visualization artifacts to MLflow")

            # 8. Print Summary
            print(f"Results Logged:")
            print(f"  Threshold: {metrics['optimal_threshold']:.4f}")
            print(f"  AP: {metrics['ap_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Recall@P={min_precision}: {metrics['recall_at_target']:.4f}")
            print(f"  Latency: {perf_metrics['inference_latency_ms_per_pair']:.2f} ms/pair")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        raise
    except Exception as e:
        print(f"ERROR during evaluation: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    # Setup MLflow connection
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_uri)

    # Step 1: Register all models from model_registry.py as 'Development' stage
    print("=" * 80)
    print("STEP 1: Registering models in MLflow Model Registry")
    print("=" * 80)
    register_models_task(stage="Development")

    # Step 2: Fetch all 'Development' models from MLflow
    print("\n" + "=" * 80)
    print("STEP 2: Fetching 'Development' models from MLflow")
    print("=" * 80)
    models_to_test = get_models_by_stage_task(stage="Development")

    # Step 3: Run evaluation on all Development models
    print("\n" + "=" * 80)
    print(f"STEP 3: Running evaluation on {len(models_to_test)} models")
    print("=" * 80)

    for model in models_to_test:
        print(f"\n{'='*80}")
        print(f"Evaluating: {model}")
        print(f"{'='*80}")
        run_evaluation(model, constants.SMALL_TRAIN_PATH, min_precision=0.8)