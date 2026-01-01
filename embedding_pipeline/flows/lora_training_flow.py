"""Prefect flow for LoRA training stage."""

from typing import List, Optional

import mlflow
from prefect import flow, task, get_run_logger

from ..config.pipeline_config import get_config, get_data_path
from ..lora_training.lora_trainer import LoRATrainer, LoRATrainingConfig, TrainingResult
from ..lora_training.model_saver import LoRAModelSaver, save_and_register_adapter
from ..mlflow_integration.experiment_manager import ExperimentManager
from ..ranking.model_ranker import RankedModel
from ..registry.model_registry import get_model_info
from ..utils.device_utils import clear_gpu_memory


@task(
    name="train-lora-adapter",
    retries=1,
    timeout_seconds=7200,  # 2 hour timeout
    log_prints=True,
    tags=["training", "gpu", "lora"],
)
def train_lora_adapter_task(
    ranked_model: RankedModel,
    train_csv: str,
    val_csv: str,
    config: LoRATrainingConfig,
) -> TrainingResult:
    """
    Train LoRA adapter for a single model.

    Args:
        ranked_model: Model to fine-tune
        train_csv: Training data path
        val_csv: Validation data path
        config: Training configuration

    Returns:
        TrainingResult
    """
    logger = get_run_logger()
    logger.info(f"Training LoRA for: {ranked_model.model_name}")

    # Get model info
    model_info = get_model_info(ranked_model.model_key)

    # Train
    trainer = LoRATrainer(model_info=model_info, config=config)
    result = trainer.train(train_csv, val_csv)

    if result.success:
        logger.info(
            f"Training completed: {ranked_model.model_name} "
            f"(F1={result.final_f1:.4f}, epochs={result.epochs_trained})"
        )
    else:
        logger.error(f"Training failed: {ranked_model.model_name}: {result.error}")

    clear_gpu_memory()

    return result


@task(name="log-training-to-mlflow", log_prints=True)
def log_training_task(
    result: TrainingResult,
    model_info_dict: dict,
    config_dict: dict,
) -> Optional[str]:
    """
    Log training results to MLFlow and register adapter.

    Args:
        result: Training result
        model_info_dict: Model info dictionary
        config_dict: Training config dictionary

    Returns:
        Run ID or None if failed
    """
    logger = get_run_logger()

    if not result.success:
        logger.warning(f"Skipping MLFlow logging for failed training: {result.model_key}")
        return None

    # Set up experiment
    exp_manager = ExperimentManager()
    exp_manager.set_experiment("lora_training")

    run_id = None

    with mlflow.start_run(run_name=f"{result.model_key}_lora") as run:
        run_id = run.info.run_id

        # Log parameters
        mlflow.log_params({
            "base_model": model_info_dict.get("model_id", ""),
            "model_key": result.model_key,
            "lora_r": config_dict.get("lora_r", 16),
            "lora_alpha": config_dict.get("lora_alpha", 32),
            "lora_dropout": config_dict.get("lora_dropout", 0.1),
            "learning_rate": config_dict.get("learning_rate", 2e-4),
            "batch_size": config_dict.get("batch_size", 64),
            "max_epochs": config_dict.get("max_epochs", 10),
            "patience": config_dict.get("patience", 3),
        })

        # Log training history
        for epoch_data in result.history:
            mlflow.log_metrics(
                {
                    "train_loss": epoch_data["train_loss"],
                    "val_loss": epoch_data["val_loss"],
                },
                step=epoch_data["epoch"],
            )

        # Log final metrics
        mlflow.log_metrics({
            "final_f1": result.final_f1,
            "final_precision": result.final_precision,
            "final_recall": result.final_recall,
            "final_accuracy": result.final_accuracy,
            "optimal_threshold": result.optimal_threshold,
            "epochs_trained": result.epochs_trained,
            "best_val_loss": result.best_val_loss,
        })

        # Log tags
        mlflow.set_tag("early_stopped", str(result.early_stopped).lower())
        mlflow.set_tag("training_success", "true")

        # Git info
        git_info = exp_manager.get_git_info()
        for key, value in git_info.items():
            mlflow.set_tag(key, str(value))

        # Register adapter
        if result.adapter_path:
            registration = save_and_register_adapter(
                result=result,
                base_model_id=model_info_dict.get("model_id", ""),
                config=config_dict,
                run_id=run_id,
                cleanup_local=True,
            )

            if registration:
                logger.info(
                    f"Registered {registration['model_name']} "
                    f"v{registration['version']}"
                )

    return run_id


@flow(name="lora-training", log_prints=True)
def lora_training_flow(
    selected_models: List[RankedModel],
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None,
    config: Optional[LoRATrainingConfig] = None,
) -> List[TrainingResult]:
    """
    Train LoRA adapters for selected models.

    Runs sequentially to avoid OOM on M4 MacBook Air.

    Args:
        selected_models: Models selected for LoRA training
        train_csv: Training data path (defaults to config)
        val_csv: Validation data path (defaults to config)
        config: Training configuration (defaults to config)

    Returns:
        List of TrainingResult
    """
    logger = get_run_logger()
    pipeline_config = get_config()

    train_csv = train_csv or str(get_data_path(pipeline_config.lora.train_data))
    val_csv = val_csv or str(get_data_path(pipeline_config.lora.val_data))
    config = config or LoRATrainingConfig.from_pipeline_config()

    logger.info(f"Training LoRA adapters for {len(selected_models)} models")
    logger.info(f"Training data: {train_csv}")
    logger.info(f"Validation data: {val_csv}")
    logger.info(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}")

    results = []

    for ranked_model in selected_models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model {ranked_model.rank}/{len(selected_models)}: "
                   f"{ranked_model.model_name}")
        logger.info(f"{'='*60}")

        # Get model info for logging
        model_info = get_model_info(ranked_model.model_key)

        # Train
        result = train_lora_adapter_task(
            ranked_model=ranked_model,
            train_csv=train_csv,
            val_csv=val_csv,
            config=config,
        )

        # Log to MLFlow
        log_training_task(
            result=result,
            model_info_dict=model_info.to_dict(),
            config_dict={
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "max_epochs": config.max_epochs,
                "patience": config.patience,
            },
        )

        results.append(result)

        # Clear memory between models
        clear_gpu_memory()

    # Summary
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    logger.info(f"\nLoRA Training Summary:")
    logger.info(f"  Total: {len(results)}")
    logger.info(f"  Successful: {len(successful)}")
    logger.info(f"  Failed: {len(failed)}")

    if successful:
        best = max(successful, key=lambda r: r.final_f1)
        logger.info(f"  Best: {best.model_key} (F1={best.final_f1:.4f})")

    return results
