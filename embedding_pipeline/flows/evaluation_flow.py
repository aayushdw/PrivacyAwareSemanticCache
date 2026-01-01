"""Prefect flow for model evaluation stage."""

from typing import List, Optional

import mlflow
from prefect import flow, task, get_run_logger

from ..config.pipeline_config import get_config, get_data_path
from ..evaluation.embedding_evaluator import EmbeddingEvaluator, EvaluationResult
from ..mlflow_integration.experiment_manager import ExperimentManager
from ..mlflow_integration.run_logger import RunLogger
from ..registry.model_info import ModelInfo
from ..registry.model_registry import get_models_for_evaluation
from ..utils.data_loader import load_dataset
from ..utils.device_utils import clear_gpu_memory


@task(
    name="evaluate-single-model",
    retries=2,
    retry_delay_seconds=30,
    log_prints=True,
    tags=["evaluation", "gpu"],
)
def evaluate_single_model_task(
    model_info: ModelInfo,
    train_df_path: str,
    test_df_path: str,
    min_precision: float,
    max_train_samples: int,
    max_test_samples: int,
) -> EvaluationResult:
    """
    Evaluate a single model with threshold tuning.

    Args:
        model_info: Model to evaluate
        train_df_path: Path to training data
        test_df_path: Path to test data
        min_precision: Minimum precision constraint
        max_train_samples: Max samples for threshold tuning
        max_test_samples: Max samples for testing

    Returns:
        EvaluationResult
    """
    logger = get_run_logger()
    logger.info(f"Evaluating model: {model_info.name}")

    # Load data
    train_df = load_dataset(train_df_path, max_samples=max_train_samples)
    test_df = load_dataset(test_df_path, max_samples=max_test_samples)

    # Set up MLFlow experiment
    exp_manager = ExperimentManager()
    exp_manager.set_experiment("threshold_tuning")

    # Initialize logger
    run_logger = RunLogger()

    # Start MLFlow run
    run_id = run_logger.start_evaluation_run(
        model_info=model_info,
        tags={"min_precision_constraint": str(min_precision)},
    )

    try:
        # Evaluate
        evaluator = EmbeddingEvaluator()
        result = evaluator.evaluate_model(
            model_info=model_info,
            train_df=train_df,
            test_df=test_df,
            min_precision=min_precision,
        )

        # Log results
        run_logger.log_evaluation_result(result, min_precision)

        logger.info(
            f"Model {model_info.name}: F1={result.f1_score:.4f}, "
            f"P={result.precision:.4f}, R={result.recall:.4f}"
        )

        run_logger.end_run("FINISHED" if result.success else "FAILED")

    except Exception as e:
        logger.error(f"Evaluation failed for {model_info.name}: {e}")
        run_logger.end_run("FAILED")

        result = EvaluationResult(
            model_key=model_info.name.lower().replace(" ", "-"),
            model_name=model_info.name,
            model_id=model_info.model_id,
            dimension=model_info.dimension,
            category=model_info.category,
            success=False,
            error=str(e),
        )

    finally:
        clear_gpu_memory()

    return result


@flow(name="evaluate-models", log_prints=True)
def evaluate_models_flow(
    models: Optional[List[ModelInfo]] = None,
    categories: Optional[List[str]] = None,
    min_precision: float = 0.80,
    max_train_samples: int = 5000,
    max_test_samples: int = 5000,
) -> List[EvaluationResult]:
    """
    Evaluate multiple embedding models.

    Args:
        models: List of models to evaluate (or use categories filter)
        categories: Model categories to include
        min_precision: Minimum precision constraint
        max_train_samples: Max samples for threshold tuning
        max_test_samples: Max samples for testing

    Returns:
        List of EvaluationResult
    """
    logger = get_run_logger()
    config = get_config()

    # Get models to evaluate
    if models is None:
        cats = categories or config.model_categories
        models = get_models_for_evaluation(
            categories=cats,
            require_lora=True,
            require_mps=True,
        )

    logger.info(f"Evaluating {len(models)} models")
    logger.info(f"Precision constraint: >= {min_precision}")

    # Get data paths
    train_path = config.evaluation.train_data
    test_path = config.evaluation.test_data

    # Check for already completed evaluations (checkpoint/resume)
    exp_manager = ExperimentManager()
    experiment_name = (
        get_config().mlflow.experiment_prefix + "/" + config.mlflow.threshold_tuning_experiment
    )
    completed = exp_manager.get_completed_model_evaluations(experiment_name)

    # Load cached results from MLFlow
    cached_results = []
    if completed:
        logger.info(f"Found {len(completed)} completed evaluations, loading cached results...")
        cached_results = exp_manager.load_cached_evaluation_results(experiment_name)
        logger.info(f"Loaded {len(cached_results)} cached evaluation results")

        # Filter out already-evaluated models
        models = [m for m in models if m.name.lower().replace(" ", "-") not in completed]
        logger.info(f"Remaining models to evaluate: {len(models)}")

    # Evaluate each model
    results = []
    for model in models:
        result = evaluate_single_model_task(
            model_info=model,
            train_df_path=train_path,
            test_df_path=test_path,
            min_precision=min_precision,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
        )
        results.append(result)

    # Combine cached and new results
    all_results = cached_results + results

    # Summary
    successful = [r for r in all_results if r.success]
    failed = [r for r in all_results if not r.success]
    newly_evaluated = len(results)
    from_cache = len(cached_results)

    logger.info(f"\nEvaluation Summary:")
    logger.info(f"  Total: {len(all_results)} ({from_cache} cached, {newly_evaluated} new)")
    logger.info(f"  Successful: {len(successful)}")
    logger.info(f"  Failed: {len(failed)}")

    if successful:
        best = max(successful, key=lambda r: r.f1_score)
        logger.info(f"  Best: {best.model_name} (F1={best.f1_score:.4f})")

    return all_results


@flow(name="quick-evaluate-models", log_prints=True)
def quick_evaluate_models_flow(
    model_keys: List[str],
    threshold: float = 0.85,
    max_samples: int = 1000,
) -> List[dict]:
    """
    Quick evaluation at fixed threshold (for rapid testing).

    Args:
        model_keys: Model keys to evaluate
        threshold: Fixed similarity threshold
        max_samples: Max samples for evaluation

    Returns:
        List of metric dictionaries
    """
    from ..registry.model_registry import get_model_info

    logger = get_run_logger()
    config = get_config()

    # Load test data
    test_df = load_dataset(config.evaluation.test_data, max_samples=max_samples)

    results = []
    evaluator = EmbeddingEvaluator()

    for model_key in model_keys:
        try:
            model_info = get_model_info(model_key)
            logger.info(f"Quick eval: {model_info.name}")

            metrics = evaluator.quick_evaluate(model_info, test_df, threshold)

            results.append({
                "model_key": model_key,
                "model_name": model_info.name,
                "threshold": threshold,
                "f1_score": metrics.f1_score,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "accuracy": metrics.accuracy,
            })

        except Exception as e:
            logger.error(f"Failed: {model_key}: {e}")
            results.append({
                "model_key": model_key,
                "error": str(e),
            })

        clear_gpu_memory()

    return results
