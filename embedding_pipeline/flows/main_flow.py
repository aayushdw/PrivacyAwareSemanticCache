"""Main orchestration flow for the embedding model evaluation pipeline."""

from typing import Dict, List, Optional

from prefect import flow, get_run_logger

from ..config.pipeline_config import get_config
from ..evaluation.embedding_evaluator import EvaluationResult
from ..lora_training.lora_trainer import TrainingResult
from ..ranking.model_ranker import RankedModel
from ..registry.model_info import ModelInfo
from ..registry.model_registry import get_models_for_evaluation
from .evaluation_flow import evaluate_models_flow
from .lora_training_flow import lora_training_flow
from .ranking_flow import rank_and_select_flow


@flow(
    name="embedding-model-evaluation-pipeline",
    description="Multi-stage pipeline: evaluate models, rank by F1, train LoRA on top-N",
    log_prints=True,
    retries=0,
)
def embedding_evaluation_pipeline(
    model_categories: Optional[List[str]] = None,
    min_precision: Optional[float] = None,
    top_n: Optional[int] = None,
    skip_evaluation: bool = False,
    skip_lora_training: bool = False,
    max_train_samples: int = 5000,
    max_test_samples: int = 5000,
) -> Dict:
    """
    Main orchestration flow for embedding model evaluation pipeline.

    Stages:
    1. Load candidate models from registry
    2. Evaluate each model with threshold tuning
    3. Rank models and select top-N
    4. Fine-tune top-N with LoRA
    5. Register LoRA adapters in MLFlow Model Registry

    Args:
        model_categories: Model categories to evaluate (fast, balanced, quality)
        min_precision: Minimum precision constraint (default: 0.80)
        top_n: Number of models for LoRA training (default: 3)
        skip_evaluation: Skip evaluation stage (use cached results)
        skip_lora_training: Skip LoRA training stage
        max_train_samples: Max samples for threshold tuning
        max_test_samples: Max samples for testing

    Returns:
        Dictionary with pipeline results
    """
    logger = get_run_logger()
    config = get_config()

    # Apply defaults from config
    model_categories = model_categories or config.model_categories
    min_precision = min_precision or config.evaluation.min_precision
    top_n = top_n or config.ranking.top_n

    logger.info("=" * 70)
    logger.info("EMBEDDING MODEL EVALUATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Categories: {model_categories}")
    logger.info(f"Precision constraint: >= {min_precision}")
    logger.info(f"Top-N for LoRA: {top_n}")
    logger.info(f"Skip evaluation: {skip_evaluation}")
    logger.info(f"Skip LoRA training: {skip_lora_training}")
    logger.info("=" * 70)

    # Initialize result storage
    evaluation_results: List[EvaluationResult] = []
    selected_models: List[RankedModel] = []
    lora_results: List[TrainingResult] = []

    # =========================================================================
    # Stage 1: Get candidate models
    # =========================================================================
    logger.info("\n[Stage 1] Loading candidate models...")

    models = get_models_for_evaluation(
        categories=model_categories,
        require_lora=True,
        require_mps=True,
    )

    logger.info(f"Found {len(models)} candidate models")
    for model in models:
        logger.info(f"  - {model.name} ({model.category}, {model.dimension}d)")

    # =========================================================================
    # Stage 2: Evaluate models
    # =========================================================================
    if not skip_evaluation:
        logger.info("\n[Stage 2] Evaluating models...")

        evaluation_results = evaluate_models_flow(
            models=models,
            min_precision=min_precision,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
        )

        successful = [r for r in evaluation_results if r.success]
        logger.info(f"Evaluation complete: {len(successful)}/{len(evaluation_results)} successful")
    else:
        logger.info("\n[Stage 2] Skipping evaluation (using cached results)...")
        # In a real implementation, you would load cached results here
        logger.warning("No cached results available - evaluation results will be empty")

    # =========================================================================
    # Stage 3: Rank and select top-N
    # =========================================================================
    if evaluation_results:
        logger.info("\n[Stage 3] Ranking models and selecting top-N...")

        selected_models = rank_and_select_flow(
            evaluation_results=evaluation_results,
            min_precision=min_precision,
            top_n=top_n,
        )

        logger.info(f"Selected {len(selected_models)} models for LoRA training")
    else:
        logger.warning("No evaluation results - skipping ranking stage")

    # =========================================================================
    # Stage 4: LoRA fine-tuning
    # =========================================================================
    if selected_models and not skip_lora_training:
        logger.info("\n[Stage 4] Training LoRA adapters...")

        lora_results = lora_training_flow(selected_models=selected_models)

        successful_lora = [r for r in lora_results if r.success]
        logger.info(f"LoRA training complete: {len(successful_lora)}/{len(lora_results)} successful")
    elif skip_lora_training:
        logger.info("\n[Stage 4] Skipping LoRA training...")
    else:
        logger.warning("No selected models - skipping LoRA training")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    summary = {
        "models_evaluated": len([r for r in evaluation_results if r.success]),
        "models_selected": len(selected_models),
        "lora_trained": len([r for r in lora_results if r.success]),
    }

    if evaluation_results:
        successful = [r for r in evaluation_results if r.success]
        if successful:
            best_eval = max(successful, key=lambda r: r.f1_score)
            summary["best_evaluation"] = {
                "model": best_eval.model_name,
                "f1_score": best_eval.f1_score,
                "precision": best_eval.precision,
            }

    if lora_results:
        successful_lora = [r for r in lora_results if r.success]
        if successful_lora:
            best_lora = max(successful_lora, key=lambda r: r.final_f1)
            summary["best_lora"] = {
                "model": best_lora.model_key,
                "f1_score": best_lora.final_f1,
                "precision": best_lora.final_precision,
            }

    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    logger.info("=" * 70)

    return {
        "summary": summary,
        "evaluation_results": evaluation_results,
        "selected_models": selected_models,
        "lora_results": lora_results,
    }


@flow(name="evaluation-only", log_prints=True)
def evaluation_only_pipeline(
    model_categories: Optional[List[str]] = None,
    min_precision: float = 0.80,
) -> List[EvaluationResult]:
    """Run only the evaluation stage (no LoRA training)."""
    return embedding_evaluation_pipeline(
        model_categories=model_categories,
        min_precision=min_precision,
        skip_lora_training=True,
    )["evaluation_results"]


@flow(name="quick-test-pipeline", log_prints=True)
def quick_test_pipeline() -> Dict:
    """Quick test with minimal models (for development/testing)."""
    return embedding_evaluation_pipeline(
        model_categories=["fast"],
        top_n=1,
        max_train_samples=500,
        max_test_samples=500,
    )


# Entry point for running the pipeline
if __name__ == "__main__":
    # Run the full pipeline
    result = embedding_evaluation_pipeline()
    print(f"\nPipeline completed: {result['summary']}")
