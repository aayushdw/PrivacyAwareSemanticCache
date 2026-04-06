"""Prefect flow for model ranking stage."""

from typing import List, Optional

import mlflow
from prefect import flow, task, get_run_logger

from ..config.pipeline_config import get_config, get_experiment_name
from ..evaluation.embedding_evaluator import EvaluationResult
from ..mlflow_integration.experiment_manager import ExperimentManager
from ..mlflow_integration.run_logger import RunLogger
from ..ranking.comparison_report import ComparisonReportGenerator
from ..ranking.model_ranker import ModelRanker, RankedModel


@task(name="rank-models", log_prints=True)
def rank_models_task(
    evaluation_results: List[EvaluationResult],
    min_precision: float,
    top_n: int,
) -> List[RankedModel]:
    """Rank models by F1 score with precision constraint."""
    logger = get_run_logger()

    ranker = ModelRanker(min_precision=min_precision)
    ranked = ranker.rank_models(evaluation_results, top_n=top_n)

    ranker.print_ranking(ranked)

    logger.info(f"Ranked {len(ranked)} models for LoRA training")
    for model in ranked:
        logger.info(
            f"  Rank {model.rank}: {model.model_name} "
            f"(F1={model.f1_score:.4f}, P={model.precision:.4f})"
        )

    return ranked


@task(name="generate-comparison-report", log_prints=True)
def generate_report_task(
    ranked_models: List[RankedModel],
    all_results: List[EvaluationResult],
    min_precision: float,
) -> str:
    """Generate comparison report for ranked models."""
    logger = get_run_logger()

    generator = ComparisonReportGenerator(min_precision=min_precision)
    report = generator.generate_markdown_report(ranked_models, all_results)

    logger.info("Generated comparison report")

    return report


@task(name="log-ranking-to-mlflow", log_prints=True)
def log_ranking_task(
    ranked_models: List[RankedModel],
    report_content: str,
    min_precision: float,
    top_n: int,
):
    """Log ranking results to MLFlow."""
    logger = get_run_logger()

    # Set up experiment
    exp_manager = ExperimentManager()
    exp_manager.set_experiment("ranking")

    run_logger = RunLogger()

    # Start run
    with mlflow.start_run(run_name="model_ranking"):
        # Log ranking results
        RunLogger.log_ranking_results(
            selected_models=[m.to_dict() for m in ranked_models],
            min_precision=min_precision,
            top_n=top_n,
        )

        # Log report as artifact
        run_logger.log_comparison_report(report_content)

        logger.info("Logged ranking results to MLFlow")


@flow(name="rank-and-select", log_prints=True)
def rank_and_select_flow(
    evaluation_results: List[EvaluationResult],
    min_precision: Optional[float] = None,
    top_n: Optional[int] = None,
) -> List[RankedModel]:
    """
    Rank evaluation results and select top-N models for LoRA training.

    Args:
        evaluation_results: Results from evaluation flow
        min_precision: Minimum precision (defaults to config)
        top_n: Number of models to select (defaults to config)

    Returns:
        List of selected RankedModel
    """
    logger = get_run_logger()
    config = get_config()

    min_precision = min_precision or config.ranking.min_precision
    top_n = top_n or config.ranking.top_n

    logger.info(f"Ranking {len(evaluation_results)} models")
    logger.info(f"Precision constraint: >= {min_precision}")
    logger.info(f"Selecting top {top_n} models")

    # Filter to successful evaluations
    successful = [r for r in evaluation_results if r.success]
    logger.info(f"Successful evaluations: {len(successful)}")

    if not successful:
        logger.error("No successful evaluations to rank!")
        return []

    # Rank models
    ranked_models = rank_models_task(
        evaluation_results=successful,
        min_precision=min_precision,
        top_n=top_n,
    )

    # Generate report
    report = generate_report_task(
        ranked_models=ranked_models,
        all_results=evaluation_results,
        min_precision=min_precision,
    )

    # Log to MLFlow
    log_ranking_task(
        ranked_models=ranked_models,
        report_content=report,
        min_precision=min_precision,
        top_n=top_n,
    )

    logger.info(f"\nSelected {len(ranked_models)} models for LoRA training:")
    for model in ranked_models:
        logger.info(f"  - {model.model_name}")

    return ranked_models
