"""Model ranking based on evaluation results."""

from dataclasses import dataclass
from typing import List, Optional

from ..evaluation.embedding_evaluator import EvaluationResult


@dataclass
class RankedModel:
    """A ranked model with its evaluation metrics."""

    rank: int
    model_key: str
    model_name: str
    model_id: str
    dimension: int
    category: str

    # Performance metrics
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    optimal_threshold: float

    # Timing
    latency_ms_per_pair: float

    # Original evaluation result
    evaluation_result: Optional[EvaluationResult] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "rank": self.rank,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "category": self.category,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "optimal_threshold": self.optimal_threshold,
            "latency_ms_per_pair": self.latency_ms_per_pair,
        }


class ModelRanker:
    """
    Ranks models based on evaluation results.

    Filters by precision constraint and ranks by F1 score.
    """

    def __init__(self, min_precision: float = 0.80):
        """
        Initialize model ranker.

        Args:
            min_precision: Minimum precision requirement
        """
        self.min_precision = min_precision

    def rank_models(
        self,
        evaluation_results: List[EvaluationResult],
        top_n: int = 3,
    ) -> List[RankedModel]:
        """
        Rank models and select top-N.

        Args:
            evaluation_results: List of evaluation results
            top_n: Number of top models to return

        Returns:
            List of RankedModel in rank order
        """
        # Filter successful evaluations that meet precision constraint
        qualifying = [
            r for r in evaluation_results
            if r.success
            and r.meets_precision_constraint
            and r.precision >= self.min_precision
        ]

        if not qualifying:
            print(
                f"Warning: No models meet precision constraint >= {self.min_precision}. "
                "Relaxing constraint to return best available."
            )
            qualifying = [r for r in evaluation_results if r.success]

        # Sort by F1 score descending
        sorted_results = sorted(
            qualifying,
            key=lambda r: r.f1_score,
            reverse=True,
        )

        # Create ranked models
        ranked = []
        for i, result in enumerate(sorted_results[:top_n]):
            # Calculate latency
            latency = 0.0
            if result.total_samples > 0:
                latency = (result.embedding_time_seconds / result.total_samples) * 1000

            ranked_model = RankedModel(
                rank=i + 1,
                model_key=result.model_key,
                model_name=result.model_name,
                model_id=result.model_id,
                dimension=result.dimension,
                category=result.category,
                f1_score=result.f1_score,
                precision=result.precision,
                recall=result.recall,
                accuracy=result.test_metrics.accuracy if result.test_metrics else 0.0,
                optimal_threshold=result.optimal_threshold or 0.0,
                latency_ms_per_pair=latency,
                evaluation_result=result,
            )
            ranked.append(ranked_model)

        return ranked

    def get_ranking_summary(
        self,
        ranked_models: List[RankedModel],
    ) -> dict:
        """
        Get summary statistics for ranking.

        Args:
            ranked_models: Ranked model list

        Returns:
            Summary dictionary
        """
        if not ranked_models:
            return {"num_ranked": 0}

        return {
            "num_ranked": len(ranked_models),
            "best_model": ranked_models[0].model_name,
            "best_f1": ranked_models[0].f1_score,
            "best_precision": ranked_models[0].precision,
            "f1_range": (
                ranked_models[-1].f1_score,
                ranked_models[0].f1_score,
            ),
            "avg_f1": sum(m.f1_score for m in ranked_models) / len(ranked_models),
            "categories_represented": list(set(m.category for m in ranked_models)),
        }

    def print_ranking(self, ranked_models: List[RankedModel]):
        """Print ranking table to console."""
        print("\n" + "=" * 80)
        print("MODEL RANKING RESULTS")
        print("=" * 80)
        print(f"\nPrecision Constraint: >= {self.min_precision:.2f}")
        print(f"Models Ranked: {len(ranked_models)}")
        print("\n" + "-" * 80)
        print(
            f"{'Rank':<6} {'Model':<25} {'F1':<8} {'Prec':<8} "
            f"{'Recall':<8} {'Thresh':<8} {'Latency':<10}"
        )
        print("-" * 80)

        for model in ranked_models:
            print(
                f"{model.rank:<6} {model.model_name:<25} "
                f"{model.f1_score:<8.4f} {model.precision:<8.4f} "
                f"{model.recall:<8.4f} {model.optimal_threshold:<8.3f} "
                f"{model.latency_ms_per_pair:<10.2f}ms"
            )

        print("-" * 80)

        if ranked_models:
            print(f"\nTop Model: {ranked_models[0].model_name}")
            print(f"  F1 Score: {ranked_models[0].f1_score:.4f}")
            print(f"  Precision: {ranked_models[0].precision:.4f}")
            print(f"  Optimal Threshold: {ranked_models[0].optimal_threshold:.3f}")

        print("=" * 80 + "\n")


def rank_evaluation_results(
    results: List[EvaluationResult],
    min_precision: float = 0.80,
    top_n: int = 3,
) -> List[RankedModel]:
    """
    Convenience function to rank evaluation results.

    Args:
        results: Evaluation results
        min_precision: Precision constraint
        top_n: Number of models to return

    Returns:
        List of ranked models
    """
    ranker = ModelRanker(min_precision=min_precision)
    return ranker.rank_models(results, top_n=top_n)
