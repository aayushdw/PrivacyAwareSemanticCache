"""
Embedding model evaluator.

Handles model loading, embedding generation, and evaluation orchestration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..config.pipeline_config import get_config
from ..registry.model_info import ModelInfo
from ..utils.device_utils import DeviceContext, clear_gpu_memory, get_device
from .metrics_calculator import ClassificationMetrics, MetricsCalculator, TripletMetrics
from .threshold_tuner import ThresholdResult, ThresholdTuner


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model."""

    # Model identification
    model_key: str
    model_name: str
    model_id: str
    dimension: int
    category: str

    # Evaluation success
    success: bool
    error: Optional[str] = None

    # Threshold tuning results
    optimal_threshold: Optional[float] = None
    threshold_result: Optional[ThresholdResult] = None

    # Test set metrics
    test_metrics: Optional[ClassificationMetrics] = None

    # Triplet-specific metrics
    triplet_metrics: Optional[TripletMetrics] = None

    # Timing information
    model_load_time_seconds: float = 0.0
    embedding_time_seconds: float = 0.0
    total_samples: int = 0

    # Additional metadata
    meets_precision_constraint: bool = False
    precision_constraint: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "category": self.category,
            "success": self.success,
            "error": self.error,
            "optimal_threshold": self.optimal_threshold,
            "meets_precision_constraint": self.meets_precision_constraint,
            "precision_constraint": self.precision_constraint,
            "model_load_time_seconds": self.model_load_time_seconds,
            "embedding_time_seconds": self.embedding_time_seconds,
            "total_samples": self.total_samples,
        }

        if self.threshold_result:
            result["threshold_tuning"] = self.threshold_result.to_dict()

        if self.test_metrics:
            result["test_metrics"] = self.test_metrics.to_dict()

        if self.triplet_metrics:
            result["triplet_metrics"] = self.triplet_metrics.to_dict()

        return result

    @property
    def f1_score(self) -> float:
        """Get F1 score from test metrics."""
        if self.test_metrics:
            return self.test_metrics.f1_score
        return 0.0

    @property
    def precision(self) -> float:
        """Get precision from test metrics."""
        if self.test_metrics:
            return self.test_metrics.precision
        return 0.0

    @property
    def recall(self) -> float:
        """Get recall from test metrics."""
        if self.test_metrics:
            return self.test_metrics.recall
        return 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        """Create instance from dictionary (for loading cached results)."""
        threshold_result = None
        if "threshold_tuning" in data:
            threshold_result = ThresholdResult.from_dict(data["threshold_tuning"])

        test_metrics = None
        if "test_metrics" in data:
            test_metrics = ClassificationMetrics.from_dict(data["test_metrics"])

        triplet_metrics = None
        if "triplet_metrics" in data:
            triplet_metrics = TripletMetrics.from_dict(data["triplet_metrics"])

        return cls(
            model_key=data["model_key"],
            model_name=data["model_name"],
            model_id=data["model_id"],
            dimension=data["dimension"],
            category=data["category"],
            success=data["success"],
            error=data.get("error"),
            optimal_threshold=data.get("optimal_threshold"),
            threshold_result=threshold_result,
            test_metrics=test_metrics,
            triplet_metrics=triplet_metrics,
            model_load_time_seconds=data.get("model_load_time_seconds", 0.0),
            embedding_time_seconds=data.get("embedding_time_seconds", 0.0),
            total_samples=data.get("total_samples", 0),
            meets_precision_constraint=data.get("meets_precision_constraint", False),
            precision_constraint=data.get("precision_constraint"),
        )


class EmbeddingEvaluator:
    """Evaluates embedding models for semantic similarity."""

    def __init__(
        self,
        batch_size: int = 64,
        show_progress: bool = True,
    ):
        """
        Initialize evaluator.

        Args:
            batch_size: Batch size for embedding generation
            show_progress: Show progress bars
        """
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.model = None
        self.model_info = None
        self.device = None
        self.calculator = MetricsCalculator()

    def load_model(self, model_info: ModelInfo) -> float:
        """
        Load a sentence transformer model.

        Args:
            model_info: Model information

        Returns:
            Load time in seconds
        """
        start_time = time.time()

        self.device = get_device()
        self.model_info = model_info

        # Get cache directory from config
        config = get_config()
        cache_folder = config.model_cache_dir

        try:
            self.model = SentenceTransformer(
                model_info.model_id,
                device=str(self.device),
                cache_folder=cache_folder,
            )
        except Exception as e:
            # Try without device specification
            self.model = SentenceTransformer(
                model_info.model_id,
                cache_folder=cache_folder,
            )
            self.model = self.model.to(self.device)

        load_time = time.time() - start_time
        print(f"Loaded {model_info.name} in {load_time:.2f}s on {self.device}")

        return load_time

    def unload_model(self):
        """Unload model and clear memory."""
        self.model = None
        self.model_info = None
        clear_gpu_memory()

    def encode_texts(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            normalize: Normalize embeddings to unit length

        Returns:
            Array of embeddings (n_texts, dimension)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Apply instruction template if needed
        if self.model_info and self.model_info.requires_instruction:
            template = self.model_info.instruction_template or ""
            texts = [template + t for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        return embeddings

    def compute_similarities(
        self,
        questions1: List[str],
        questions2: List[str],
    ) -> Tuple[np.ndarray, float]:
        """
        Compute cosine similarities between question pairs.

        Returns:
            Tuple of (similarities, encoding_time)
        """
        start_time = time.time()

        # Encode both sets
        embeddings1 = self.encode_texts(questions1)
        embeddings2 = self.encode_texts(questions2)

        # Compute pairwise cosine similarities
        # For normalized vectors, dot product = cosine similarity
        similarities = np.sum(embeddings1 * embeddings2, axis=1)

        encoding_time = time.time() - start_time

        return similarities, encoding_time

    def compute_triplet_similarities(
        self,
        anchors: List[str],
        positives: List[str],
        negatives: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute similarities for triplet data.

        Args:
            anchors: Anchor questions
            positives: Positive (similar) questions
            negatives: Negative (dissimilar) questions

        Returns:
            Tuple of (positive_similarities, negative_similarities, encoding_time)
        """
        start_time = time.time()

        # Encode all texts
        anchor_emb = self.encode_texts(anchors)
        positive_emb = self.encode_texts(positives)
        negative_emb = self.encode_texts(negatives)

        # Compute similarities
        positive_sims = np.sum(anchor_emb * positive_emb, axis=1)
        negative_sims = np.sum(anchor_emb * negative_emb, axis=1)

        encoding_time = time.time() - start_time

        return positive_sims, negative_sims, encoding_time

    def evaluate_model(
        self,
        model_info: ModelInfo,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        min_precision: float = 0.80,
    ) -> EvaluationResult:
        """
        Full evaluation of a single model using triplet data format.

        1. Load model
        2. Tune threshold on training triplets
        3. Evaluate on test triplets

        Args:
            model_info: Model to evaluate
            train_df: Training data with columns (anchor, positive, negative)
            test_df: Test data with columns (anchor, positive, negative)
            min_precision: Minimum precision constraint

        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult(
            model_key=model_info.name.lower().replace(" ", "-"),
            model_name=model_info.name,
            model_id=model_info.model_id,
            dimension=model_info.dimension,
            category=model_info.category,
            success=False,
            precision_constraint=min_precision,
        )

        try:
            with DeviceContext(model_info.estimated_memory_mb):
                # Load model
                print(f"\n{'='*60}")
                print(f"Evaluating: {model_info.name}")
                print(f"{'='*60}")

                result.model_load_time_seconds = self.load_model(model_info)

                # Extract triplet data
                train_anchors = train_df["anchor"].tolist()
                train_positives = train_df["positive"].tolist()
                train_negatives = train_df["negative"].tolist()

                test_anchors = test_df["anchor"].tolist()
                test_positives = test_df["positive"].tolist()
                test_negatives = test_df["negative"].tolist()

                # Compute triplet similarities on training data
                print(f"\nComputing training similarities ({len(train_anchors)} triplets)...")
                train_pos_sims, train_neg_sims, train_time = self.compute_triplet_similarities(
                    train_anchors, train_positives, train_negatives
                )

                # Compute triplet metrics on training data
                train_triplet_metrics = self.calculator.compute_triplet_metrics(
                    train_pos_sims, train_neg_sims
                )
                print(f"Training triplet accuracy: {train_triplet_metrics.triplet_accuracy:.4f}")

                # Tune threshold using triplet data
                print(f"Tuning threshold (min_precision={min_precision:.2f})...")
                tuner = ThresholdTuner()
                threshold_result = tuner.find_optimal_threshold_from_triplets(
                    train_pos_sims,
                    train_neg_sims,
                    min_precision=min_precision,
                    metric="f1",
                )

                result.optimal_threshold = threshold_result.threshold
                result.threshold_result = threshold_result
                result.meets_precision_constraint = threshold_result.meets_constraint

                print(
                    f"Optimal threshold: {threshold_result.threshold:.3f} "
                    f"(F1={threshold_result.f1_score:.4f}, P={threshold_result.precision:.4f})"
                )

                # Evaluate on test data
                print(f"\nEvaluating on test set ({len(test_anchors)} triplets)...")
                test_pos_sims, test_neg_sims, test_time = self.compute_triplet_similarities(
                    test_anchors, test_positives, test_negatives
                )

                result.embedding_time_seconds = train_time + test_time
                result.total_samples = len(train_anchors) + len(test_anchors)

                # Compute triplet metrics on test data
                test_triplet_metrics = self.calculator.compute_triplet_metrics(
                    test_pos_sims, test_neg_sims
                )
                result.triplet_metrics = test_triplet_metrics

                # Convert to classification format for threshold-based metrics
                test_sims, test_labels = self.calculator.triplets_to_classification_data(
                    test_pos_sims, test_neg_sims
                )

                # Compute test metrics at optimal threshold
                test_metrics = self.calculator.compute_metrics_at_threshold(
                    test_sims,
                    test_labels,
                    threshold_result.threshold,
                )

                result.test_metrics = test_metrics
                result.success = True

                print(f"\nTest Results:")
                print(f"  Triplet Accuracy: {test_triplet_metrics.triplet_accuracy:.4f}")
                print(f"  Mean Margin:      {test_triplet_metrics.mean_margin:.4f}")
                print(f"  F1 Score:         {test_metrics.f1_score:.4f}")
                print(f"  Precision:        {test_metrics.precision:.4f}")
                print(f"  Recall:           {test_metrics.recall:.4f}")
                print(f"  Accuracy:         {test_metrics.accuracy:.4f}")

        except Exception as e:
            result.error = str(e)
            print(f"Error evaluating {model_info.name}: {e}")

        finally:
            self.unload_model()

        return result

    def quick_evaluate(
        self,
        model_info: ModelInfo,
        df: pd.DataFrame,
        threshold: float,
    ) -> ClassificationMetrics:
        """
        Quick evaluation at a fixed threshold using triplet data.

        Args:
            model_info: Model to evaluate
            df: Dataset with columns (anchor, positive, negative)
            threshold: Fixed threshold to use
        """
        try:
            with DeviceContext(model_info.estimated_memory_mb):
                self.load_model(model_info)

                anchors = df["anchor"].tolist()
                positives = df["positive"].tolist()
                negatives = df["negative"].tolist()

                pos_sims, neg_sims, _ = self.compute_triplet_similarities(
                    anchors, positives, negatives
                )

                # Convert to classification format
                sims, labels = self.calculator.triplets_to_classification_data(
                    pos_sims, neg_sims
                )

                metrics = self.calculator.compute_metrics_at_threshold(
                    sims, labels, threshold
                )

                return metrics

        finally:
            self.unload_model()
