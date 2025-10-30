"""
Evaluator for Privacy Aware Semantic Cache.
Evaluates embedding engine performance on question similarity dataset.
"""

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_engine import EmbeddingEngine
from dataset import load_dataset


class SimilarityEvaluator:
    """
    Evaluates the performance of an embedding engine on semantic similarity tasks.
    """

    def __init__(self, embedding_engine: EmbeddingEngine, similarity_threshold: float = 0.85):
        """
        Initialize the evaluator.

        Args:
            embedding_engine: EmbeddingEngine instance to evaluate
            similarity_threshold: Cosine similarity threshold for considering questions similar
        """
        self.embedding_engine = embedding_engine
        self.similarity_threshold = similarity_threshold

    def compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (assumes embeddings are already normalized)
        """
        return np.dot(embedding1, embedding2)

    def evaluate_dataset(self, df: pd.DataFrame, max_samples: int = 1000) -> Dict:
        """
        Evaluate the embedding engine on a question similarity dataset.

        Args:
            df: DataFrame with columns ['question1', 'question2', 'is_duplicate']
            max_samples: Maximum number of samples to evaluate (default: 1000)

        Returns:
            Dictionary containing evaluation metrics
        """
        # Limit dataset size
        df_eval = df.head(max_samples).copy()
        n_samples = len(df_eval)

        print(f"Evaluating on {n_samples} question pairs...")
        print(f"Similarity threshold: {self.similarity_threshold}")

        # Extract questions
        questions1 = df_eval['question1'].tolist()
        questions2 = df_eval['question2'].tolist()
        ground_truth = df_eval['is_duplicate'].values

        # Generate embeddings in batches
        print("Generating embeddings for question1...")
        embeddings1 = self.embedding_engine.encode_batch(
            questions1,
            normalize=True,
            show_progress_bar=True
        )

        print("Generating embeddings for question2...")
        embeddings2 = self.embedding_engine.encode_batch(
            questions2,
            normalize=True,
            show_progress_bar=True
        )

        # Compute similarities
        print("Computing similarity scores...")
        similarities = np.array([
            self.compute_cosine_similarity(emb1, emb2)
            for emb1, emb2 in zip(embeddings1, embeddings2)
        ])

        # Apply threshold to get predictions
        predictions = (similarities >= self.similarity_threshold).astype(int)

        # Compute metrics
        metrics = self._compute_metrics(ground_truth, predictions, similarities)

        return metrics

    def _compute_metrics(self, ground_truth: np.ndarray, predictions: np.ndarray,
                        similarities: np.ndarray) -> Dict:
        """
        Compute evaluation metrics.

        Args:
            ground_truth: True labels
            predictions: Predicted labels
            similarities: Similarity scores

        Returns:
            Dictionary of metrics
        """
        # Basic classification metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions, zero_division=0)
        recall = recall_score(ground_truth, predictions, zero_division=0)
        f1 = f1_score(ground_truth, predictions, zero_division=0)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Similarity score statistics
        similarity_stats = {
            'mean': np.mean(similarities),
            'std': np.std(similarities),
            'min': np.min(similarities),
            'max': np.max(similarities),
            'median': np.median(similarities)
        }

        # Distribution of predictions
        true_positives_avg_sim = np.mean(similarities[(ground_truth == 1) & (predictions == 1)]) if tp > 0 else 0
        false_positives_avg_sim = np.mean(similarities[(ground_truth == 0) & (predictions == 1)]) if fp > 0 else 0
        true_negatives_avg_sim = np.mean(similarities[(ground_truth == 0) & (predictions == 0)]) if tn > 0 else 0
        false_negatives_avg_sim = np.mean(similarities[(ground_truth == 1) & (predictions == 0)]) if fn > 0 else 0

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'threshold': self.similarity_threshold,
            'similarity_stats': similarity_stats,
            'avg_similarity': {
                'true_positives': float(true_positives_avg_sim),
                'false_positives': float(false_positives_avg_sim),
                'true_negatives': float(true_negatives_avg_sim),
                'false_negatives': float(false_negatives_avg_sim)
            }
        }

        return metrics

    def print_evaluation_report(self, metrics: Dict):
        """
        Print a formatted evaluation report.

        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("\n" + "=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)

        print(f"\nThreshold: {metrics['threshold']:.3f}")

        print("\n--- Classification Metrics ---")
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1 Score:    {metrics['f1_score']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")

        print("\n--- Confusion Matrix ---")
        print(f"True Positives:  {metrics['true_positives']}")
        print(f"True Negatives:  {metrics['true_negatives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")

        print("\n--- Similarity Statistics ---")
        stats = metrics['similarity_stats']
        print(f"Mean:   {stats['mean']:.4f}")
        print(f"Std:    {stats['std']:.4f}")
        print(f"Min:    {stats['min']:.4f}")
        print(f"Max:    {stats['max']:.4f}")
        print(f"Median: {stats['median']:.4f}")

        print("\n--- Average Similarity by Category ---")
        avg_sim = metrics['avg_similarity']
        print(f"True Positives:  {avg_sim['true_positives']:.4f}")
        print(f"False Positives: {avg_sim['false_positives']:.4f}")
        print(f"True Negatives:  {avg_sim['true_negatives']:.4f}")
        print(f"False Negatives: {avg_sim['false_negatives']:.4f}")

        print("\n" + "=" * 70)


def run_evaluation(threshold: float = 0.85, max_samples: int = 1000):
    """
    Run evaluation on the questions dataset.

    Args:
        threshold: Similarity threshold for classification
        max_samples: Maximum number of samples to evaluate
    """
    print("Initializing Embedding Engine...")
    engine = EmbeddingEngine()

    print("\nLoading dataset...")
    df = load_dataset()
    print(f"Total dataset size: {len(df)} rows")

    print(f"\nInitializing Evaluator (threshold={threshold})...")
    evaluator = SimilarityEvaluator(engine, similarity_threshold=threshold)

    print("\nRunning evaluation...")
    metrics = evaluator.evaluate_dataset(df, max_samples=max_samples)

    evaluator.print_evaluation_report(metrics)

    return metrics


if __name__ == "__main__":
    # Run evaluation with default settings
    metrics = run_evaluation(threshold=0.95, max_samples=10000)