"""
Evaluation Script for LoRA Fine-tuned MiniLM Model.

This script:
1. Loads the fine-tuned LoRA model
2. Finds optimal threshold on training data
3. Displays confusion matrix and metrics
4. Evaluates on validation data with the optimal threshold
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)


class LoRAModelEvaluator:
    """Evaluator for LoRA fine-tuned MiniLM model."""

    def __init__(self, model_path: str, base_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize the evaluator.

        Args:
            model_path: Path to the saved LoRA model
            base_model_name: Base model name
        """
        self.model_path = model_path
        self.base_model_name = base_model_name

        # Setup device with MPS support for Apple Silicon
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print(f"Loading LoRA model from: {model_path}")
        print(f"Using device: {self.device}")

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)

        # Load base model
        try:
            base_model = AutoModel.from_pretrained(base_model_name, local_files_only=True)
            print("Base model loaded from cache.")
        except Exception as e:
            print(f"Loading from cache failed: {e}")
            base_model = AutoModel.from_pretrained(base_model_name)

        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")

        # Load metadata if available
        metadata_path = os.path.join(model_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                print(f"\nModel metadata:")
                print(f"  Training date: {self.metadata.get('training_date', 'N/A')}")
                print(f"  LoRA rank: {self.metadata.get('lora_r', 'N/A')}")
                print(f"  Saved threshold: {self.metadata.get('optimal_threshold', 'N/A'):.4f}")
        else:
            self.metadata = {}

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, texts, batch_size=32, show_progress=True):
        """
        Encode texts to embeddings with batched processing.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings
        """
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", total=(len(texts) + batch_size - 1) // batch_size)

        for i in iterator:
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = self.mean_pooling(outputs, encoded['attention_mask'])
                embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def load_dataset(self, csv_path: str) -> pd.DataFrame:
        """Load the question similarity dataset."""
        print(f"\nLoading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)

        # Check for missing values and drop them
        initial_len = len(df)
        df = df.dropna(subset=['question1', 'question2', 'is_duplicate'])

        if len(df) < initial_len:
            print(f"Warning: Dropped {initial_len - len(df)} rows with missing values")

        print(f"Dataset loaded: {len(df)} samples")
        print(f"Duplicate ratio: {df['is_duplicate'].mean():.2%}")
        return df

    def compute_similarities(self, df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        """
        Compute cosine similarities for all question pairs using batched encoding.

        Args:
            df: DataFrame with question pairs
            batch_size: Batch size for encoding

        Returns:
            Array of similarity scores
        """
        print("\nComputing similarities with batched encoding...")

        print("Encoding first questions...")
        q1_embeddings = self.encode(df['question1'].tolist(), batch_size=batch_size, show_progress=True)

        print("Encoding second questions...")
        q2_embeddings = self.encode(df['question2'].tolist(), batch_size=batch_size, show_progress=True)

        # Cosine similarity (embeddings are normalized)
        similarities = (q1_embeddings * q2_embeddings).sum(axis=1)

        return similarities

    def find_optimal_threshold(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, Dict[str, float]]:
        """Find the optimal threshold that maximizes the specified metric."""
        print(f"\nFinding optimal threshold (optimizing {metric})...")

        thresholds = np.linspace(similarities.min(), similarities.max(), 100)
        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(labels, predictions)
            elif metric == 'accuracy':
                score = accuracy_score(labels, predictions)
            elif metric == 'precision':
                score = precision_score(labels, predictions, zero_division=0)
            elif metric == 'recall':
                score = recall_score(labels, predictions, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        # Compute metrics at optimal threshold
        predictions = (similarities >= best_threshold).astype(int)

        metrics = {
            'threshold': best_threshold,
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0)
        }

        print(f"Optimal threshold: {best_threshold:.4f}")
        print(f"Best {metric}: {best_score:.4f}")

        return best_threshold, metrics

    def plot_confusion_matrix(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: str = None
    ):
        """Plot confusion matrix with percentages."""
        cm = confusion_matrix(labels, predictions)

        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Create annotations with both count and percentage
        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_percent,
            annot=annotations,
            fmt='',
            cmap='Blues',
            xticklabels=['Not Duplicate', 'Duplicate'],
            yticklabels=['Not Duplicate', 'Duplicate'],
            cbar_kws={'label': 'Percentage (%)'}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")

        plt.show()

    def plot_threshold_analysis(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        optimal_threshold: float,
        save_path: str = None
    ):
        """Plot threshold analysis."""
        thresholds = np.linspace(similarities.min(), similarities.max(), 100)
        precisions = []
        recalls = []
        f1_scores = []

        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)
            precisions.append(precision_score(labels, predictions, zero_division=0))
            recalls.append(recall_score(labels, predictions, zero_division=0))
            f1_scores.append(f1_score(labels, predictions, zero_division=0))

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
        plt.axvline(x=optimal_threshold, color='red', linestyle='--',
                   label=f'Optimal Threshold: {optimal_threshold:.4f}', linewidth=2)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Threshold Analysis (LoRA Fine-tuned Model)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold analysis plot saved to: {save_path}")

        plt.show()

    def plot_similarity_distribution(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        optimal_threshold: float,
        save_path: str = None
    ):
        """Plot distribution of similarity scores."""
        plt.figure(figsize=(10, 6))

        duplicates = similarities[labels == 1]
        non_duplicates = similarities[labels == 0]

        plt.hist(non_duplicates, bins=50, alpha=0.5, label='Not Duplicate', color='blue')
        plt.hist(duplicates, bins=50, alpha=0.5, label='Duplicate', color='orange')
        plt.axvline(x=optimal_threshold, color='red', linestyle='--',
                   label=f'Optimal Threshold: {optimal_threshold:.4f}', linewidth=2)

        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Similarity Distribution (LoRA Fine-tuned Model)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Similarity distribution plot saved to: {save_path}")

        plt.show()

    def evaluate(
        self,
        df: pd.DataFrame,
        threshold: float,
        dataset_name: str = "Dataset"
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluate the model on a dataset with a given threshold."""
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'='*60}")

        # Compute similarities
        similarities = self.compute_similarities(df)
        labels = df['is_duplicate'].values

        # Make predictions
        predictions = (similarities >= threshold).astype(int)

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0)
        }

        # Print results
        print(f"\nResults using threshold = {threshold:.4f}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            labels,
            predictions,
            target_names=['Not Duplicate', 'Duplicate'],
            digits=4
        ))

        return metrics, similarities, predictions


def main():
    """Main evaluation pipeline."""

    print("="*60)
    print("LoRA Fine-tuned MiniLM Model Evaluation")
    print("Quota Question Similarity Dataset")
    print("="*60)

    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    train_csv = os.path.join(project_root, 'data', 'medium', 'questions_train.csv')
    val_csv = os.path.join(project_root, 'data', 'medium', 'questions_test.csv')

    # Path to the trained model
    model_path = os.path.join(current_dir, 'models', 'best_model')

    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first by running: python train_minilm_lora.py")
        return

    # Create output directory for plots
    output_dir = os.path.join(current_dir, 'lora_evaluation_results')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = LoRAModelEvaluator(model_path)

    # Load training dataset
    train_df = evaluator.load_dataset(train_csv)

    # Compute similarities on training data
    train_similarities = evaluator.compute_similarities(train_df)
    train_labels = train_df['is_duplicate'].values

    # Find optimal threshold
    optimal_threshold, train_metrics = evaluator.find_optimal_threshold(
        train_similarities,
        train_labels,
        metric='f1'
    )

    # Print training results
    print(f"\n{'='*60}")
    print("Training Set Performance (Optimal Threshold)")
    print(f"{'='*60}")
    print(f"Threshold:  {train_metrics['threshold']:.4f}")
    print(f"Accuracy:   {train_metrics['accuracy']:.4f}")
    print(f"Precision:  {train_metrics['precision']:.4f}")
    print(f"Recall:     {train_metrics['recall']:.4f}")
    print(f"F1 Score:   {train_metrics['f1']:.4f}")

    # Generate predictions for training data
    train_predictions = (train_similarities >= optimal_threshold).astype(int)

    # Plot confusion matrix for training data
    print("\n" + "="*60)
    print("Generating visualizations for training data...")
    print("="*60)

    evaluator.plot_confusion_matrix(
        train_labels,
        train_predictions,
        title=f"Training Set Confusion Matrix (LoRA, Threshold={optimal_threshold:.4f})",
        save_path=os.path.join(output_dir, 'confusion_matrix_train_lora.png')
    )

    # Plot threshold analysis
    evaluator.plot_threshold_analysis(
        train_similarities,
        train_labels,
        optimal_threshold,
        save_path=os.path.join(output_dir, 'threshold_analysis_lora.png')
    )

    # Plot similarity distribution
    evaluator.plot_similarity_distribution(
        train_similarities,
        train_labels,
        optimal_threshold,
        save_path=os.path.join(output_dir, 'similarity_distribution_train_lora.png')
    )

    # Evaluate on validation dataset
    print("\n" + "="*60)
    print("Evaluating on Validation Set")
    print("="*60)

    val_df = evaluator.load_dataset(val_csv)
    val_metrics, val_similarities, val_predictions = evaluator.evaluate(
        val_df,
        optimal_threshold,
        dataset_name="Validation Set"
    )

    # Plot confusion matrix for validation data
    evaluator.plot_confusion_matrix(
        val_df['is_duplicate'].values,
        val_predictions,
        title=f"Validation Set Confusion Matrix (LoRA, Threshold={optimal_threshold:.4f})",
        save_path=os.path.join(output_dir, 'confusion_matrix_val_lora.png')
    )

    # Plot similarity distribution for validation data
    evaluator.plot_similarity_distribution(
        val_similarities,
        val_df['is_duplicate'].values,
        optimal_threshold,
        save_path=os.path.join(output_dir, 'similarity_distribution_val_lora.png')
    )

    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY - LoRA FINE-TUNED MODEL")
    print("="*60)
    print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
    print("\nTraining Set:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")
    print(f"  F1 Score:  {train_metrics['f1']:.4f}")
    print("\nValidation Set:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1 Score:  {val_metrics['f1']:.4f}")

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
