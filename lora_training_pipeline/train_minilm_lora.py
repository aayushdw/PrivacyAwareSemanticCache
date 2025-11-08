"""
LoRA Fine-tuning Script for MiniLM on Quota Question Similarity Dataset.

This script:
1. Loads the MiniLM model
2. Applies LoRA for parameter-efficient fine-tuning
3. Trains on questions_train.csv
4. Saves the best model checkpoint
"""

import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Set environment variables to force offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Disable tokenizers parallelism to avoid fork warnings with DataLoader workers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# MPS (Apple Silicon) performance optimizations
if torch.backends.mps.is_available():
    # Enable MPS fallback to CPU for unsupported ops (improves stability)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


class QuestionPairDataset(Dataset):
    """Dataset for question pair similarity with on-the-fly tokenization."""

    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        """
        Initialize the dataset.

        Args:
            csv_path: Path to CSV file with question pairs
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.df = pd.read_csv(csv_path)

        # Check for missing values and drop them
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=['question1', 'question2', 'is_duplicate'])

        if len(self.df) < initial_len:
            print(f"Warning: Dropped {initial_len - len(self.df)} rows with missing values")

        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loaded {len(self.df)} question pairs from {csv_path}")
        print(f"Duplicate ratio: {self.df['is_duplicate'].mean():.2%}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get a single example, tokenizing on-the-fly."""
        row = self.df.iloc[idx]

        # Tokenize both questions
        q1_encoded = self.tokenizer(
            row['question1'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        q2_encoded = self.tokenizer(
            row['question2'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'q1_input_ids': q1_encoded['input_ids'].squeeze(0),
            'q1_attention_mask': q1_encoded['attention_mask'].squeeze(0),
            'q2_input_ids': q2_encoded['input_ids'].squeeze(0),
            'q2_attention_mask': q2_encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(row['is_duplicate'], dtype=torch.float32)
        }


class MiniLMLoRATrainer:
    """Trainer for fine-tuning MiniLM with LoRA."""

    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        learning_rate: float = 2e-4,
        batch_size: int = 32,
        num_epochs: int = 2,
        gradient_accumulation_steps: int = 1,
        output_dir: str = 'models',
        device: Optional[str] = None
    ):
        """
        Initialize the trainer.

        Args:
            model_name: Base model name
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout rate for LoRA layers
            learning_rate: Learning rate
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            gradient_accumulation_steps: Number of steps to accumulate gradients
            output_dir: Directory to save models
            device: Device to use (None for auto-detect)
        """
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Setup device with MPS support for Apple Silicon
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Setup output directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(current_dir, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Load tokenizer and model with offline support
        print(f"Loading model: {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.base_model = AutoModel.from_pretrained(model_name, local_files_only=True)
            print("Model loaded from cache.")
        except Exception as e:
            print(f"Loading from cache failed, trying online: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.base_model = AutoModel.from_pretrained(model_name)

        # Configure LoRA
        print("\nConfiguring LoRA...")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value", "dense"],  # Apply to all attention layers
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.to(self.device)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler will be initialized in train() after we know the dataset size
        self.scheduler = None

        # Loss function
        self.criterion = nn.CosineEmbeddingLoss()

        # Training history
        self.history = {
            'train_loss': [],
            'train_metrics': [],
            'best_f1': 0.0,
            'optimal_threshold': 0.5
        }

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, input_ids, attention_mask):
        """Encode text to embeddings."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs, attention_mask)
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def train_epoch(self, dataloader):
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        accumulated_loss = 0.0

        progress_bar = tqdm(dataloader, desc="Training")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            q1_input_ids = batch['q1_input_ids'].to(self.device)
            q1_attention_mask = batch['q1_attention_mask'].to(self.device)
            q2_input_ids = batch['q2_input_ids'].to(self.device)
            q2_attention_mask = batch['q2_attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # OPTIMIZATION: Batch both questions together for single forward pass
            # This reduces 2 forward passes to 1, giving ~2x speedup
            batch_size = q1_input_ids.size(0)
            combined_input_ids = torch.cat([q1_input_ids, q2_input_ids], dim=0)
            combined_attention_mask = torch.cat([q1_attention_mask, q2_attention_mask], dim=0)

            # Single forward pass for both questions
            combined_embeddings = self.encode(combined_input_ids, combined_attention_mask)

            # Split back into q1 and q2 embeddings
            q1_embeddings = combined_embeddings[:batch_size]
            q2_embeddings = combined_embeddings[batch_size:]

            # Convert labels: 1.0 -> 1, 0.0 -> -1 for CosineEmbeddingLoss
            cos_labels = torch.where(labels > 0.5,
                                    torch.ones_like(labels),
                                    -torch.ones_like(labels))

            # Compute loss
            loss = self.criterion(q1_embeddings, q2_embeddings, cos_labels)

            # Scale loss by accumulation steps
            loss = loss / self.gradient_accumulation_steps
            accumulated_loss += loss.item()

            # Backward pass
            loss.backward()

            # Only update weights after accumulating gradients
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update learning rate scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                total_loss += accumulated_loss
                accumulated_loss = 0.0

            # Show current loss and learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps, 'lr': f'{current_lr:.2e}'})

        avg_loss = total_loss / (len(dataloader) // self.gradient_accumulation_steps)
        return avg_loss

    def quick_evaluate(self, dataloader, threshold: float = 0.5) -> Dict[str, float]:
        """
        Fast evaluation using a fixed threshold (for monitoring during training).

        Args:
            dataloader: DataLoader for evaluation
            threshold: Fixed similarity threshold to use

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_similarities = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Quick eval", leave=False):
                q1_input_ids = batch['q1_input_ids'].to(self.device)
                q1_attention_mask = batch['q1_attention_mask'].to(self.device)
                q2_input_ids = batch['q2_input_ids'].to(self.device)
                q2_attention_mask = batch['q2_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # OPTIMIZATION: Batch both questions for single forward pass
                batch_size = q1_input_ids.size(0)
                combined_input_ids = torch.cat([q1_input_ids, q2_input_ids], dim=0)
                combined_attention_mask = torch.cat([q1_attention_mask, q2_attention_mask], dim=0)

                combined_embeddings = self.encode(combined_input_ids, combined_attention_mask)
                q1_embeddings = combined_embeddings[:batch_size]
                q2_embeddings = combined_embeddings[batch_size:]

                # Cosine similarity (embeddings are already normalized)
                similarities = (q1_embeddings * q2_embeddings).sum(dim=1)

                all_similarities.extend(similarities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        similarities = np.array(all_similarities)
        labels = np.array(all_labels)

        # Compute metrics with fixed threshold (much faster)
        predictions = (similarities >= threshold).astype(int)
        metrics = {
            'threshold': threshold,
            'f1': f1_score(labels, predictions),
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'avg_similarity': similarities.mean(),
            'similarity_std': similarities.std()
        }

        return metrics

    def evaluate(self, dataloader) -> Tuple[float, Dict[str, float]]:
        """
        Full evaluation with optimal threshold search (use only at the end).

        Returns:
            Tuple of (optimal_threshold, metrics_dict)
        """
        self.model.eval()
        all_similarities = []
        all_labels = []

        print("Computing similarities for evaluation...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                q1_input_ids = batch['q1_input_ids'].to(self.device)
                q1_attention_mask = batch['q1_attention_mask'].to(self.device)
                q2_input_ids = batch['q2_input_ids'].to(self.device)
                q2_attention_mask = batch['q2_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # OPTIMIZATION: Batch both questions for single forward pass
                batch_size = q1_input_ids.size(0)
                combined_input_ids = torch.cat([q1_input_ids, q2_input_ids], dim=0)
                combined_attention_mask = torch.cat([q1_attention_mask, q2_attention_mask], dim=0)

                combined_embeddings = self.encode(combined_input_ids, combined_attention_mask)
                q1_embeddings = combined_embeddings[:batch_size]
                q2_embeddings = combined_embeddings[batch_size:]

                # Cosine similarity (embeddings are already normalized)
                similarities = (q1_embeddings * q2_embeddings).sum(dim=1)

                all_similarities.extend(similarities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        similarities = np.array(all_similarities)
        labels = np.array(all_labels)

        # Find optimal threshold
        print("Finding optimal threshold...")
        thresholds = np.linspace(similarities.min(), similarities.max(), 100)
        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)
            f1 = f1_score(labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Compute final metrics with optimal threshold
        predictions = (similarities >= best_threshold).astype(int)
        metrics = {
            'threshold': best_threshold,
            'f1': f1_score(labels, predictions),
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0)
        }

        return best_threshold, metrics

    def train(self, train_csv: str):
        """
        Train the model.

        Args:
            train_csv: Path to training CSV file
        """
        print("="*60)
        print("Starting LoRA Fine-tuning")
        print("="*60)

        # Load dataset
        train_dataset = QuestionPairDataset(train_csv, self.tokenizer)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,  # Optimized for MacBook Air
            pin_memory=False,  # MPS doesn't benefit from pinned memory
            persistent_workers=True  # Keep workers alive between epochs
        )

        # Initialize learning rate scheduler with warmup
        num_training_steps = len(train_dataloader) * self.num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        print(f"\nTraining Configuration:")
        print(f"  Total training steps: {num_training_steps}")
        print(f"  Warmup steps: {num_warmup_steps} (10%)")
        print(f"  Initial LR: {self.learning_rate:.2e}")
        print(f"  LR Schedule: Cosine annealing with warmup")
        print(f"  Gradient Clipping: max_norm=1.0")

        # Training loop with fast monitoring
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 60)

            # Train
            train_loss = self.train_epoch(train_dataloader)
            print(f"Training Loss: {train_loss:.4f}")

            # Quick evaluation with fixed threshold (much faster, for monitoring)
            metrics = self.quick_evaluate(train_dataloader, threshold=0.5)

            print(f"\nQuick Evaluation (fixed threshold=0.5):")
            print(f"  F1 Score:      {metrics['f1']:.4f}")
            print(f"  Accuracy:      {metrics['accuracy']:.4f}")
            print(f"  Precision:     {metrics['precision']:.4f}")
            print(f"  Recall:        {metrics['recall']:.4f}")
            print(f"  Avg Similarity: {metrics['avg_similarity']:.4f} ± {metrics['similarity_std']:.4f}")

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_metrics'].append(metrics)

        # Final evaluation with optimal threshold search
        print("\n" + "="*60)
        print("Training Complete! Running final evaluation with threshold optimization...")
        print("="*60)

        optimal_threshold, final_metrics = self.evaluate(train_dataloader)

        print(f"\nFinal Evaluation Results:")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  F1 Score:  {final_metrics['f1']:.4f}")
        print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
        print(f"  Precision: {final_metrics['precision']:.4f}")
        print(f"  Recall:    {final_metrics['recall']:.4f}")

        self.history['best_f1'] = final_metrics['f1']
        self.history['optimal_threshold'] = optimal_threshold

        # Save final model
        final_model_path = os.path.join(self.output_dir, 'final_model')
        print(f"\nSaving final model to {final_model_path}")
        self.save_model(final_model_path, optimal_threshold, final_metrics)

        # Print summary
        print("\n" + "="*60)
        print("Model Training and Evaluation Complete!")
        print("="*60)
        print(f"Final F1 Score: {self.history['best_f1']:.4f}")
        print(f"Optimal Threshold: {self.history['optimal_threshold']:.4f}")
        print(f"Model saved to: {final_model_path}")

    def save_model(self, save_path: str, threshold: float, metrics: Dict[str, Any]):
        """Save the model and metadata."""
        os.makedirs(save_path, exist_ok=True)

        # Save LoRA model
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_python_type(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_python_type(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_type(item) for item in obj]
            return obj

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'optimal_threshold': float(threshold),
            'metrics': convert_to_python_type(metrics),
            'training_date': datetime.now().isoformat()
        }

        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved successfully to {save_path}")


def main():
    """Main training pipeline."""

    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    train_csv = os.path.join(project_root, 'data', 'small', 'questions_train.csv')

    # Check if training data exists
    if not os.path.exists(train_csv):
        print(f"Error: Training data not found at {train_csv}")
        return

    # Initialize trainer
    trainer = MiniLMLoRATrainer(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        lora_r=16,           # Increased from 8 to 16 for more trainable params
        lora_alpha=32,       # Increased proportionally (2x rank)
        lora_dropout=0.1,
        learning_rate=2e-4,
        batch_size=64,       # Increased from 32 to 64 for better GPU utilization
        num_epochs=5,
        gradient_accumulation_steps=1,
        output_dir='models'
    )

    # Train
    trainer.train(train_csv)


if __name__ == "__main__":
    main()
