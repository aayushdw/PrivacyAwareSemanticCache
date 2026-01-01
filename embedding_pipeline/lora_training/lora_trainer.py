"""LoRA fine-tuning trainer for embedding models."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..config.pipeline_config import get_config
from ..registry.model_info import ModelInfo
from ..utils.device_utils import DeviceContext, clear_gpu_memory, get_device


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA training."""

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )
    learning_rate: float = 2e-4
    batch_size: int = 64
    max_epochs: int = 10
    patience: int = 3
    min_delta: float = 0.0001
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 128

    @classmethod
    def from_pipeline_config(cls) -> "LoRATrainingConfig":
        """Create from pipeline config."""
        config = get_config()
        return cls(
            lora_r=config.lora.lora_r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=config.lora.target_modules,
            learning_rate=config.lora.learning_rate,
            batch_size=config.lora.batch_size,
            max_epochs=config.lora.max_epochs,
            patience=config.lora.patience,
            min_delta=config.lora.min_delta,
            gradient_accumulation_steps=config.lora.gradient_accumulation_steps,
            max_seq_length=config.lora.max_seq_length,
        )


@dataclass
class TrainingResult:
    """Result of LoRA training."""

    success: bool
    model_key: str
    adapter_path: Optional[str] = None
    error: Optional[str] = None

    # Training metrics
    epochs_trained: int = 0
    early_stopped: bool = False
    best_val_loss: float = float("inf")

    # Final evaluation metrics
    final_f1: float = 0.0
    final_precision: float = 0.0
    final_recall: float = 0.0
    final_accuracy: float = 0.0
    optimal_threshold: float = 0.5

    # Training history
    history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "model_key": self.model_key,
            "adapter_path": self.adapter_path,
            "error": self.error,
            "epochs_trained": self.epochs_trained,
            "early_stopped": self.early_stopped,
            "best_val_loss": self.best_val_loss,
            "final_f1": self.final_f1,
            "final_precision": self.final_precision,
            "final_recall": self.final_recall,
            "final_accuracy": self.final_accuracy,
            "optimal_threshold": self.optimal_threshold,
        }


class TripletDataset(Dataset):
    """Dataset for triplet-based training (anchor, positive, negative)."""

    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        """Initialize dataset."""
        self.df = pd.read_csv(csv_path)

        # Clean data
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=["anchor", "positive", "negative"])
        if len(self.df) < initial_len:
            print(f"Dropped {initial_len - len(self.df)} rows with missing values")

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        anchor_encoded = self.tokenizer(
            str(row["anchor"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        positive_encoded = self.tokenizer(
            str(row["positive"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        negative_encoded = self.tokenizer(
            str(row["negative"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "anchor_input_ids": anchor_encoded["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor_encoded["attention_mask"].squeeze(0),
            "positive_input_ids": positive_encoded["input_ids"].squeeze(0),
            "positive_attention_mask": positive_encoded["attention_mask"].squeeze(0),
            "negative_input_ids": negative_encoded["input_ids"].squeeze(0),
            "negative_attention_mask": negative_encoded["attention_mask"].squeeze(0),
        }


class LoRATrainer:
    """
    LoRA fine-tuning trainer for embedding models.

    Supports early stopping based on validation loss.
    """

    def __init__(
        self,
        model_info: ModelInfo,
        config: Optional[LoRATrainingConfig] = None,
        output_dir: str = "embedding_pipeline/outputs/models",
    ):
        """
        Initialize trainer.

        Args:
            model_info: Model to fine-tune
            config: Training configuration
            output_dir: Directory to save trained models
        """
        self.model_info = model_info
        self.config = config or LoRATrainingConfig.from_pipeline_config()
        self.output_dir = output_dir

        # Set up environment
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if torch.backends.mps.is_available():
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        self.device = get_device()
        self.model = None
        self.tokenizer = None

        # Training state
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.best_model_state = None

    def _load_model(self):
        """Load base model and apply LoRA."""
        print(f"Loading model: {self.model_info.model_id}")

        # Get cache directory from config
        config = get_config()
        cache_dir = config.model_cache_dir

        # Load tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_info.model_id,
            cache_dir=cache_dir,
        )
        base_model = AutoModel.from_pretrained(
            self.model_info.model_id,
            cache_dir=cache_dir,
        )

        # Determine target modules based on model architecture
        target_modules = self.config.target_modules
        model_name_lower = self.model_info.model_id.lower()
        
        if "instructor" in model_name_lower or "t5" in model_name_lower:
            print(f"Detected T5/Instructor model ({self.model_info.model_id}). Using T5-specific target modules.")
            # T5 uses q, k, v, o in attention layers
            target_modules = ["q", "v"]
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        # Apply LoRA
        self.model = get_peft_model(base_model, lora_config)
        self.model.to(self.device)
        self.model.print_trainable_parameters()

        # Setup optimizer and loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        # Triplet margin loss: encourages sim(anchor, positive) > sim(anchor, negative) + margin
        self.criterion = nn.TripletMarginLoss(margin=0.2, p=2)

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _encode(self, input_ids, attention_mask):
        """Encode text to normalized embeddings."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self._mean_pooling(outputs, attention_mask)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def _train_epoch(self, dataloader) -> float:
        """Train for one epoch using triplet loss."""
        self.model.train()
        total_loss = 0.0
        accumulated_loss = 0.0

        progress_bar = tqdm(dataloader, desc="Training")
        for step, batch in enumerate(progress_bar):
            # Move to device
            anchor_ids = batch["anchor_input_ids"].to(self.device)
            anchor_mask = batch["anchor_attention_mask"].to(self.device)
            positive_ids = batch["positive_input_ids"].to(self.device)
            positive_mask = batch["positive_attention_mask"].to(self.device)
            negative_ids = batch["negative_input_ids"].to(self.device)
            negative_mask = batch["negative_attention_mask"].to(self.device)

            # Batch all three for efficiency
            batch_size = anchor_ids.size(0)
            combined_ids = torch.cat([anchor_ids, positive_ids, negative_ids], dim=0)
            combined_mask = torch.cat([anchor_mask, positive_mask, negative_mask], dim=0)

            # Forward pass
            combined_emb = self._encode(combined_ids, combined_mask)
            anchor_emb = combined_emb[:batch_size]
            positive_emb = combined_emb[batch_size : 2 * batch_size]
            negative_emb = combined_emb[2 * batch_size :]

            # Compute triplet loss
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            loss = loss / self.config.gradient_accumulation_steps
            accumulated_loss += loss.item()

            # Backward
            loss.backward()

            # Update weights
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += accumulated_loss
                accumulated_loss = 0.0

            progress_bar.set_postfix(
                {"loss": loss.item() * self.config.gradient_accumulation_steps}
            )

        return total_loss / (len(dataloader) // self.config.gradient_accumulation_steps)

    def _compute_val_loss(self, dataloader) -> float:
        """Compute validation loss using triplet data."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", leave=False):
                anchor_ids = batch["anchor_input_ids"].to(self.device)
                anchor_mask = batch["anchor_attention_mask"].to(self.device)
                positive_ids = batch["positive_input_ids"].to(self.device)
                positive_mask = batch["positive_attention_mask"].to(self.device)
                negative_ids = batch["negative_input_ids"].to(self.device)
                negative_mask = batch["negative_attention_mask"].to(self.device)

                batch_size = anchor_ids.size(0)
                combined_ids = torch.cat([anchor_ids, positive_ids, negative_ids], dim=0)
                combined_mask = torch.cat([anchor_mask, positive_mask, negative_mask], dim=0)

                combined_emb = self._encode(combined_ids, combined_mask)
                anchor_emb = combined_emb[:batch_size]
                positive_emb = combined_emb[batch_size : 2 * batch_size]
                negative_emb = combined_emb[2 * batch_size :]

                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else float("inf")

    def _evaluate(self, dataloader) -> Tuple[float, Dict]:
        """Full evaluation with threshold optimization using triplet data."""
        self.model.eval()
        all_pos_sims = []
        all_neg_sims = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                anchor_ids = batch["anchor_input_ids"].to(self.device)
                anchor_mask = batch["anchor_attention_mask"].to(self.device)
                positive_ids = batch["positive_input_ids"].to(self.device)
                positive_mask = batch["positive_attention_mask"].to(self.device)
                negative_ids = batch["negative_input_ids"].to(self.device)
                negative_mask = batch["negative_attention_mask"].to(self.device)

                batch_size = anchor_ids.size(0)
                combined_ids = torch.cat([anchor_ids, positive_ids, negative_ids], dim=0)
                combined_mask = torch.cat([anchor_mask, positive_mask, negative_mask], dim=0)

                combined_emb = self._encode(combined_ids, combined_mask)
                anchor_emb = combined_emb[:batch_size]
                positive_emb = combined_emb[batch_size : 2 * batch_size]
                negative_emb = combined_emb[2 * batch_size :]

                # Compute similarities
                pos_sims = (anchor_emb * positive_emb).sum(dim=1)
                neg_sims = (anchor_emb * negative_emb).sum(dim=1)

                all_pos_sims.extend(pos_sims.cpu().numpy())
                all_neg_sims.extend(neg_sims.cpu().numpy())

        positive_sims = np.array(all_pos_sims)
        negative_sims = np.array(all_neg_sims)

        # Compute triplet accuracy (% where positive_sim > negative_sim)
        triplet_accuracy = float(np.mean(positive_sims > negative_sims))

        # Convert to classification format for threshold optimization
        similarities = np.concatenate([positive_sims, negative_sims])
        labels = np.concatenate([np.ones(len(positive_sims)), np.zeros(len(negative_sims))])

        # Find optimal threshold
        thresholds = np.linspace(similarities.min(), similarities.max(), 100)
        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)
            f1 = f1_score(labels, predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Compute final metrics
        predictions = (similarities >= best_threshold).astype(int)
        metrics = {
            "threshold": best_threshold,
            "triplet_accuracy": triplet_accuracy,
            "mean_margin": float(np.mean(positive_sims - negative_sims)),
            "f1": f1_score(labels, predictions, zero_division=0),
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
        }

        return best_threshold, metrics

    def train(self, train_csv: str, val_csv: str) -> TrainingResult:
        """
        Train LoRA adapter.

        Args:
            train_csv: Path to training data
            val_csv: Path to validation data

        Returns:
            TrainingResult with metrics and adapter path
        """
        model_key = self.model_info.name.lower().replace(" ", "-")

        result = TrainingResult(
            success=False,
            model_key=model_key,
        )

        try:
            with DeviceContext(self.model_info.estimated_memory_mb):
                print(f"\n{'='*60}")
                print(f"LoRA Training: {self.model_info.name}")
                print(f"{'='*60}")

                # Load model
                self._load_model()

                # Create dataloaders with triplet data
                train_dataset = TripletDataset(
                    train_csv, self.tokenizer, self.config.max_seq_length
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=False,
                )

                val_dataset = TripletDataset(
                    val_csv, self.tokenizer, self.config.max_seq_length
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=False,
                )

                print(f"\nTraining samples: {len(train_dataset)}")
                print(f"Validation samples: {len(val_dataset)}")
                print(f"Batch size: {self.config.batch_size}")
                print(f"Max epochs: {self.config.max_epochs}")
                print(f"Early stopping patience: {self.config.patience}")

                # Training loop
                for epoch in range(self.config.max_epochs):
                    print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
                    print("-" * 40)

                    # Train
                    train_loss = self._train_epoch(train_loader)
                    print(f"Train Loss: {train_loss:.4f}")

                    # Validate
                    val_loss = self._compute_val_loss(val_loader)
                    print(f"Val Loss: {val_loss:.4f}")

                    # Record history
                    result.history.append(
                        {
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                        }
                    )
                    result.epochs_trained = epoch + 1

                    # Early stopping check
                    if val_loss < self.best_val_loss - self.config.min_delta:
                        self.best_val_loss = val_loss
                        self.epochs_without_improvement = 0
                        self.best_model_state = {
                            k: v.cpu().clone() for k, v in self.model.state_dict().items()
                        }
                        print(f"New best model (val_loss: {val_loss:.4f})")
                    else:
                        self.epochs_without_improvement += 1
                        print(
                            f"No improvement for {self.epochs_without_improvement} epoch(s)"
                        )

                    if self.epochs_without_improvement >= self.config.patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        result.early_stopped = True
                        break

                # Restore best model
                if self.best_model_state is not None:
                    self.model.load_state_dict(self.best_model_state)
                    print("\nRestored best model weights")

                # Final evaluation
                print("\nFinal Evaluation:")
                threshold, metrics = self._evaluate(val_loader)

                result.optimal_threshold = float(threshold)
                result.final_f1 = float(metrics["f1"])
                result.final_precision = float(metrics["precision"])
                result.final_recall = float(metrics["recall"])
                result.final_accuracy = float(metrics["accuracy"])
                result.best_val_loss = float(self.best_val_loss)

                print(f"  Optimal Threshold: {threshold:.3f}")
                print(f"  Triplet Accuracy: {metrics['triplet_accuracy']:.4f}")
                print(f"  Mean Margin: {metrics['mean_margin']:.4f}")
                print(f"  F1 Score: {metrics['f1']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")

                # Save adapter
                adapter_dir = os.path.join(self.output_dir, f"{model_key}_lora")
                os.makedirs(adapter_dir, exist_ok=True)
                self.model.save_pretrained(adapter_dir)
                self.tokenizer.save_pretrained(adapter_dir)

                result.adapter_path = adapter_dir
                result.success = True

                print(f"\nAdapter saved to: {adapter_dir}")

        except Exception as e:
            result.error = str(e)
            print(f"Training failed: {e}")

        finally:
            self.model = None
            self.tokenizer = None
            clear_gpu_memory()

        return result
