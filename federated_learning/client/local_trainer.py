"""Local training logic for FL clients."""

import os
from typing import Dict, List

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from embedding_pipeline.lora_training.lora_trainer import TripletDataset
from embedding_pipeline.utils.device_utils import clear_gpu_memory, get_device


class LocalLoRATrainer:
    """Handles local LoRA training for a Flower client."""

    def __init__(
        self,
        base_model_name: str,
        lora_config: Dict,
        train_data_path: str,
        learning_rate: float = 2e-4,
        batch_size: int = 32,
        max_seq_length: int = 128,
    ):
        """
        Initialize local trainer.

        Args:
            base_model_name: HuggingFace model ID for base model
            lora_config: Dict with lora_r, lora_alpha, lora_dropout, target_modules
            train_data_path: Path to triplet training CSV
            learning_rate: Learning rate for AdamW optimizer
            batch_size: Training batch size
            max_seq_length: Maximum sequence length for tokenization
        """
        self.base_model_name = base_model_name
        self.lora_config = lora_config
        self.train_data_path = train_data_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        self.device = get_device()
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.criterion = nn.TripletMarginLoss(margin=0.2, p=2)

    def initialize_model(self) -> None:
        """Initialize base model with LoRA adapter."""
        if self.model is not None:
            return

        # Setup environment
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if torch.backends.mps.is_available():
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        # Load base model
        base_model = AutoModel.from_pretrained(self.base_model_name)

        # Apply LoRA
        lora_cfg = LoraConfig(
            r=self.lora_config["lora_r"],
            lora_alpha=self.lora_config["lora_alpha"],
            target_modules=self.lora_config["target_modules"],
            lora_dropout=self.lora_config["lora_dropout"],
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        self.model = get_peft_model(base_model, lora_cfg)
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

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
        return nn.functional.normalize(embeddings, p=2, dim=1)

    def train_local_epochs(self, num_epochs: int = 1) -> Dict:
        """
        Train for specified number of local epochs.

        Args:
            num_epochs: Number of local training epochs

        Returns:
            Dict with training metrics (train_loss, num_samples, num_batches)
        """
        self.initialize_model()

        # Create dataloader
        dataset = TripletDataset(
            self.train_data_path, self.tokenizer, self.max_seq_length
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=False,
        )

        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0

            progress_bar = tqdm(
                dataloader, desc=f"Local Epoch {epoch + 1}/{num_epochs}", leave=False
            )
            for batch in progress_bar:
                # Move to device
                anchor_ids = batch["anchor_input_ids"].to(self.device)
                anchor_mask = batch["anchor_attention_mask"].to(self.device)
                positive_ids = batch["positive_input_ids"].to(self.device)
                positive_mask = batch["positive_attention_mask"].to(self.device)
                negative_ids = batch["negative_input_ids"].to(self.device)
                negative_mask = batch["negative_attention_mask"].to(self.device)

                # Batch encode for efficiency
                batch_size = anchor_ids.size(0)
                combined_ids = torch.cat(
                    [anchor_ids, positive_ids, negative_ids], dim=0
                )
                combined_mask = torch.cat(
                    [anchor_mask, positive_mask, negative_mask], dim=0
                )

                # Forward pass
                self.optimizer.zero_grad()
                combined_emb = self._encode(combined_ids, combined_mask)

                anchor_emb = combined_emb[:batch_size]
                positive_emb = combined_emb[batch_size : 2 * batch_size]
                negative_emb = combined_emb[2 * batch_size :]

                # Compute loss
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({"loss": loss.item()})

            total_loss += epoch_loss

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {
            "train_loss": avg_loss,
            "num_samples": len(dataset),
            "num_batches": num_batches,
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        clear_gpu_memory()
