"""Local training logic for FL clients."""

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from embedding_pipeline.lora_training.lora_trainer import TripletDataset
from embedding_pipeline.utils.device_utils import clear_gpu_memory, get_device

from .lora_utils import freeze_lora_a


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
        freeze_lora_a: bool = True,
        dp_config: Optional[Dict] = None,
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
            freeze_lora_a: If True, freeze lora_A and only train lora_B for stability
            dp_config: Optional dict with enable_dp, dp_epsilon, dp_delta, dp_max_grad_norm
        """
        self.base_model_name = base_model_name
        self.lora_config = lora_config
        self.train_data_path = train_data_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.freeze_lora_a_flag = freeze_lora_a

        # DP configuration
        self.dp_config = dp_config or {"enable_dp": False}
        self.enable_dp = self.dp_config.get("enable_dp", False)

        self.device = get_device()
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.privacy_engine = None
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

        # Validate model for Opacus compatibility if DP is enabled
        if self.enable_dp:
            if not ModuleValidator.is_valid(self.model):
                self.model = ModuleValidator.fix(self.model)
                print("Fixed model for Opacus compatibility")

        self.model.to(self.device)

        # Freeze lora_A if configured (only train lora_B for stability)
        if self.freeze_lora_a_flag:
            frozen_count = freeze_lora_a(self.model)
            print(f"Froze {frozen_count} lora_A parameters, training only lora_B")

        # Setup optimizer (only includes trainable parameters)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
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

    def _setup_privacy_engine(self, dataloader: DataLoader) -> DataLoader:
        """Wrap model, optimizer, and dataloader with Opacus PrivacyEngine."""
        epsilon = self.dp_config.get("dp_epsilon", 8.0)
        delta = self.dp_config.get("dp_delta", 1e-5)
        max_grad_norm = self.dp_config.get("dp_max_grad_norm", 1.0)

        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, dataloader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=dataloader,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
            epochs=1,
            # Use functorch for per-sample gradient computation.
            # This is required for LoRA adapters with frozen base weights because
            # the default hooks-based approach doesn't work when the gradient chain
            # goes through frozen parameters.
            grad_sample_mode="functorch",
        )

        print(f"DP-SGD enabled: epsilon={epsilon}, delta={delta}, max_grad_norm={max_grad_norm}")
        return dataloader

    def train_local_epochs(self, num_epochs: int = 1) -> Dict:
        """
        Train for specified number of local epochs.

        Args:
            num_epochs: Number of local training epochs

        Returns:
            Dict with training metrics (train_loss, num_samples, num_batches)
        """
        self.initialize_model()

        # Create dataloader (num_workers=0 required for Opacus compatibility)
        dataset = TripletDataset(
            self.train_data_path, self.tokenizer, self.max_seq_length
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        # Setup DP if enabled (only once - don't double-wrap in subsequent rounds)
        if self.enable_dp and self.privacy_engine is None:
            dataloader = self._setup_privacy_engine(dataloader)

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

                self.optimizer.zero_grad()

                if self.enable_dp:
                    # For DP-SGD with Opacus: we must only pass the anchor samples
                    # through the DP-wrapped model to get correct per-sample gradients.
                    # Opacus hooks expect the batch dimension to match the dataloader.
                    #
                    # Strategy: Compute anchor embedding with gradients (goes through DP hooks),
                    # compute positive/negative embeddings without gradients using the
                    # underlying base model accessed through the wrapped module.
                    
                    # Get anchor embeddings with DP gradient tracking
                    anchor_emb = self._encode(anchor_ids, anchor_mask)
                    
                    # For positive/negative, we need embeddings for loss but no gradients.
                    # Use torch.no_grad to avoid interfering with Opacus hooks.
                    with torch.no_grad():
                        # Access the underlying model (works for both wrapped and unwrapped)
                        base_model = self.model
                        if hasattr(self.model, '_module'):
                            # GradSampleModule wraps the actual module
                            base_model = self.model._module
                        
                        # Compute positive embeddings
                        pos_outputs = base_model(input_ids=positive_ids, attention_mask=positive_mask)
                        positive_emb = self._mean_pooling(pos_outputs, positive_mask)
                        positive_emb = nn.functional.normalize(positive_emb, p=2, dim=1)
                        
                        # Compute negative embeddings
                        neg_outputs = base_model(input_ids=negative_ids, attention_mask=negative_mask)
                        negative_emb = self._mean_pooling(neg_outputs, negative_mask)
                        negative_emb = nn.functional.normalize(negative_emb, p=2, dim=1)
                else:
                    # Standard combined encoding for efficiency (non-DP mode)
                    batch_size = anchor_ids.size(0)
                    combined_ids = torch.cat(
                        [anchor_ids, positive_ids, negative_ids], dim=0
                    )
                    combined_mask = torch.cat(
                        [anchor_mask, positive_mask, negative_mask], dim=0
                    )
                    combined_emb = self._encode(combined_ids, combined_mask)
                    anchor_emb = combined_emb[:batch_size]
                    positive_emb = combined_emb[batch_size : 2 * batch_size]
                    negative_emb = combined_emb[2 * batch_size :]

                # Compute loss
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                loss.backward()

                # Only clip gradients manually when DP is disabled (Opacus handles it)
                if not self.enable_dp:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # If DP is enabled, Opacus injects Gaussian noise and performs clipping here
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({"loss": loss.item()})

            total_loss += epoch_loss

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Build result dict
        result = {
            "train_loss": avg_loss,
            "num_samples": len(dataset),
            "num_batches": num_batches,
        }

        # Add privacy metrics if DP is enabled
        if self.enable_dp and self.privacy_engine is not None:
            epsilon_spent = self.privacy_engine.get_epsilon(
                delta=self.dp_config.get("dp_delta", 1e-5)
            )
            result["epsilon_spent"] = epsilon_spent
            print(f"Privacy budget spent: epsilon={epsilon_spent:.2f}")

        return result

    def cleanup(self) -> None:
        """Clean up resources."""
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.privacy_engine = None
        clear_gpu_memory()
