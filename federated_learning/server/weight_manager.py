"""Manages LoRA weights for the FL server."""

import os
from typing import List, Tuple

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModel

from ..config import FLServerConfig


class ServerWeightManager:
    """Manages loading and aggregating LoRA weights on the server."""

    def __init__(self, config: FLServerConfig):
        """
        Initialize weight manager.

        Args:
            config: Server configuration
        """
        self.config = config
        self.lora_param_names: List[str] = []

    def load_initial_lora_weights(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load initial LoRA weights from the existing trained model.

        Returns:
            Tuple of (list of numpy arrays, list of parameter names)
        """
        lora_path = self.config.initial_lora_path

        if lora_path and os.path.exists(lora_path):
            # Load from existing safetensors file
            safetensors_path = os.path.join(lora_path, "adapter_model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)

                # Sort keys for consistent ordering across clients
                sorted_keys = sorted(state_dict.keys())
                self.lora_param_names = sorted_keys

                parameters = [state_dict[k].numpy() for k in sorted_keys]
                print(f"Loaded {len(parameters)} LoRA parameters from {safetensors_path}")
                return parameters, sorted_keys

        # Initialize fresh LoRA weights from a new model
        print("No existing LoRA weights found, initializing fresh weights")
        return self._initialize_fresh_lora()

    def _initialize_fresh_lora(self) -> Tuple[List[np.ndarray], List[str]]:
        """Initialize fresh LoRA weights by creating a new PEFT model."""
        base_model = AutoModel.from_pretrained(self.config.base_model_name)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        peft_model = get_peft_model(base_model, lora_config)

        # Extract LoRA parameters (sorted by name)
        lora_params = []
        lora_names = []

        for name, param in peft_model.named_parameters():
            if param.requires_grad and ("lora_A" in name or "lora_B" in name):
                lora_names.append(name)
                lora_params.append(param.detach().cpu().numpy())

        # Sort by name for consistency
        sorted_pairs = sorted(zip(lora_names, lora_params), key=lambda x: x[0])
        self.lora_param_names = [n for n, _ in sorted_pairs]
        sorted_params = [p for _, p in sorted_pairs]

        # Clean up
        del peft_model
        del base_model

        print(f"Initialized {len(sorted_params)} fresh LoRA parameters")
        return sorted_params, self.lora_param_names

    def save_aggregated_weights(
        self, parameters: List[np.ndarray], round_num: int
    ) -> str:
        """
        Save aggregated weights after each round.

        Args:
            parameters: List of numpy arrays containing aggregated LoRA weights
            round_num: Current FL round number

        Returns:
            Path where weights were saved
        """
        output_dir = os.path.join(self.config.output_dir, f"round_{round_num}")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "adapter_model.safetensors")

        state_dict = {
            name: torch.tensor(param)
            for name, param in zip(self.lora_param_names, parameters)
        }
        save_file(state_dict, output_path)

        # Also save adapter_config.json for compatibility
        config_path = os.path.join(output_dir, "adapter_config.json")
        self._save_adapter_config(config_path)

        print(f"Saved aggregated weights to {output_path}")
        return output_path

    def _save_adapter_config(self, config_path: str) -> None:
        """Save adapter_config.json for PEFT compatibility."""
        import json

        config = {
            "peft_type": "LORA",
            "base_model_name_or_path": self.config.base_model_name,
            "task_type": "FEATURE_EXTRACTION",
            "r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "target_modules": self.config.target_modules,
            "bias": "none",
            "inference_mode": True,
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
