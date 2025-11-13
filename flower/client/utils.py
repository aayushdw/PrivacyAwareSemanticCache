"""
Utility functions for handling LoRA weights in Flower federated learning.
"""

import torch
import numpy as np
from typing import List, Dict, OrderedDict
from collections import OrderedDict as OrderedDictType


def get_lora_parameters(model) -> List[np.ndarray]:
    """
    Extract only LoRA adapter parameters from the model.

    Args:
        model: PEFT model with LoRA adapters

    Returns:
        List of numpy arrays containing LoRA parameters
    """
    lora_params = []

    for name, param in model.named_parameters():
        # Only extract trainable LoRA parameters (lora_A and lora_B)
        if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
            lora_params.append(param.detach().cpu().numpy())

    return lora_params


def get_lora_state_dict(model) -> OrderedDictType[str, np.ndarray]:
    """
    Extract LoRA parameters as a state dictionary.

    Args:
        model: PEFT model with LoRA adapters

    Returns:
        OrderedDict mapping parameter names to numpy arrays
    """
    lora_state = OrderedDict()

    for name, param in model.named_parameters():
        if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
            lora_state[name] = param.detach().cpu().numpy()

    return lora_state


def set_lora_parameters(model, parameters: List[np.ndarray]) -> None:
    """
    Update LoRA adapter parameters in the model.

    Args:
        model: PEFT model with LoRA adapters
        parameters: List of numpy arrays containing new LoRA parameters
    """
    param_idx = 0

    for name, param in model.named_parameters():
        if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
            if param_idx >= len(parameters):
                raise ValueError(f"Not enough parameters provided. Expected more for {name}")

            # Convert numpy array to tensor and update parameter
            new_param = torch.tensor(parameters[param_idx], dtype=param.dtype, device=param.device)

            # Verify shape matches
            if new_param.shape != param.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {param.shape}, got {new_param.shape}"
                )

            param.data.copy_(new_param)
            param_idx += 1

    if param_idx != len(parameters):
        raise ValueError(
            f"Parameter count mismatch: model expects {param_idx}, got {len(parameters)}"
        )


def set_lora_state_dict(model, state_dict: Dict[str, np.ndarray]) -> None:
    """
    Update LoRA parameters from a state dictionary.

    Args:
        model: PEFT model with LoRA adapters
        state_dict: Dictionary mapping parameter names to numpy arrays
    """
    for name, param in model.named_parameters():
        if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
            if name not in state_dict:
                raise ValueError(f"Missing parameter in state_dict: {name}")

            # Convert numpy array to tensor and update parameter
            new_param = torch.tensor(state_dict[name], dtype=param.dtype, device=param.device)

            # Verify shape matches
            if new_param.shape != param.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {param.shape}, got {new_param.shape}"
                )

            param.data.copy_(new_param)


def count_lora_parameters(model) -> Dict[str, int]:
    """
    Count the number of LoRA parameters in the model.

    Args:
        model: PEFT model with LoRA adapters

    Returns:
        Dictionary with parameter counts
    """
    total_lora_params = 0
    lora_param_names = []

    for name, param in model.named_parameters():
        if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
            total_lora_params += param.numel()
            lora_param_names.append(name)

    return {
        'total_lora_params': total_lora_params,
        'num_lora_tensors': len(lora_param_names),
        'lora_param_names': lora_param_names
    }


def calculate_lora_size_mb(parameters: List[np.ndarray]) -> float:
    """
    Calculate the size of LoRA parameters in megabytes.

    Args:
        parameters: List of numpy arrays containing LoRA parameters

    Returns:
        Size in megabytes
    """
    total_bytes = sum(param.nbytes for param in parameters)
    return total_bytes / (1024 * 1024)
