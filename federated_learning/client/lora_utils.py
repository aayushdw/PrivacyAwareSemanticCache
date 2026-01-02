"""LoRA parameter utilities for Flower clients."""

import torch
import numpy as np
from typing import List, Dict


def get_lora_parameters(model) -> List[np.ndarray]:
    """
    Extract LoRA adapter parameters from the PEFT model.

    Returns parameters in sorted order by name for consistency across clients.

    Args:
        model: PEFT model with LoRA adapters

    Returns:
        List of numpy arrays containing LoRA parameters
    """
    lora_params = []
    lora_names = []

    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            lora_names.append(name)
            lora_params.append(param.detach().cpu().numpy())

    # Sort by name for consistent ordering
    sorted_pairs = sorted(zip(lora_names, lora_params), key=lambda x: x[0])
    return [p for _, p in sorted_pairs]


def set_lora_parameters(model, parameters: List[np.ndarray]) -> None:
    """
    Set LoRA adapter parameters in the PEFT model.

    Parameters must be in sorted order by parameter name.

    Args:
        model: PEFT model with LoRA adapters
        parameters: List of numpy arrays containing new LoRA parameters
    """
    # Get sorted parameter items
    lora_param_items = []
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            lora_param_items.append((name, param))

    sorted_items = sorted(lora_param_items, key=lambda x: x[0])

    if len(sorted_items) != len(parameters):
        raise ValueError(
            f"Parameter count mismatch: model has {len(sorted_items)}, "
            f"received {len(parameters)}"
        )

    for (name, param), new_value in zip(sorted_items, parameters):
        new_tensor = torch.tensor(new_value, dtype=param.dtype, device=param.device)
        if new_tensor.shape != param.shape:
            raise ValueError(
                f"Shape mismatch for {name}: expected {param.shape}, "
                f"got {new_tensor.shape}"
            )
        param.data.copy_(new_tensor)


def get_lora_param_names(model) -> List[str]:
    """Get sorted list of LoRA parameter names."""
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            names.append(name)
    return sorted(names)


def count_lora_parameters(model) -> Dict[str, int]:
    """Count the number of LoRA parameters in the model."""
    total_params = 0
    num_tensors = 0

    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            total_params += param.numel()
            num_tensors += 1

    return {"total_params": total_params, "num_tensors": num_tensors}
