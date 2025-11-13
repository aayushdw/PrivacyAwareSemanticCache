"""
Flower Client for Privacy-Aware Federated Learning with LoRA.

This module implements a Flower client that performs LoRA fine-tuning
on local data while maintaining state of the LoRA adapter weights.
"""

from .lora_client import LoRAFlowerClient
from .config import ClientConfig

__all__ = ['LoRAFlowerClient', 'ClientConfig']
