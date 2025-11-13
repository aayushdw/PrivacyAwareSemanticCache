"""
Configuration for Flower Client with LoRA fine-tuning.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClientConfig:
    """Configuration for the Flower LoRA client."""

    # Model configuration
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Training configuration
    learning_rate: float = 2e-4
    batch_size: int = 32
    local_epochs: int = 1  # Number of epochs per federated round
    gradient_accumulation_steps: int = 1
    max_length: int = 128

    # Data configuration
    train_data_path: str = ''
    num_workers: int = 2

    # Device configuration
    device: Optional[str] = None  # Auto-detect if None

    # Flower configuration
    client_id: str = 'client_0'

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.lora_r <= 0:
            raise ValueError(f"lora_r must be positive, got {self.lora_r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        if self.local_epochs <= 0:
            raise ValueError(f"local_epochs must be positive, got {self.local_epochs}")
        if not self.train_data_path:
            raise ValueError("train_data_path must be specified")
