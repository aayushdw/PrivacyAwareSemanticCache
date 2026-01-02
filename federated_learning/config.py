"""Configuration dataclasses for Federated Learning."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FLServerConfig:
    """Configuration for the Federated Learning server."""

    # Model configuration
    base_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    initial_lora_path: Optional[str] = "embedding_pipeline/outputs/models/mpnet-base_lora"

    # LoRA configuration (must match across server and clients)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        # MPNet attention layer names for LoRA
        # Using 'q', 'k', 'v', 'o' (not 'query', 'key', 'value', 'dense')
        # This is required for Opacus DP-SGD compatibility
        default_factory=lambda: ["q", "k", "v", "o"]
    )
    freeze_lora_a: bool = True  # Freeze lora_A matrix, only train lora_B for stability

    # FL configuration
    num_rounds: int = 5
    min_clients: int = 2
    min_available_clients: int = 2
    fraction_fit: float = 1.0  # Use all available clients for training
    fraction_evaluate: float = 0.0  # Disable evaluation rounds

    # Server network
    server_address: str = "[::]:8080"

    # Output
    output_dir: str = "federated_learning/outputs"

    # Differential Privacy configuration
    enable_dp: bool = False  # DP is off by default
    dp_epsilon: float = 8.0  # Target epsilon (privacy budget)
    dp_delta: float = 1e-5  # Target delta (failure probability)
    dp_max_grad_norm: float = 1.0  # Gradient clipping bound for DP-SGD


@dataclass
class FLClientConfig:
    """Configuration for FL clients."""

    # Data
    train_data_path: str = ""  # Required: path to client's triplet CSV

    # Training
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 2e-4
    max_seq_length: int = 128

    # Client identification
    client_id: str = "client_0"

    # Server connection
    server_address: str = "127.0.0.1:8080"
