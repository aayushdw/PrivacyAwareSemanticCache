"""Flower client for federated LoRA training."""

import flwr as fl
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from ..config import FLClientConfig
from .local_trainer import LocalLoRATrainer
from .lora_utils import get_lora_parameters, set_lora_parameters


class LoRAFlowerClient(fl.client.Client):
    """
    Flower client for federated LoRA training.

    Receives base_model_name and LoRA config from server, trains locally
    on triplet data, and returns updated LoRA adapter weights.
    """

    def __init__(self, config: FLClientConfig):
        """Initialize Flower client."""
        self.config = config
        self.trainer: LocalLoRATrainer = None
        self.base_model_name: str = None
        self.lora_config: dict = None
        self._initialized = False

    def _ensure_initialized(self, config: dict) -> None:
        """Initialize trainer from server config if not already done."""
        if self._initialized:
            return

        # Parse base model name from server config
        self.base_model_name = config.get("base_model_name")
        if not self.base_model_name:
            raise ValueError("Server did not provide base_model_name")

        # Parse LoRA config from server
        target_modules_str = config.get("target_modules", "query,key,value,dense")
        self.lora_config = {
            "lora_r": int(config.get("lora_r", 16)),
            "lora_alpha": int(config.get("lora_alpha", 32)),
            "lora_dropout": float(config.get("lora_dropout", 0.1)),
            "target_modules": target_modules_str.split(","),
        }

        # Parse freeze_lora_a config (default True for stability)
        self.freeze_lora_a = config.get("freeze_lora_a", "true").lower() == "true"

        # Parse DP configuration from server
        self.dp_config = {
            "enable_dp": config.get("enable_dp", "false").lower() == "true",
            "dp_epsilon": float(config.get("dp_epsilon", 8.0)),
            "dp_delta": float(config.get("dp_delta", 1e-5)),
            "dp_max_grad_norm": float(config.get("dp_max_grad_norm", 1.0)),
        }

        # Create trainer
        self.trainer = LocalLoRATrainer(
            base_model_name=self.base_model_name,
            lora_config=self.lora_config,
            train_data_path=self.config.train_data_path,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            max_seq_length=self.config.max_seq_length,
            freeze_lora_a=self.freeze_lora_a,
            dp_config=self.dp_config,
        )

        self._initialized = True
        print(f"Client {self.config.client_id} initialized")
        print(f"  Base model: {self.base_model_name}")
        print(
            f"  LoRA config: r={self.lora_config['lora_r']}, "
            f"alpha={self.lora_config['lora_alpha']}, freeze_lora_a={self.freeze_lora_a}"
        )
        if self.dp_config["enable_dp"]:
            print(
                f"  DP-SGD: epsilon={self.dp_config['dp_epsilon']}, "
                f"delta={self.dp_config['dp_delta']}"
            )
        else:
            print("  DP-SGD: Disabled")

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Get current LoRA parameters (only lora_B if lora_A is frozen)."""
        if self.trainer is None or self.trainer.model is None:
            # Return empty if not initialized
            return GetParametersRes(
                status=Status(code=Code.OK, message="Not initialized"),
                parameters=ndarrays_to_parameters([]),
            )

        lora_params = get_lora_parameters(
            self.trainer.model, lora_b_only=self.freeze_lora_a
        )
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(lora_params),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Train on local data and return updated LoRA parameters."""
        # Ensure initialized from server config
        self._ensure_initialized(ins.config)

        # Initialize model if needed
        self.trainer.initialize_model()

        # Apply received LoRA weights from server (only lora_B if lora_A is frozen)
        server_weights = parameters_to_ndarrays(ins.parameters)
        if len(server_weights) > 0:
            set_lora_parameters(
                self.trainer.model, server_weights, lora_b_only=self.freeze_lora_a
            )
            print(
                f"Client {self.config.client_id}: Applied {len(server_weights)} "
                f"{'lora_B' if self.freeze_lora_a else 'LoRA'} parameters from server"
            )

        # Train locally
        server_round = ins.config.get("server_round", 0)
        print(f"Client {self.config.client_id}: Starting local training (round {server_round})")
        metrics = self.trainer.train_local_epochs(self.config.local_epochs)
        print(
            f"Client {self.config.client_id}: Completed training, "
            f"loss={metrics['train_loss']:.4f}"
        )

        # Extract updated LoRA weights (only lora_B if lora_A is frozen)
        updated_weights = get_lora_parameters(
            self.trainer.model, lora_b_only=self.freeze_lora_a
        )

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(updated_weights),
            num_examples=metrics["num_samples"],
            metrics={"train_loss": metrics["train_loss"]},
        )


def start_client(config: FLClientConfig) -> None:
    """Start Flower client and connect to server."""
    client = LoRAFlowerClient(config)
    fl.client.start_client(
        server_address=config.server_address,
        client=client,
    )
