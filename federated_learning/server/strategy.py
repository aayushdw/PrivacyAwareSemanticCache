"""Custom FedAvg strategy for LoRA adapter aggregation."""

from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .weight_manager import ServerWeightManager


class LoRAFedAvg(FedAvg):
    """
    Custom FedAvg strategy for LoRA adapter training.

    Extends FedAvg to:
    1. Send base_model_name and LoRA config to clients
    2. Save aggregated weights after each round
    """

    def __init__(
        self,
        base_model_name: str,
        lora_config: Dict,
        weight_manager: ServerWeightManager,
        initial_parameters: Optional[Parameters] = None,
        **kwargs,
    ):
        """
        Initialize LoRA FedAvg strategy.

        Args:
            base_model_name: HuggingFace model ID for base model
            lora_config: Dict with lora_r, lora_alpha, lora_dropout, target_modules
            weight_manager: Server weight manager for saving checkpoints
            initial_parameters: Initial LoRA parameters
            **kwargs: Additional arguments for FedAvg
        """
        super().__init__(initial_parameters=initial_parameters, **kwargs)
        self.base_model_name = base_model_name
        self.lora_config = lora_config
        self.weight_manager = weight_manager
        self.current_round = 0

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure clients for training with model config."""
        # Get base configuration from parent
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        # Add model configuration for clients
        config.update(
            {
                "base_model_name": self.base_model_name,
                "lora_r": self.lora_config["lora_r"],
                "lora_alpha": self.lora_config["lora_alpha"],
                "lora_dropout": self.lora_config["lora_dropout"],
                "target_modules": ",".join(self.lora_config["target_modules"]),
                "freeze_lora_a": str(self.lora_config.get("freeze_lora_a", True)).lower(),
                "server_round": server_round,
            }
        )

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create FitIns for each client
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate LoRA weights using FedAvg and save checkpoint."""
        if not results:
            return None, {}

        # Log failures
        if failures:
            print(f"Round {server_round}: {len(failures)} client(s) failed")

        # Call parent's aggregate_fit for weighted averaging
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Save aggregated weights
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            self.weight_manager.save_aggregated_weights(ndarrays, server_round)

        # Add aggregation metrics
        metrics["num_clients"] = len(results)
        avg_loss = sum(
            r.metrics.get("train_loss", 0) * r.num_examples
            for _, r in results
        ) / sum(r.num_examples for _, r in results)
        metrics["avg_train_loss"] = avg_loss

        print(
            f"Round {server_round}: Aggregated {len(results)} clients, "
            f"avg_loss={avg_loss:.4f}"
        )

        self.current_round = server_round
        return aggregated_parameters, metrics
