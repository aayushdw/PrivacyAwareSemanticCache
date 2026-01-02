"""Flower FL Server for LoRA training."""

import flwr as fl
from flwr.common import ndarrays_to_parameters

from ..config import FLServerConfig
from .strategy import LoRAFedAvg
from .weight_manager import ServerWeightManager


def start_server(config: FLServerConfig) -> None:
    """
    Start the Flower FL server.

    Args:
        config: Server configuration
    """
    print("=" * 60)
    print("Federated Learning Server for LoRA Training")
    print("=" * 60)

    # Initialize weight manager
    weight_manager = ServerWeightManager(config)

    # Load initial LoRA weights
    initial_weights, param_names = weight_manager.load_initial_lora_weights()
    initial_parameters = ndarrays_to_parameters(initial_weights)

    print(f"\nConfiguration:")
    print(f"  Base model: {config.base_model_name}")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  Number of LoRA parameters: {len(param_names)}")
    print(f"  FL rounds: {config.num_rounds}")
    print(f"  Min clients: {config.min_clients}")
    print(f"  Server address: {config.server_address}")

    # Create LoRA config dict
    lora_config = {
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": config.target_modules,
    }

    # Create strategy
    strategy = LoRAFedAvg(
        base_model_name=config.base_model_name,
        lora_config=lora_config,
        weight_manager=weight_manager,
        initial_parameters=initial_parameters,
        min_fit_clients=config.min_clients,
        min_available_clients=config.min_available_clients,
        fraction_fit=config.fraction_fit,
        fraction_evaluate=config.fraction_evaluate,
    )

    print(f"\nWaiting for {config.min_clients} client(s) to connect...")
    print("-" * 60)

    # Start server
    fl.server.start_server(
        server_address=config.server_address,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
    )

    print("\n" + "=" * 60)
    print("Federated Learning Complete")
    print(f"Final model saved to: {config.output_dir}/round_{config.num_rounds}/")
    print("=" * 60)
