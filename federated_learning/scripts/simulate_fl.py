"""Simulation mode for FL - runs server and clients in a single process."""

import argparse
from typing import Dict

import flwr as fl
from flwr.common import Context, ndarrays_to_parameters
from flwr.simulation import start_simulation

from federated_learning.client import LoRAFlowerClient
from federated_learning.config import FLClientConfig, FLServerConfig
from federated_learning.server.strategy import LoRAFedAvg
from federated_learning.server.weight_manager import ServerWeightManager


def client_fn(context: Context) -> fl.client.Client:
    """Create a client for simulation using Flower Context API."""
    # Get client ID from partition_id in context
    cid = str(context.node_config.get("partition-id", context.node_id))
    config = FLClientConfig(
        client_id=f"client_{cid}",
        train_data_path=f"data/client_data/client_{cid}/train.csv",
    )
    return LoRAFlowerClient(config).to_client()


def run_simulation(
    num_clients: int = 2,
    num_rounds: int = 3,
    lora_path: str = "embedding_pipeline/outputs/models/mpnet-base_lora",
    base_model: str = "sentence-transformers/all-mpnet-base-v2",
    output_dir: str = "federated_learning/outputs",
    enable_dp: bool = False,
    dp_epsilon: float = 8.0,
    dp_delta: float = 1e-5,
    dp_max_grad_norm: float = 1.0,
) -> Dict:
    """
    Run FL simulation with multiple clients in a single process.

    Args:
        num_clients: Number of simulated clients
        num_rounds: Number of FL rounds
        lora_path: Path to initial LoRA adapter
        base_model: HuggingFace base model ID
        output_dir: Output directory for results
        enable_dp: Enable differential privacy (DP-SGD)
        dp_epsilon: Target epsilon for DP
        dp_delta: Target delta for DP
        dp_max_grad_norm: Max gradient norm for DP clipping

    Returns:
        History of FL training
    """
    print("=" * 60)
    print("Federated Learning Simulation")
    print("=" * 60)
    print(f"  Clients: {num_clients}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Base model: {base_model}")
    print(f"  LoRA path: {lora_path}")
    print(f"  Differential Privacy: {'Enabled' if enable_dp else 'Disabled'}")
    if enable_dp:
        print(f"    Epsilon: {dp_epsilon}, Delta: {dp_delta}")
    print("=" * 60)

    # Create server config
    server_config = FLServerConfig(
        base_model_name=base_model,
        initial_lora_path=lora_path,
        num_rounds=num_rounds,
        min_clients=num_clients,
        min_available_clients=num_clients,
        output_dir=output_dir,
        enable_dp=enable_dp,
        dp_epsilon=dp_epsilon,
        dp_delta=dp_delta,
        dp_max_grad_norm=dp_max_grad_norm,
    )

    # Initialize weight manager and load initial weights
    weight_manager = ServerWeightManager(server_config)
    initial_weights, _ = weight_manager.load_initial_lora_weights()
    initial_parameters = ndarrays_to_parameters(initial_weights)

    # Create LoRA config
    lora_config = {
        "lora_r": server_config.lora_r,
        "lora_alpha": server_config.lora_alpha,
        "lora_dropout": server_config.lora_dropout,
        "target_modules": server_config.target_modules,
    }

    # Create DP config
    dp_config = {
        "enable_dp": server_config.enable_dp,
        "dp_epsilon": server_config.dp_epsilon,
        "dp_delta": server_config.dp_delta,
        "dp_max_grad_norm": server_config.dp_max_grad_norm,
    }

    # Create strategy
    strategy = LoRAFedAvg(
        base_model_name=server_config.base_model_name,
        lora_config=lora_config,
        weight_manager=weight_manager,
        dp_config=dp_config,
        initial_parameters=initial_parameters,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
    )

    # Run simulation
    history = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.0},
    )

    print("\n" + "=" * 60)
    print("Simulation Complete")
    print(f"Final model saved to: {output_dir}/round_{num_rounds}/")
    print("=" * 60)

    return history


def main():
    parser = argparse.ArgumentParser(
        description="Run FL simulation (server + clients in single process)"
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=2,
        help="Number of simulated clients (default: 2)",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=3,
        help="Number of FL rounds (default: 3)",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="embedding_pipeline/outputs/models/mpnet-base_lora",
        help="Path to initial LoRA adapter",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Base model name (HuggingFace model ID)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="federated_learning/outputs",
        help="Output directory for aggregated weights",
    )
    parser.add_argument(
        "--enable-dp",
        action="store_true",
        help="Enable differential privacy (DP-SGD)",
    )
    parser.add_argument(
        "--dp-epsilon",
        type=float,
        default=8.0,
        help="Target epsilon for DP (default: 8.0)",
    )
    parser.add_argument(
        "--dp-delta",
        type=float,
        default=1e-5,
        help="Target delta for DP (default: 1e-5)",
    )
    parser.add_argument(
        "--dp-max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for DP clipping (default: 1.0)",
    )

    args = parser.parse_args()

    run_simulation(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        lora_path=args.lora_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        enable_dp=args.enable_dp,
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        dp_max_grad_norm=args.dp_max_grad_norm,
    )


if __name__ == "__main__":
    main()
