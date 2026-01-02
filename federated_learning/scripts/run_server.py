"""CLI entry point for FL server."""

import argparse

from federated_learning.config import FLServerConfig
from federated_learning.server import start_server


def main():
    parser = argparse.ArgumentParser(
        description="Start Federated Learning server for LoRA training"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
        help="Number of FL rounds (default: 5)",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        help="Minimum number of clients required (default: 2)",
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
        "--address",
        type=str,
        default="[::]:8080",
        help="Server address (default: [::]:8080)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="federated_learning/outputs",
        help="Output directory for aggregated weights",
    )

    args = parser.parse_args()

    config = FLServerConfig(
        base_model_name=args.base_model,
        initial_lora_path=args.lora_path,
        num_rounds=args.num_rounds,
        min_clients=args.min_clients,
        min_available_clients=args.min_clients,
        server_address=args.address,
        output_dir=args.output_dir,
    )

    start_server(config)


if __name__ == "__main__":
    main()
