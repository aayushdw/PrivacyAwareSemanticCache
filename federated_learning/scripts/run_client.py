"""CLI entry point for FL client."""

import argparse

from federated_learning.client import start_client
from federated_learning.config import FLClientConfig


def main():
    parser = argparse.ArgumentParser(
        description="Start Federated Learning client for LoRA training"
    )
    parser.add_argument(
        "--client-id",
        type=str,
        required=True,
        help="Unique client identifier",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to client's triplet training CSV",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="127.0.0.1:8080",
        help="Server address (default: 127.0.0.1:8080)",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Number of local training epochs per round (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )

    args = parser.parse_args()

    config = FLClientConfig(
        client_id=args.client_id,
        train_data_path=args.train_data,
        server_address=args.server,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    print(f"Starting FL Client {args.client_id}")
    print(f"  Training data: {args.train_data}")
    print(f"  Server: {args.server}")
    print(f"  Local epochs: {args.local_epochs}")

    start_client(config)


if __name__ == "__main__":
    main()
