"""Utility to split training data into client partitions."""

import os
from pathlib import Path

import pandas as pd


def split_data_for_clients(
    source_csv: str,
    num_clients: int,
    output_dir: str = "data/client_data",
    shuffle: bool = True,
    random_seed: int = 42,
) -> list[str]:
    """
    Split triplet training data into equal partitions for FL clients.

    Args:
        source_csv: Path to source triplet CSV (with anchor, positive, negative columns)
        num_clients: Number of client partitions to create
        output_dir: Base directory for client data
        shuffle: Whether to shuffle data before splitting
        random_seed: Random seed for reproducibility

    Returns:
        List of paths to created client data files
    """
    # Load source data
    df = pd.read_csv(source_csv)
    print(f"Loaded {len(df)} samples from {source_csv}")

    # Validate columns
    required_cols = ["anchor", "positive", "negative"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Shuffle if requested
    if shuffle:
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Calculate split sizes
    samples_per_client = len(df) // num_clients
    remainder = len(df) % num_clients

    # Split and save
    output_paths = []
    start_idx = 0

    for client_id in range(num_clients):
        # Add one extra sample to first 'remainder' clients
        extra = 1 if client_id < remainder else 0
        end_idx = start_idx + samples_per_client + extra

        client_df = df.iloc[start_idx:end_idx]

        # Create client directory
        client_dir = Path(output_dir) / f"client_{client_id}"
        client_dir.mkdir(parents=True, exist_ok=True)

        # Save client data
        output_path = client_dir / "train.csv"
        client_df.to_csv(output_path, index=False)

        output_paths.append(str(output_path))
        print(f"Client {client_id}: {len(client_df)} samples -> {output_path}")

        start_idx = end_idx

    print(f"\nCreated {num_clients} client partitions in {output_dir}")
    return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split data for FL clients")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source triplet CSV",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=2,
        help="Number of client partitions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/client_data",
        help="Output directory for client data",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle data before splitting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )

    args = parser.parse_args()

    split_data_for_clients(
        source_csv=args.source,
        num_clients=args.num_clients,
        output_dir=args.output_dir,
        shuffle=not args.no_shuffle,
        random_seed=args.seed,
    )
