"""Data loading utilities with DVC integration."""

import subprocess
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ..config.pipeline_config import get_project_root, get_data_path


def load_dataset(
    dataset_path: str,
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load a triplet dataset from the project.

    Args:
        dataset_path: Path relative to project root
        max_samples: Maximum samples to load (random sample)
        random_state: Random seed for sampling

    Returns:
        DataFrame with triplets (anchor, positive, negative)
    """
    full_path = get_data_path(dataset_path)

    if not full_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {full_path}. "
            "Run 'dvc pull' to fetch data."
        )

    df = pd.read_csv(full_path)

    # Clean dataset
    df = clean_dataset(df)

    # Sample if requested
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean triplet dataset by removing rows with missing or invalid data.

    Args:
        df: DataFrame with triplets

    Returns:
        Cleaned DataFrame
    """
    original_len = len(df)

    # Required columns for triplet format
    required_cols = ["anchor", "positive", "negative"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Remove rows with missing values
    df = df.dropna(subset=required_cols)

    # Remove rows with empty strings
    df = df[df["anchor"].str.strip() != ""]
    df = df[df["positive"].str.strip() != ""]
    df = df[df["negative"].str.strip() != ""]

    # Reset index
    df = df.reset_index(drop=True)

    cleaned_len = len(df)
    if cleaned_len < original_len:
        print(
            f"Cleaned dataset: {original_len} -> {cleaned_len} rows "
            f"({original_len - cleaned_len} removed)"
        )

    return df


def load_train_dataset(max_samples: Optional[int] = None) -> pd.DataFrame:
    """Load training dataset."""
    return load_dataset("data/processed/train.csv", max_samples=max_samples)


def load_mini_train_dataset(max_samples: Optional[int] = None) -> pd.DataFrame:
    """Load mini training dataset for threshold tuning."""
    return load_dataset("data/processed/mini_train.csv", max_samples=max_samples)


def load_val_dataset(max_samples: Optional[int] = None) -> pd.DataFrame:
    """Load validation dataset."""
    return load_dataset("data/processed/val.csv", max_samples=max_samples)


def load_mini_val_dataset(max_samples: Optional[int] = None) -> pd.DataFrame:
    """Load mini validation dataset."""
    return load_dataset("data/processed/mini_val.csv", max_samples=max_samples)


def load_test_dataset(max_samples: Optional[int] = None) -> pd.DataFrame:
    """Load test dataset."""
    return load_dataset("data/processed/test.csv", max_samples=max_samples)


def load_mini_test_dataset(max_samples: Optional[int] = None) -> pd.DataFrame:
    """Load mini test dataset for quick evaluation."""
    return load_dataset("data/processed/mini_test.csv", max_samples=max_samples)


def get_dvc_dataset_version() -> str:
    """Get current DVC dataset version hash."""
    try:
        # Get the Git commit of the last DVC data change
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", "*.dvc"],
            capture_output=True,
            text=True,
            cwd=get_project_root(),
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    # Fallback: try to read dvc.lock
    dvc_lock = get_project_root() / "dvc.lock"
    if dvc_lock.exists():
        try:
            import hashlib

            with open(dvc_lock, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()[:8]
        except Exception:
            pass

    return "unknown"


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Get statistics about a triplet dataset."""
    return {
        "total_triplets": len(df),
        "avg_anchor_length": df["anchor"].str.len().mean(),
        "avg_positive_length": df["positive"].str.len().mean(),
        "avg_negative_length": df["negative"].str.len().mean(),
    }


def get_triplets(df: pd.DataFrame) -> Tuple[list, list, list]:
    """Extract triplets from DataFrame."""
    anchors = df["anchor"].tolist()
    positives = df["positive"].tolist()
    negatives = df["negative"].tolist()
    return anchors, positives, negatives


def triplets_to_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert triplet dataset to pair dataset for threshold-based evaluation.

    Creates two rows per triplet:
    - (anchor, positive) with label=1 (similar)
    - (anchor, negative) with label=0 (dissimilar)

    Args:
        df: Triplet DataFrame

    Returns:
        Pair DataFrame with columns: question1, question2, is_similar
    """
    positive_pairs = pd.DataFrame({
        "question1": df["anchor"],
        "question2": df["positive"],
        "is_similar": 1,
    })

    negative_pairs = pd.DataFrame({
        "question1": df["anchor"],
        "question2": df["negative"],
        "is_similar": 0,
    })

    pairs_df = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
    return pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)
