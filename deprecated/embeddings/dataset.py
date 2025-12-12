"""
Dataset loader for Privacy Aware Semantic Cache.
Loads questions dataset from CSV into a pandas DataFrame.
"""

import os
import pandas as pd


def get_csv_file_path(split: str = 'train') -> str:
    """
    Get the path to the questions CSV file.

    Args:
        split: Dataset split to load ('train', 'val', 'test', or 'full')
               Default is 'train'. Use 'full' for original questions.csv

    Returns:
        Absolute path to the specified questions CSV file
    """
    # Get the project root directory (parent of embeddings directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Map split names to file names
    if split == 'full':
        filename = 'questions.csv'
    elif split in ['train', 'val', 'test']:
        filename = f'questions_{split}.csv'
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', 'test', or 'full'")

    csv_file_path = os.path.join(project_root, 'data', filename)
    return csv_file_path


def load_dataset(csv_file_path: str = None, split: str = 'train') -> pd.DataFrame:
    """
    Load the questions dataset from CSV file into a DataFrame.

    Args:
        csv_file_path: Optional path to CSV file. If provided, split parameter is ignored.
        split: Dataset split to load ('train', 'val', 'test', or 'full')
               Default is 'train'. Only used if csv_file_path is None.

    Returns:
        pandas DataFrame containing the questions dataset
    """
    if csv_file_path is None:
        csv_file_path = get_csv_file_path(split)

    df = pd.read_csv(csv_file_path, escapechar='\\', sep=',')
    return df


def load_train_dataset() -> pd.DataFrame:
    """Load the training dataset."""
    return load_dataset(split='train')


def load_val_dataset() -> pd.DataFrame:
    """Load the validation dataset."""
    return load_dataset(split='val')


def load_test_dataset() -> pd.DataFrame:
    """Load the test dataset."""
    return load_dataset(split='test')


def load_full_dataset() -> pd.DataFrame:
    """Load the full original dataset."""
    return load_dataset(split='full')


if __name__ == "__main__":
    # Example usage
    print("Loading training dataset...")
    train_df = load_train_dataset()
    print(f"Train dataset: {len(train_df)} rows")

    print("\nLoading validation dataset...")
    val_df = load_val_dataset()
    print(f"Val dataset: {len(val_df)} rows")

    print("\nLoading test dataset...")
    test_df = load_test_dataset()
    print(f"Test dataset: {len(test_df)} rows")

    print(f"\nColumns: {list(train_df.columns)}")

    print(f"\nDataset file paths:")
    print(f"  Train: {get_csv_file_path('train')}")
    print(f"  Val:   {get_csv_file_path('val')}")
    print(f"  Test:  {get_csv_file_path('test')}")
    print(f"  Full:  {get_csv_file_path('full')}")
