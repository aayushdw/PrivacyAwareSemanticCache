"""
Split questions.csv dataset into train, validation, and test sets.
Splits: 50% train, 30% validation, 20% test
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    input_file: str = 'questions.csv',
    output_dir: str = None,
    train_size: float = 0.5,
    val_size: float = 0.3,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split questions dataset into train, validation, and test sets.

    Args:
        input_file: Path to input CSV file (default: questions.csv)
        output_dir: Directory to save split files (default: same as input_file)
        train_size: Proportion of data for training (default: 0.5)
        val_size: Proportion of data for validation (default: 0.3)
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    """
    # Validate split proportions
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split proportions must sum to 1.0, got {total}")

    # Determine file paths
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_file))

    if not os.path.isabs(input_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(script_dir, input_file)

    print(f"Reading dataset from: {input_file}")

    # Read the dataset
    df = pd.read_csv(input_file, escapechar='\\', sep=',')
    total_rows = len(df)

    print(f"Total rows: {total_rows}")
    print(f"Columns: {list(df.columns)}")

    # Calculate expected sizes
    expected_train = int(total_rows * train_size)
    expected_val = int(total_rows * val_size)
    expected_test = total_rows - expected_train - expected_val

    print(f"\nTarget split:")
    print(f"  Train: {train_size*100:.1f}% (~{expected_train} rows)")
    print(f"  Val:   {val_size*100:.1f}% (~{expected_val} rows)")
    print(f"  Test:  {test_size*100:.1f}% (~{expected_test} rows)")

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # Second split: separate train and validation from remaining data
    # val_size_adjusted accounts for the fact we already removed test data
    val_size_adjusted = val_size / (train_size + val_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=True
    )

    print(f"\nActual split:")
    print(f"  Train: {len(train_df)} rows ({len(train_df)/total_rows*100:.1f}%)")
    print(f"  Val:   {len(val_df)} rows ({len(val_df)/total_rows*100:.1f}%)")
    print(f"  Test:  {len(test_df)} rows ({len(test_df)/total_rows*100:.1f}%)")

    # Save split datasets
    train_path = os.path.join(output_dir, 'questions_train.csv')
    val_path = os.path.join(output_dir, 'questions_val.csv')
    test_path = os.path.join(output_dir, 'questions_test.csv')

    print(f"\nSaving splits...")
    train_df.to_csv(train_path, index=False)
    print(f"  Train saved to: {train_path}")

    val_df.to_csv(val_path, index=False)
    print(f"  Val saved to:   {val_path}")

    test_df.to_csv(test_path, index=False)
    print(f"  Test saved to:  {test_path}")

    # Print statistics about duplicate distribution
    print(f"\nDuplicate distribution:")
    print(f"  Original:")
    print(f"    Duplicates: {df['is_duplicate'].sum()} ({df['is_duplicate'].sum()/len(df)*100:.1f}%)")
    print(f"    Non-duplicates: {(~df['is_duplicate'].astype(bool)).sum()} ({(~df['is_duplicate'].astype(bool)).sum()/len(df)*100:.1f}%)")

    print(f"  Train:")
    print(f"    Duplicates: {train_df['is_duplicate'].sum()} ({train_df['is_duplicate'].sum()/len(train_df)*100:.1f}%)")
    print(f"    Non-duplicates: {(~train_df['is_duplicate'].astype(bool)).sum()} ({(~train_df['is_duplicate'].astype(bool)).sum()/len(train_df)*100:.1f}%)")

    print(f"  Val:")
    print(f"    Duplicates: {val_df['is_duplicate'].sum()} ({val_df['is_duplicate'].sum()/len(val_df)*100:.1f}%)")
    print(f"    Non-duplicates: {(~val_df['is_duplicate'].astype(bool)).sum()} ({(~val_df['is_duplicate'].astype(bool)).sum()/len(val_df)*100:.1f}%)")

    print(f"  Test:")
    print(f"    Duplicates: {test_df['is_duplicate'].sum()} ({test_df['is_duplicate'].sum()/len(test_df)*100:.1f}%)")
    print(f"    Non-duplicates: {(~test_df['is_duplicate'].astype(bool)).sum()} ({(~test_df['is_duplicate'].astype(bool)).sum()/len(test_df)*100:.1f}%)")

    print(f"\n✓ Dataset split completed successfully!")

    return train_df, val_df, test_df


def create_sized_datasets(
    input_file: str = 'questions.csv',
    train_size: float = 0.5,
    val_size: float = 0.3,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Create small and medium sized datasets with specified number of entries.
    Small: 5000/3000/2000 entries (train/val/test)
    Medium: 50000/30000/20000 entries (train/val/test)

    Args:
        input_file: Path to input CSV file (default: questions.csv)
        train_size: Proportion of data for training (default: 0.5)
        val_size: Proportion of data for validation (default: 0.3)
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    """
    # Validate split proportions
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split proportions must sum to 1.0, got {total}")

    # Determine file paths
    if not os.path.isabs(input_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(script_dir, input_file)

    print(f"Reading dataset from: {input_file}")

    # Read the dataset
    df = pd.read_csv(input_file, escapechar='\\', sep=',')
    total_rows = len(df)

    print(f"Total rows in original dataset: {total_rows}")
    print(f"Columns: {list(df.columns)}")

    # Define dataset sizes
    datasets = {
        'small': {'train': 5000, 'val': 3000, 'test': 2000},
        'medium': {'train': 50000, 'val': 30000, 'test': 20000}
    }

    for dataset_name, sizes in datasets.items():
        total_needed = sizes['train'] + sizes['val'] + sizes['test']

        print(f"\n{'='*60}")
        print(f"Creating {dataset_name} dataset")
        print(f"{'='*60}")

        if total_needed > total_rows:
            print(f"Warning: Requested {total_needed} rows but dataset only has {total_rows} rows")
            print(f"Skipping {dataset_name} dataset")
            continue

        # Create output directory
        output_dir = os.path.join(os.path.dirname(input_file), dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Output directory: {output_dir}")
        print(f"Target sizes: Train={sizes['train']}, Val={sizes['val']}, Test={sizes['test']}")

        # Sample the required total number of rows
        sampled_df = df.sample(n=total_needed, random_state=random_state)

        # Split the sampled data
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            sampled_df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        # Second split: separate train and validation
        val_size_adjusted = val_size / (train_size + val_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            shuffle=True
        )

        print(f"\nActual split:")
        print(f"  Train: {len(train_df)} rows")
        print(f"  Val:   {len(val_df)} rows")
        print(f"  Test:  {len(test_df)} rows")

        # Save split datasets
        train_path = os.path.join(output_dir, 'questions_train.csv')
        val_path = os.path.join(output_dir, 'questions_val.csv')
        test_path = os.path.join(output_dir, 'questions_test.csv')

        print(f"\nSaving splits...")
        train_df.to_csv(train_path, index=False)
        print(f"  Train saved to: {train_path}")

        val_df.to_csv(val_path, index=False)
        print(f"  Val saved to:   {val_path}")

        test_df.to_csv(test_path, index=False)
        print(f"  Test saved to:  {test_path}")

        # Print duplicate distribution
        print(f"\nDuplicate distribution:")
        print(f"  Train:")
        print(f"    Duplicates: {train_df['is_duplicate'].sum()} ({train_df['is_duplicate'].sum()/len(train_df)*100:.1f}%)")
        print(f"    Non-duplicates: {(~train_df['is_duplicate'].astype(bool)).sum()} ({(~train_df['is_duplicate'].astype(bool)).sum()/len(train_df)*100:.1f}%)")

        print(f"  Val:")
        print(f"    Duplicates: {val_df['is_duplicate'].sum()} ({val_df['is_duplicate'].sum()/len(val_df)*100:.1f}%)")
        print(f"    Non-duplicates: {(~val_df['is_duplicate'].astype(bool)).sum()} ({(~val_df['is_duplicate'].astype(bool)).sum()/len(val_df)*100:.1f}%)")

        print(f"  Test:")
        print(f"    Duplicates: {test_df['is_duplicate'].sum()} ({test_df['is_duplicate'].sum()/len(test_df)*100:.1f}%)")
        print(f"    Non-duplicates: {(~test_df['is_duplicate'].astype(bool)).sum()} ({(~test_df['is_duplicate'].astype(bool)).sum()/len(test_df)*100:.1f}%)")

        print(f"\n✓ {dataset_name.capitalize()} dataset created successfully!")

    print(f"\n{'='*60}")
    print("All datasets created successfully!")
    print(f"{'='*60}")


def main():
    """Main function to create sized datasets."""
    # Create small (5K/3K/2K) and medium (50K/30K/20K) datasets
    # with 50-30-20 train/val/test split
    create_sized_datasets(
        input_file='questions.csv',
        train_size=0.5,
        val_size=0.3,
        test_size=0.2,
        random_state=42
    )


if __name__ == "__main__":
    main()
