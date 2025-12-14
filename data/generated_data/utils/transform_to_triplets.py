"""
Transform synthetic dataset from pair format to triplet format.

This script converts the synthetic dataset from the format:
    (id, original_question, generated_question, is_semantically_similar)
to triplet format:
    (anchor, positive, negative, source_question_id)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

# Configuration
INPUT_PATH = 'data/generated_data/synthetic_dataset_filtered.csv'
OUTPUT_PATH = 'data/generated_data/triplet_dataset.csv'


def load_dataset(input_path: str) -> pd.DataFrame:
    """Load and validate the input dataset."""
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError("Input CSV file is empty")

    # Validate required columns
    required_columns = ['id', 'original_question', 'generated_question', 'is_semantically_similar']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def extract_question_id(row_id: str) -> str:
    return row_id.split('_')[0]


def create_triplets(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Create triplets from the dataset.
    Returns:
        List of triplet dictionaries with keys:
            'anchor', 'positive', 'negative', 'source_question_id'
    """
    # Extract question_id from id column
    df['question_id'] = df['id'].apply(extract_question_id)

    # Group by question_id
    grouped = df.groupby('question_id')

    triplets = []

    for question_id, group in tqdm(grouped, desc="Processing questions"):
        # Get the anchor (original question)
        anchor = group['original_question'].iloc[0]

        # Get first True example (semantically similar)
        true_examples = group[group['is_semantically_similar'] == True]
        if len(true_examples) == 0:
            continue
        positive = true_examples['generated_question'].iloc[0]

        # Get first False example (semantically different)
        false_examples = group[group['is_semantically_similar'] == False]
        if len(false_examples) == 0:
            continue
        negative = false_examples['generated_question'].iloc[0]

        # Create triplet
        triplets.append({
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'source_question_id': question_id
        })

    return triplets


def validate_triplets(triplets: List[Dict[str, str]]) -> Tuple[bool, List[str]]:
    """
    Validate the created triplets.

    Args:
        triplets: List of triplet dictionaries

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    required_fields = ['anchor', 'positive', 'negative', 'source_question_id']

    if len(triplets) == 0:
        errors.append("No triplets created")
        return False, errors

    # Check each triplet has required fields and no empty values
    for i, triplet in enumerate(triplets):
        # Check required fields present
        missing_fields = set(required_fields) - set(triplet.keys())
        if missing_fields:
            errors.append(f"Triplet {i} missing fields: {missing_fields}")
            continue

        # Check no empty values
        for field in required_fields:
            value = triplet.get(field)
            if value is None or (isinstance(value, str) and value.strip() == ''):
                errors.append(f"Triplet {i} has empty value for field '{field}'")

    # Limit error messages to first 10
    if len(errors) > 10:
        errors = errors[:10] + [f"... and {len(errors) - 10} more errors"]

    is_valid = len(errors) == 0
    return is_valid, errors


def save_triplets(triplets: List[Dict[str, str]], output_path: str) -> None:
    """
    Save triplets to CSV file.

    Args:
        triplets: List of triplet dictionaries
        output_path: Path to output CSV file
    """
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame and save
    df = pd.DataFrame(triplets)
    df.to_csv(output_path, index=False, encoding='utf-8')


def main() -> int:
    """Main entry point for the script."""
    try:
        # Load dataset
        df = load_dataset(INPUT_PATH)

        # Create triplets
        triplets = create_triplets(df)

        # Validate triplets
        is_valid, errors = validate_triplets(triplets)
        if not is_valid:
            print("Validation failed:")
            for error in errors:
                print(f"  - {error}")
            return 1

        # Save triplets
        save_triplets(triplets, OUTPUT_PATH)

        # Print summary
        print("\n" + "=" * 60)
        print("TRANSFORMATION SUMMARY")
        print("=" * 60)
        print(f"Input file:        {INPUT_PATH}")
        print(f"Output file:       {OUTPUT_PATH}")
        print(f"Triplets created:  {len(triplets)}")
        print(f"Columns:           anchor, positive, negative, source_question_id")
        print("=" * 60)


    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    sys.exit(main())
