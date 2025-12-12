import pandas as pd
from typing import Tuple, List

def load_data(file_path: str) -> Tuple[List[str], List[str], List[int]]:
    """
    Loads a CSV and extracts q1, q2, and labels.
    Assumes columns: 'question1', 'question2', 'is_duplicate'
    """
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    # Basic data validation
    required_cols = {'question1', 'question2', 'is_duplicate'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset missing required columns. Found: {df.columns}")

    return (
        df['question1'].tolist(),
        df['question2'].tolist(),
        df['is_duplicate'].tolist()
    )