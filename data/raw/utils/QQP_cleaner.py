import pandas as pd
from difflib import SequenceMatcher
from tqdm import tqdm


def calculate_lexical_similarity(text1: str, text2: str) -> float:
    """Calculate lexical similarity between two texts using SequenceMatcher."""
    # Convert to string and handle None/NaN values
    text1_str = str(text1) if pd.notna(text1) else ""
    text2_str = str(text2) if pd.notna(text2) else ""
    return SequenceMatcher(None, text1_str.lower(), text2_str.lower()).ratio()


def remove_similar_questions(
    input_file: str,
    output_file: str,
    similarity_threshold: float = 0.95,
    lookback_window: int = 5,
    dry_run: bool = False
) -> None:
    """
    Remove questions that have lexical similarity above the threshold.
    Only checks against the previous N questions (lookback_window).

    Args:
        dry_run: If True, shows what would be removed without saving output file.
    """
    # Load the data
    df = pd.read_csv(input_file)
    mode_str = "[DRY RUN MODE]" if dry_run else ""
    print(f"{mode_str} Loaded {len(df)} questions from {input_file}")

    kept_questions = []
    removed_examples = []
    removed_count = 0

    # Iterate through each question
    for i in tqdm(range(len(df)), desc="Processing questions"):
        current_question = df.iloc[i]['question']
        is_duplicate = False
        match_info = None

        # Only check against the last N kept questions
        start_idx = max(0, len(kept_questions) - lookback_window)
        for prev_idx in range(start_idx, len(kept_questions)):
            prev_question = kept_questions[prev_idx]['question']
            similarity = calculate_lexical_similarity(current_question, prev_question)

            if similarity > similarity_threshold:
                is_duplicate = True
                removed_count += 1
                match_info = {
                    'similarity': similarity,
                    'kept': prev_question,
                    'removed': current_question,
                    'removed_idx': i
                }
                
                removed_examples.append(match_info)

                if not dry_run and removed_count <= 10:
                    print(f"\nRemoving similar question (similarity: {similarity:.3f}):")
                    print(f"  Keep:   {prev_question}")
                    print(f"  Remove: {current_question}")
                break

        if not is_duplicate:
            kept_questions.append(df.iloc[i].to_dict())

    # Create cleaned dataframe
    cleaned_df = pd.DataFrame(kept_questions)

    # Show examples in dry run mode
    if dry_run and removed_examples:
        # Sort examples by similarity (ascending) to show boundary cases first
        removed_examples_sorted = sorted(removed_examples, key=lambda x: x['similarity'])
        removed_examples_sorted = removed_examples_sorted[0:100]

        print(f"\n{'='*80}")
        print(f"BORDERLINE CASES - Questions closest to threshold (showing {len(removed_examples_sorted)}):")
        print(f"Threshold: {similarity_threshold:.2f}")
        print(f"These are the examples CLOSEST to the boundary - most important for tuning!")
        print(f"{'='*80}")
        for idx, example in enumerate(removed_examples_sorted, 1):
            print(f"\n[Example {idx}] Similarity: {example['similarity']:.4f} (exceeds threshold by {example['similarity']-similarity_threshold:.4f})")
            print(f"  KEEP:   {example['kept']}")
            print(f"  REMOVE: {example['removed']}")

    # Statistics
    print(f"\n{'='*80}")
    if dry_run:
        print("DRY RUN SUMMARY:")
    print(f"Original questions: {len(df)}")
    print(f"Questions {'would be' if dry_run else ''} removed: {removed_count}")
    print(f"Questions {'would' if dry_run else ''} remaining: {len(cleaned_df)}")
    print(f"Removal rate: {removed_count/len(df)*100:.2f}%")
    print(f"{'='*80}")

    # Save cleaned data only if not dry run
    if not dry_run:
        cleaned_df.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to {output_file}")
    else:
        print(f"\nDRY RUN: No file was created. Run with dry_run=False to save results.")


if __name__ == "__main__":
    input_file = "unique_questions.csv"
    output_file = "unique_questions_cleaned.csv"

    # Configuration - adjust these to experiment
    similarity_threshold = 0.69  # Questions above this similarity will be removed
    lookback_window = 5          # Number of previous questions to check
    dry_run = False               # Set to False to actually create the output file

    remove_similar_questions(
        input_file,
        output_file,
        similarity_threshold,
        lookback_window,
        dry_run
    )
