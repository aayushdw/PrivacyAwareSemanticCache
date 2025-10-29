"""
Generate synthetic dataset of semantically similar statements using Gemini 2.5 Flash.
Reads from data/statements.csv and outputs to data/synthetic_dataset.csv
Uses batching to optimize API calls.
"""

import os
import csv
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Batch size for processing
BATCH_SIZE = 10

def generate_semantic_variants_batch(statements: List[str]) -> List[str]:
    """
    Generate semantically similar statements for a batch using Gemini 2.5 Flash.

    Args:
        statements: List of original statements

    Returns:
        List of semantically similar statements in the same order
    """
    # Create a batch prompt with all statements
    prompt = """Generate semantically equivalent statements that convey the same meaning as the following statements, but using different words and phrasing.

For each statement below, provide ONLY the rephrased version without any numbering, explanation, or additional text. Separate each rephrased statement with a newline.

"""

    for i, statement in enumerate(statements, 1):
        prompt += f"{i}. {statement}\n"

    prompt += "\nProvide the rephrased statements in order, one per line:"

    try:
        response = model.generate_content(prompt)
        variants = [line.strip() for line in response.text.strip().split('\n') if line.strip()]

        # Remove any numbering that might have been added
        cleaned_variants = []
        for variant in variants:
            # Remove leading numbers and dots/parentheses
            cleaned = variant.lstrip('0123456789.)- \t')
            cleaned_variants.append(cleaned)

        # Ensure we have the same number of variants as statements
        if len(cleaned_variants) != len(statements):
            print(f"Warning: Expected {len(statements)} variants but got {len(cleaned_variants)}")
            # Pad with empty strings if needed
            while len(cleaned_variants) < len(statements):
                cleaned_variants.append("")

        return cleaned_variants[:len(statements)]

    except Exception as e:
        print(f"Error generating variants for batch: {e}")
        return [""] * len(statements)

def main():
    input_file = 'data/statements.csv'
    output_file = 'data/synthetic_dataset.csv'

    # Read original statements
    print(f"Reading statements from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        statements = [line.strip() for line in f if line.strip()]

    print(f"Found {len(statements)} statements")
    print(f"Processing in batches of {BATCH_SIZE}...")

    # Generate synthetic dataset using batching
    results = []
    total_batches = (len(statements) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(statements), BATCH_SIZE):
        batch_num = batch_idx // BATCH_SIZE + 1
        batch_statements = statements[batch_idx:batch_idx + BATCH_SIZE]

        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_statements)} statements)...")
        for stmt in batch_statements:
            print(f"  - {stmt[:60]}...")

        # Generate variants for the entire batch
        semantic_variants = generate_semantic_variants_batch(batch_statements)

        # Store results
        for statement, variant in zip(batch_statements, semantic_variants):
            if variant:
                results.append({
                    'statement': statement,
                    'semantically_same_statement': variant
                })
                print(f"  ✓ Generated: {variant[:60]}...")
            else:
                print(f"  ✗ Failed to generate variant for: {statement[:60]}...")

    # Write results to CSV
    print(f"\nWriting {len(results)} pairs to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['statement', 'semantically_same_statement'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Done! Synthetic dataset saved to {output_file}")
    print(f"Successfully generated {len(results)}/{len(statements)} statement pairs")

if __name__ == "__main__":
    main()
