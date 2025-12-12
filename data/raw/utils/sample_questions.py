import csv
import random
import os

def sample_questions(input_path, output_path, sample_size=20000):
    """
    Reads a CSV file, samples a specified number of rows, and writes to a new CSV file.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Reading from {input_path}...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
            
        print(f"Total rows found: {len(rows)}")
        
        if len(rows) < sample_size:
            print(f"Warning: efficient rows ({len(rows)}) is less than sample size ({sample_size}). Taking all rows.")
            sampled_rows = rows
        else:
            print(f"Sampling {sample_size} rows...")
            sampled_rows = random.sample(rows, sample_size)
            
        print(f"Writing to {output_path}...")
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(sampled_rows)
            
        print("Done!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Input is in the parent directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, '../unique_questions_filtered.csv')
    # Output is in the same directory as this script
    output_csv = os.path.join(script_dir, 'final_questions.csv')
    
    sample_questions(input_csv, output_csv)
