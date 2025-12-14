import pandas as pd
import os
import numpy as np

def split_dataset(input_path, output_dir, seed=42):
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Define split sizes
    total_len = len(df)
    train_size = int(0.3 * total_len)
    val_size = int(0.2 * total_len)
    test_size = total_len - train_size - val_size
    
    # Split data
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    # Create mini dataset (5% of original)
    mini_frac = 0.05
    mini_train_df = train_df.sample(frac=mini_frac, random_state=seed)
    mini_val_df = val_df.sample(frac=mini_frac, random_state=seed)
    mini_test_df = test_df.sample(frac=mini_frac, random_state=seed)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    mini_train_df.to_csv(os.path.join(output_dir, 'mini_train.csv'), index=False)
    mini_val_df.to_csv(os.path.join(output_dir, 'mini_val.csv'), index=False)
    mini_test_df.to_csv(os.path.join(output_dir, 'mini_test.csv'), index=False)
    
    print(f"Data splitting complete.")
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Mini Train size: {len(mini_train_df)}")

if __name__ == '__main__':
    split_dataset('data/generated_data/triplet_dataset.csv', 'data/processed')
