"""
Minimal train-test split script for JSONL datasets.
Splits a JSONL file into train and test sets with configurable ratio.
"""

import json
import random
import argparse
from pathlib import Path

def split_jsonl(input_file, train_ratio=0.8, seed=42, output_dir=None):
    """
    Split a JSONL file into train and test sets.
    
    Args:
        input_file (str): Path to input JSONL file
        train_ratio (float): Ratio of data for training (default: 0.8)
        seed (int): Random seed for reproducibility (default: 42)
        output_dir (str): Output directory (default: same as input file)
    
    Returns:
        tuple: (train_file_path, test_file_path)
    """
    input_path = Path(input_file)
    
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Generate output filenames
    base_name = input_path.stem
    train_file = output_dir / f"{base_name}_train.jsonl"
    test_file = output_dir / f"{base_name}_test.jsonl"
    
    # Load all data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    
    print(f"Total samples: {len(data)}")
    
    # Shuffle data with fixed seed
    random.seed(seed)
    random.shuffle(data)
    
    # Split data
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Write train set
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # Write test set
    with open(test_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✓ Train set saved to: {train_file}")
    print(f"✓ Test set saved to: {test_file}")
    
    return str(train_file), str(test_file)

def main():
    parser = argparse.ArgumentParser(description="Split JSONL dataset into train/test sets")
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("--train-ratio", type=float, default=0.8, 
                       help="Training set ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: same as input)")
    
    args = parser.parse_args()
    
    split_jsonl(
        input_file=args.input_file,
        train_ratio=args.train_ratio,
        seed=args.seed,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()