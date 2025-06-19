# llama2_medical/data/preprocess.py

import json
import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw_data(input_path):
    """Load your raw data file (CSV, JSON, etc.)"""
    if input_path.endswith('.csv'):
        return pd.read_csv(input_path)
    elif input_path.endswith('.jsonl'):
        return pd.read_json(input_path, lines=True)
    else:
        raise ValueError("Unsupported file format.")

def format_instruction(row):
    """Format each row for instruction tuning."""
    return {
        "instruction": row["question"],  # adapt as per your column names
        "input": "",
        "output": row["answer"]
    }

def preprocess_and_save(input_path, output_path, test_size=0.1):
    df = load_raw_data(input_path)
    processed = [format_instruction(row) for _, row in df.iterrows()]
    train, val = train_test_split(processed, test_size=test_size, random_state=42)
    with open(output_path + "/train.jsonl", "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")
    with open(output_path + "/val.jsonl", "w") as f:
        for item in val:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    preprocess_and_save(args.input_path, args.output_path)
