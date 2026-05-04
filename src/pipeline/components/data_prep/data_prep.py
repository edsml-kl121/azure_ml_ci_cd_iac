"""
Data preparation component.

Loads the sklearn diabetes dataset and converts the continuous regression
target into a binary classification label:
  1 = high disease progression (above dataset median)
  0 = low disease progression

Outputs train/test CSVs to the Azure ML output folders.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--test-output", required=True)
    args = parser.parse_args()

    # Load data
    bunch = load_diabetes(as_frame=True)
    df: pd.DataFrame = bunch.frame.copy()

    # Convert regression target → binary label
    median_val = df["target"].median()
    df["label"] = (df["target"] > median_val).astype(int)
    df = df.drop(columns=["target"])

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)

    train_df.to_csv(os.path.join(args.train_output, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_output, "test.csv"), index=False)

    print(f"Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")
    print(f"Label split (train):\n{train_df['label'].value_counts()}")


if __name__ == "__main__":
    main()
