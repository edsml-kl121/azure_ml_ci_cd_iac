"""
Training component.

Reads train/test CSVs produced by data_prep, fits a RandomForestClassifier,
logs accuracy and AUC via MLflow autolog, and saves the model in MLflow format.
"""
import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data",    required=True)
    parser.add_argument("--test-data",     required=True)
    parser.add_argument("--model-output",  required=True)
    args = parser.parse_args()

    mlflow.sklearn.autolog()

    train_df = pd.read_csv(os.path.join(args.train_data, "train.csv"))
    test_df  = pd.read_csv(os.path.join(args.test_data,  "test.csv"))

    feature_cols = [c for c in train_df.columns if c != "label"]
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test,  y_test  = test_df[feature_cols],  test_df["label"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc",      auc)
    print(f"Accuracy: {acc:.4f}  |  AUC: {auc:.4f}")

    os.makedirs(args.model_output, exist_ok=True)
    mlflow.sklearn.save_model(model, args.model_output)


if __name__ == "__main__":
    main()
