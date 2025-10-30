# scripts/train.py  (FULL REPLACEMENT)

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing prepped.csv from the prep step")
    p.add_argument("--train_dir", type=str, required=True,
                   help="Directory to write model artifacts")
    p.add_argument("--metrics", type=str, required=True,
                   help="File OR directory path to write metrics (pipeline output)")
    # Hyperparameters that AML pipeline may pass
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=12)
    return p.parse_args()


def ensure_metrics_path(path_str: str) -> Path:
    """
    Azure ML sometimes mounts the 'metrics' output as a directory,
    other times you might provide a file path. This makes both work.
    """
    p = Path(path_str)
    if p.exists() and p.is_dir():
        return p / "metrics.json"
    # If it's a file path (doesn't exist yet), ensure parent dir exists.
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main():
    args = parse_args()

    data_path = Path(args.data_dir) / "prepped.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Expected file not found: {data_path}")

    print(f"[train] Loading data: {data_path}")
    df = pd.read_csv(data_path)

    if "price" not in df.columns:
        raise ValueError("Expected target column 'price' in prepped.csv")

    X = df.drop(columns=["price"])
    y = df["price"]

    # ---------------- Train ----------------
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Quick in-sample metrics (or replace with a train/test split if desired)
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = float(r2_score(y, preds))
    metrics = {"rmse": rmse, "r2": r2,
               "n_estimators": int(args.n_estimators),
               "max_depth": int(args.max_depth)}

    print(f"[train] RMSE: {rmse:.4f} | R2: {r2:.4f}")

    # ---------------- Save model ----------------
    train_dir = Path(args.train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    model_path = train_dir / "model.pkl"
    joblib.dump(model, model_path)
    # Save feature columns for downstream compatibility
    (train_dir / "features.json").write_text(json.dumps(list(X.columns)))
    print(f"[train] Model saved to: {model_path}")

    # ---------------- Save metrics ----------------
    metrics_file = ensure_metrics_path(args.metrics)
    metrics_file.write_text(json.dumps(metrics, indent=2))
    print(f"[train] Metrics written to: {metrics_file}")


if __name__ == "__main__":
    main()
