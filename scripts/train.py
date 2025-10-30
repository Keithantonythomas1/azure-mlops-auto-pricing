# scripts/train.py  (FULL REPLACEMENT)

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Optional MLflow/AzureML context
try:
    from azureml.core import Run
    import mlflow
    HAS_AZUREML = True
except ImportError:
    HAS_AZUREML = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_dir) / "prepped.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing expected file: {data_path}")

    df = pd.read_csv(data_path)
    if "price" not in df.columns:
        raise ValueError("Expected target column 'price' missing from data.")

    X = df.drop(columns=["price"])
    y = df["price"]

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = float(r2_score(y, preds))
    metrics = {"rmse": rmse, "r2": r2}

    print(f"[train] RMSE={rmse:.4f}, R2={r2:.4f}")

    # MLflow integration (safe for AzureML jobs)
    if HAS_AZUREML:
        run = Run.get_context()
        mlflow.set_tracking_uri(run.experiment.workspace.get_mlflow_tracking_uri())
        mlflow.start_run(run_id=run.id)
        mlflow.log_params({"n_estimators": args.n_estimators, "max_depth": args.max_depth})
        mlflow.log_metrics(metrics)
        mlflow.end_run()

    Path(args.train_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(args.train_dir) / "model.pkl")

    metrics_path = Path(args.metrics)
    if metrics_path.is_dir():
        metrics_path = metrics_path / "metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] Model and metrics written to {args.train_dir}")


if __name__ == "__main__":
    main()
