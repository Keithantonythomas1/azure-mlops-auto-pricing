#!/usr/bin/env python3
"""
Hyperparameter tuning step for the Auto Pricing pipeline.

- Uses Azure ML job context for MLflow (no set_experiment needed)
- Runs a small Ridge regression grid search
- Logs params/metrics to MLflow tracking in the AML workspace
- Saves the best model to --output_dir/best_model.pkl for downstream steps
"""

import argparse
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# ---- Azure ML / MLflow tracking (Azure-ML compatible) ----
from azureml.core import Run
import mlflow

run = Run.get_context()  # works in AML job, falls back to offline in local runs
# Point MLflow to the AML workspace tracking server
mlflow.set_tracking_uri(run.experiment.workspace.get_mlflow_tracking_uri())
# Start an MLflow run tied to this AML job
mlflow.start_run(run_id=run.id)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing prepped.csv from the prep step")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write best model artifacts")
    return p.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_dir) / "prepped.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find input data at: {data_path}")

    print(f"[tune] Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    if "price" not in df.columns:
        raise ValueError("Expected target column 'price' not found in prepped.csv")

    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----- Define model & search space -----
    model = Ridge()
    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="r2",
        cv=5,
        n_jobs=-1
    )

    print("[tune] Starting grid search...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # ----- Evaluate -----
    y_pred = best_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    print(f"[tune] Best params: {search.best_params_}")
    print(f"[tune] RMSE: {rmse:.4f}, R2: {r2:.4f}")

    # ----- Log to MLflow (tracking only; not using registry) -----
    mlflow.log_param("best_alpha", search.best_params_["alpha"])
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # This logs the model to the run’s artifact store (tracking),
    # not to the model registry (we are NOT passing registered_model_name).
    mlflow.sklearn.log_model(best_model, artifact_path="model")

    # ----- Save for downstream pipeline step -----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"[tune] Saved best model to: {model_path}")

    mlflow.end_run()
    print("[tune] Completed successfully.")


if __name__ == "__main__":
    main()
