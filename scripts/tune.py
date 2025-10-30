# scripts/tune.py  (FULL REPLACEMENT)

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Defensive MLflow import for AzureML context
try:
    from azureml.core import Run
    import mlflow
    HAS_AZUREML = True
except ImportError:
    HAS_AZUREML = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--best_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_dir) / "prepped.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing expected file: {data_path}")

    df = pd.read_csv(data_path)
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [100, 200], "max_depth": [10, 12, 15]},
        cv=3, scoring="neg_mean_squared_error", n_jobs=-1, verbose=2
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    metrics = {"rmse": rmse, "r2": r2, **grid.best_params_}

    print(f"[tune] Best Params: {grid.best_params_}")
    print(f"[tune] RMSE={rmse:.4f}, R2={r2:.4f}")

    # MLflow safe-guarded logging
    if HAS_AZUREML:
        run = Run.get_context()
        mlflow.set_tracking_uri(run.experiment.workspace.get_mlflow_tracking_uri())
        mlflow.start_run(run_id=run.id)
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({"rmse": rmse, "r2": r2})
        mlflow.end_run()

    best_dir = Path(args.best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, best_dir / "best_model.pkl")

    with open(best_dir / "tuning_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[tune] Completed successfully and artifacts saved.")


if __name__ == "__main__":
    main()
