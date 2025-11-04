import argparse
import json
import os
import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--output", type=str, required=True)   # model_dir
    parser.add_argument("--metrics", type=str, required=True)  # metrics dir
    args = parser.parse_args()

    # Load CSV from the AzureML data asset
    df = pd.read_csv(args.data)
    y = df[args.target].values
    X = df.drop(columns=[args.target]).select_dtypes(include=[np.number]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    # Save model
    model_dir = pathlib.Path(args.output)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save metrics
    metrics_dir = pathlib.Path(args.metrics)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump({"mae": mae, "r2": r2}, f)

    print(f"Saved model to: {model_dir}")
    print(f"Saved metrics to: {metrics_dir} (MAE={mae:.4f}, R2={r2:.4f})")


if __name__ == "__main__":
    main()
