# ================================================================
# train.py — MLTable-based RandomForest training (Option B fix)
# ================================================================

import argparse
import os
import json

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# MLTable loader (ensure mltable is in your conda/pip deps)
from mltable import load as mltable_load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to MLTable (mounted dir)")
    p.add_argument("--target", type=str, required=True, help="Target column name")
    p.add_argument("--n_estimators", type=int, required=True)
    p.add_argument("--max_depth", type=int, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--metrics_out", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.metrics_out, exist_ok=True)

    # ----------------------------
    # Load dataset from MLTable
    # ----------------------------
    print(f"📂 Loading MLTable from: {args.data}")
    table = mltable_load(args.data)
    df = table.to_pandas_dataframe()
    print(f"✅ Loaded dataframe: {df.shape[0]} rows × {df.shape[1]} columns")

    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not found. Available: {df.columns.tolist()}"
        )

    y = df[args.target]
    X = df.drop(columns=[args.target])

    # Optional: handle non-numeric features safely
    # If your data is already numeric, this is a no-op.
    X = pd.get_dummies(X, drop_first=True)

    # Simple holdout to compute quick metrics (keeps your original fit behavior)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth <= 0 else args.max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    # Persist model
    model_path = os.path.join(args.output, "model.joblib")
    dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")

    # Basic metrics
    y_pred = model.predict(X_te)
    metrics = {
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "train_rows": int(len(df)),
        "features_after_encoding": int(X.shape[1]),
        "r2": float(r2_score(y_te, y_pred)),
        "mse": float(mean_squared_error(y_te, y_pred)),
    }

    metrics_path = os.path.join(args.metrics_out, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"📊 Metrics: {metrics}")
    print(f"📝 Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
