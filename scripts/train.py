import argparse, json, os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)              # mltable path
    p.add_argument("--target_column", required=True)
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=10)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    # MLTable path contains a single parquet/csv. Pandas reads automatically.
    df = pd.read_parquet(args.data) if any(Path(args.data).glob("**/*.parquet")) else pd.read_csv(Path(args.data) / "data.csv")
    y = df[args.target_column]
    X = df.drop(columns=[args.target_column])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.pkl")

    metrics = {"rmse": float(rmse)}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Training complete. RMSE:", rmse)

if __name__ == "__main__":
    main()
