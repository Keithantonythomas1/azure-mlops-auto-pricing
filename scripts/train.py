import argparse
import json
import os
import pathlib
import shutil
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

def _load_dataframe(input_path: str) -> pd.DataFrame:
    """
    Load a dataframe from an Azure ML job input that might be:
      - an MLTable (folder containing MLTable file)
      - a Parquet file/folder
      - a CSV file
    """
    p = pathlib.Path(input_path)

    # 1) MLTable (folder containing "MLTable" file)
    mltable_file = p / "MLTable"
    if mltable_file.exists():
        try:
            from mltable import load as mltable_load
            return mltable_load(str(p)).to_pandas_dataframe()
        except Exception as e:
            print(f"[WARN] MLTable load failed, will try other formats. Error: {e}", file=sys.stderr)

    # 2) Parquet (file or folder)
    try:
        # pandas can read a directory of parquet files if they share schema
        if p.is_dir():
            # try glob parquet under folder
            parquet_files = list(p.rglob("*.parquet"))
            if parquet_files:
                return pd.read_parquet(str(p))
        elif p.suffix.lower() in [".parquet", ".pq"]:
            return pd.read_parquet(str(p))
    except Exception as e:
        print(f"[WARN] Parquet load failed, will try CSV. Error: {e}", file=sys.stderr)

    # 3) CSV (single file or search under folder)
    try:
        if p.is_dir():
            csvs = list(p.rglob("*.csv"))
            if not csvs:
                raise FileNotFoundError("No CSV files found in input folder.")
            return pd.read_csv(csvs[0])
        else:
            return pd.read_csv(str(p))
    except Exception as e:
        raise RuntimeError(
            f"Could not load data from '{input_path}' as MLTable/Parquet/CSV. "
            f"Last error: {e}"
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Input dataset (MLTable/Parquet/CSV)")
    parser.add_argument("--target_column", type=str, default="price")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--model_dir", type=str, required=True)
    args = parser.parse_args()

    print(f"[INFO] Loading data from: {args.data}")
    df = _load_dataframe(args.data)
    print(f"[INFO] Loaded shape: {df.shape}")

    if args.target_column not in df.columns:
        raise ValueError(
            f"Target column '{args.target_column}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    y = df[args.target_column]
    X = df.drop(columns=[args.target_column])

    # Handle any non-numeric columns quickly (simple one-hot)
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print(f"[METRIC] rmse={rmse:.4f} | r2={r2:.4f}")

    # Write metrics.json (the job UI reads this)
    metrics = {"rmse": float(rmse), "r2": float(r2)}
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Persist the model artefact
    model_dir = pathlib.Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"[INFO] Saved model to: {model_path}")

    # For convenience, also drop a minimal MLmodel file, making it easy to register later
    # (not strictly required, but nice for clarity)
    (model_dir / "MLmodel").write_text("flavor: sklearn\n")

    # Copy metrics alongside model_dir as an artefact too
    shutil.copy("outputs/metrics.json", model_dir / "metrics.json")

if __name__ == "__main__":
    main()
