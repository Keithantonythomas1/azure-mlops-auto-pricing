import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def read_dataset(data_path: str) -> pd.DataFrame:
    """Load an MLTable/Parquet/CSV as pandas DataFrame, best-effort."""
    p = Path(data_path)

    # MLTable (folder containing MLTable file)
    mltable_file = p / "MLTable"
    if mltable_file.exists():
        # lazy import to avoid dependency issues if not present
        import mltable
        tbl = mltable.load(str(p))
        return tbl.to_pandas_dataframe()

    # Parquet folder or file
    if p.is_dir():
        parquet_files = list(p.rglob("*.parquet"))
        if parquet_files:
            return pd.read_parquet([str(f) for f in parquet_files])
        csv_files = list(p.rglob("*.csv"))
        if csv_files:
            return pd.concat([pd.read_csv(str(f)) for f in csv_files], ignore_index=True)

    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(str(p))
    if p.suffix.lower() == ".csv":
        return pd.read_csv(str(p))

    raise ValueError(f"Could not load dataset from path: {data_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--model_dir", type=str, required=True)
    args = parser.parse_args()

    df = read_dataset(args.data)
    print(f"Loaded dataset shape: {df.shape}")

    # ------- tolerant target column (case-insensitive) -------
    cols_lower = {c.lower(): c for c in df.columns}
    tc_lower = args.target_column.lower()
    if args.target_column not in df.columns:
        if tc_lower in cols_lower:
            args.target_column = cols_lower[tc_lower]
        else:
            raise ValueError(
                f"Target column '{args.target_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
    target = args.target_column

    # Split features/target
    y = df[target]
    X = df.drop(columns=[target])

    # Simple preprocessing: one-hot encode categoricals
    X = pd.get_dummies(X, drop_first=True)
    print(f"After one-hot, shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "n_features": int(X_train.shape[1]),
        "n_train_rows": int(X_train.shape[0]),
    }
    print("Metrics:", metrics)

    # Save model and metrics to output folder
    out_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save schema of training columns (for inference later)
    pd.Series(X.columns).to_csv(out_dir / "feature_columns.csv", index=False)

    print(f"Saved model to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
