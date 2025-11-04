import argparse
import os
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump

def load_dataframe(input_path: str) -> pd.DataFrame:
    """Load data from a uri_folder/MLTable/CSV. We try a few common patterns."""
    p = Path(input_path)

    # If this is an MLTable asset folder, try to locate a csv under it
    csvs = list(p.rglob("*.csv"))
    if csvs:
        return pd.read_csv(csvs[0])

    # Fallback: if the path is a file and is CSV
    if p.is_file() and p.suffix.lower() == ".csv":
        return pd.read_csv(p)

    # As a last resort, try reading a MLTable directly (if present)
    mltable_file = p / "MLTable"
    if mltable_file.exists():
        # Simple fallback for MLTable: look for CSVs anyway
        csvs = list(p.rglob("*.csv"))
        if csvs:
            return pd.read_csv(csvs[0])

    raise FileNotFoundError(
        f"Could not find a CSV inside {input_path}. "
        "If your asset is an MLTable, ensure it contains/points to one or more CSVs."
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    df = load_dataframe(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not in data columns: {df.columns.tolist()}")

    X = df.drop(columns=[args.target])
    y = df[args.target]

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)
    print(f"Train RMSE: {rmse:.4f}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(model, out_dir / "model.joblib")

if __name__ == "__main__":
    main()
