import argparse
import os
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import dump

def find_first_csv(folder: Path) -> Path | None:
    for p in folder.rglob("*.csv"):
        return p
    return None

def load_table(data_path: str) -> pd.DataFrame:
    """
    Accepts either:
      - path to an MLTable folder (with CSVs inside), or
      - a path directly to a CSV file.
    """
    p = Path(data_path)
    if p.is_dir():
        # Try to find a CSV under the MLTable folder
        csv = find_first_csv(p)
        if csv is None:
            raise FileNotFoundError(f"No CSV found under {p}")
        return pd.read_csv(csv)
    elif p.is_file():
        return pd.read_csv(p)
    else:
        raise FileNotFoundError(f"{data_path} not found")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()

    df = load_table(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in data. Columns: {list(df.columns)}")

    X = df.drop(columns=[args.target]).select_dtypes(include=["number"]).fillna(0)
    y = df[args.target].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.joblib")
    dump(model, model_path)

    metrics = {"r2": r2, "mae": mae, "n_estimators": args.n_estimators, "max_depth": args.max_depth}
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to: {model_path}")
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()
