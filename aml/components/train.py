import argparse, os, json
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def load_csv_from_uri(path: str) -> pd.DataFrame:
    # If a direct CSV file is mounted
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return pd.read_csv(path)

    # If a directory is mounted, find a csv inside
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".csv"):
                    return pd.read_csv(os.path.join(root, f))

    raise FileNotFoundError(f"Could not locate a CSV at or under: {path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--n_estimators", type=int, required=True)
    p.add_argument("--max_depth", type=int, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--metrics_out", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.metrics_out, exist_ok=True)

    df = load_csv_from_uri(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not in columns: {df.columns.tolist()}")

    y = df[args.target]
    X = df.drop(columns=[args.target])

    # Handle non-numeric columns safely
    X = pd.get_dummies(X, drop_first=True)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth <= 0 else args.max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr)

    dump(model, os.path.join(args.output, "model.joblib"))

    y_pred = model.predict(Xte)
    metrics = {
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "train_rows": int(len(df)),
        "features_after_encoding": int(X.shape[1]),
        "r2": float(r2_score(yte, y_pred)),
        "mse": float(mean_squared_error(yte, y_pred)),
    }
    with open(os.path.join(args.metrics_out, "metrics.json"), "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()
