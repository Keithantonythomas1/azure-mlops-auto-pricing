import argparse, json, os, sys, joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)              # uri_file path to csv (mounted)
    p.add_argument("--target", required=True)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=10)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--metrics", required=True)
    args = p.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.metrics, exist_ok=True)

    print(f"Reading data from: {args.data}")
    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in columns: {list(df.columns)}")

    # very simple feature split: use all numeric columns except target
    X = df.select_dtypes(include="number").drop(columns=[args.target], errors="ignore")
    y = df[args.target]

    if X.empty:
        raise ValueError("No numeric features found to train on.")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    mae = float(mean_absolute_error(yte, pred))
    r2  = float(r2_score(yte, pred))

    # save model
    model_path = os.path.join(args.model_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")

    # save metrics
    metrics_path = os.path.join(args.metrics, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"mae": mae, "r2": r2}, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")
    print({"mae": mae, "r2": r2})

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        raise
