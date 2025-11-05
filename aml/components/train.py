import argparse, os, sys, json, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

def eprint(*a): print(*a, file=sys.stderr, flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--n_estimators", type=int, required=True)
    p.add_argument("--max_depth", type=int, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--metrics_out", type=str, required=True)
    args = p.parse_args()

    # Log environment + args
    print("==== TrainRandomForest starting ====")
    print(f"Python: {sys.version}")
    print(f"Working dir: {os.getcwd()}")
    print(f"Args: {args}")

    # Make sure output folders exist
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).mkdir(parents=True, exist_ok=True)

    # The input is a uri_file mounted by AML; verify it exists
    data_path = Path(args.data)
    print(f"Checking data path: {data_path.resolve()}")
    if not data_path.exists():
        # Sometimes AML passes the mount folder when ForceFolder=False; if so, try to find the only csv in there
        parent = data_path.parent
        print(f"Data path not found. Listing parent: {parent}")
        for pth in parent.glob("**/*"):
            print(" -", pth)
        raise FileNotFoundError(f"Input CSV not found at: {data_path}")

    # Read CSV
    print("Reading CSV…")
    df = pd.read_csv(data_path)
    print(f"Loaded shape: {df.shape}")
    print("Columns:", list(df.columns))

    target_col = args.target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in CSV columns: {list(df.columns)}")

    # Separate target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Coerce numeric features; drop columns that cannot be coerced
    print("Coercing feature columns to numeric (non-parsable → NaN)…")
    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    # Drop columns that are all NaN after coercion
    all_nan_cols = [c for c in X_numeric.columns if X_numeric[c].isna().all()]
    if all_nan_cols:
        print("Dropping non-numeric columns:", all_nan_cols)
        X_numeric = X_numeric.drop(columns=all_nan_cols)

    # Drop rows with NaNs to keep training simple
    before = len(X_numeric)
    X_numeric = X_numeric.dropna()
    y = y.loc[X_numeric.index]
    print(f"Dropped {before - len(X_numeric)} rows with NaNs. Final shape: {X_numeric.shape}")

    if X_numeric.empty:
        raise ValueError("No usable features after numeric coercion. Check your CSV schema.")

    # Train model
    print("Training RandomForest…")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth <= 0 else args.max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_numeric, y)

    # Save model
    model_path = Path(args.output) / "model.joblib"
    dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Simple metrics (train set)
    preds = model.predict(X_numeric)
    metrics = {
        "rows": int(len(X_numeric)),
        "features": list(X_numeric.columns),
        "n_features": int(X_numeric.shape[1]),
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "mse": float(mean_squared_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }
    metrics_path = Path(args.metrics_out) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    print("==== TrainRandomForest finished ====")

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        eprint("FATAL:", repr(ex))
        raise
