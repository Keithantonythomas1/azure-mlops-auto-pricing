import argparse, os, sys, json
from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

def eprint(*a): print(*a, file=sys.stderr, flush=True)

def parse_args():
    p = argparse.ArgumentParser()
    # New-style flags (component- and direct-job friendly)
    p.add_argument("--training_data", type=str)
    p.add_argument("--target_column", type=str)
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=10)
    p.add_argument("--model_dir", type=str)
    p.add_argument("--metrics", type=str)  # folder

    # Old-style flags (your original script & YAML variants)
    p.add_argument("--data", type=str)
    p.add_argument("--target", type=str)
    p.add_argument("--output", type=str)
    p.add_argument("--metrics_out", type=str)

    args = p.parse_args()

    # Normalize: prefer new flags, fall back to old flags if new missing
    data_path   = args.training_data or args.data
    target_col  = args.target_column or args.target
    model_dir   = args.model_dir or args.output
    metrics_dir = args.metrics or args.metrics_out

    if not data_path or not target_col or not model_dir or not metrics_dir:
        p.error("Missing required arguments. "
                "Need training_data/data, target_column/target, model_dir/output, metrics/metrics_out.")

    return data_path, target_col, args.n_estimators, args.max_depth, model_dir, metrics_dir

def main():
    data_path, target_col, n_estimators, max_depth, model_dir, metrics_dir = parse_args()

    print("==== TrainRandomForest (robust) ====")
    print("Working dir:", os.getcwd())
    print("Data path:", data_path)
    print("Target:", target_col)
    print("n_estimators:", n_estimators, "max_depth:", max_depth)

    model_dir = Path(model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(metrics_dir); metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Detect column types
    num_cols = list(X.select_dtypes(include=["number"]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]

    print(f"Detected {len(num_cols)} numeric and {len(cat_cols)} categorical feature columns.")

    # Preprocess: impute + one-hot for categoricals, impute for numerics
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,  # allow sparse from OHE
    )

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None if (max_depth is None or max_depth <= 0) else max_depth,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocess", pre), ("rf", rf)])
    pipe.fit(X, y)

    # Save model
    model_path = model_dir / "model.joblib"
    dump(pipe, model_path)
    print("Model saved:", model_path)

    # Metrics on train set
    preds = pipe.predict(X)
    mse = float(mean_squared_error(y, preds))
    r2  = float(r2_score(y, preds))

    metrics = {
        "rows": int(len(X)),
        "n_numeric_features_in": int(len(num_cols)),
        "n_categorical_features_in": int(len(cat_cols)),
        "n_estimators": int(n_estimators),
        "max_depth": None if (max_depth is None or max_depth <= 0) else int(max_depth),
        "mse_train": mse,
        "r2_train": r2,
    }
    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved:", metrics_dir / "metrics.json")
    print("==== Done ====")

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        eprint("FATAL:", repr(ex))
        raise
