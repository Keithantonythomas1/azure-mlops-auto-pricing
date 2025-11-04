import argparse, json, pathlib, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=10)
    p.add_argument("--output", required=True)   # model_dir
    p.add_argument("--metrics", required=True)  # metrics dir
    args = p.parse_args()

    df = pd.read_csv(args.data)
    y = df[args.target].values
    X = df.drop(columns=[args.target]).select_dtypes(include=[np.number]).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=args.n_estimators,
                                  max_depth=args.max_depth,
                                  random_state=42)
    model.fit(Xtr, ytr)

    preds = model.predict(Xte)
    mae = float(mean_absolute_error(yte, preds))
    r2 = float(r2_score(yte, preds))

    model_dir = pathlib.Path(args.output); model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    metrics_dir = pathlib.Path(args.metrics); metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump({"mae": mae, "r2": r2}, f)

    print(f"Saved model -> {model_dir}")
    print(f"Saved metrics -> {metrics_dir}  (MAE={mae:.4f}, R2={r2:.4f})")

if __name__ == "__main__":
    main()
