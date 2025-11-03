# scripts/train.py
import argparse, os, json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data_path", type=str, required=True)   # single file path (uri_file)
  p.add_argument("--train_dir", type=str, required=True)
  p.add_argument("--metrics",   type=str, required=True)
  p.add_argument("--n_estimators", type=int, default=100)
  p.add_argument("--max_depth",    type=int, default=None)
  args = p.parse_args()

  data_path = Path(args.data_path)
  if not data_path.exists():
    raise FileNotFoundError(f"Missing expected file: {data_path}")

  print(f"[train.py] Reading data from: {data_path}")
  df = pd.read_csv(data_path) if data_path.suffix.lower()==".csv" else pd.read_parquet(data_path)

  # Simple heuristic: last column is target if 'price' not present
  target = "price" if "price" in df.columns else df.columns[-1]
  X, y = df.drop(columns=[target]), df[target]

  Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
  model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
  model.fit(Xtr, ytr)
  preds = model.predict(Xte)
  rmse = mean_squared_error(yte, preds, squared=False)

  Path(args.train_dir).mkdir(parents=True, exist_ok=True)
  Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)

  model_path = Path(args.train_dir) / "model.joblib"
  joblib.dump(model, model_path)
  with open(args.metrics, "w") as f:
    json.dump({"rmse": rmse}, f)

  print(f"[train.py] Saved model -> {model_path}")
  print(f"[train.py] Saved metrics -> {args.metrics} (rmse={rmse:.4f})")

if __name__ == "__main__":
  main()
