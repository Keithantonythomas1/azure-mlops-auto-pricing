import argparse, os, json, io
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def find_csv(path: str) -> str:
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return path
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".csv"):
                    return os.path.join(root, f)
    raise FileNotFoundError(f"No CSV found at or under: {path}")

def read_csv_robust(csv_path: str) -> pd.DataFrame:
    # try standard read
    try:
        return pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    except Exception as e:
        print(f"utf-8 read failed ({e}); trying latin-1")
        return pd.read_csv(csv_path, encoding="latin-1", on_bad_lines="skip")

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

print(f"📂 Input mount: {args.data}")
csv_path = find_csv(args.data)
print(f"🗂️ Using CSV: {csv_path}")

df = read_csv_robust(csv_path)
print(f"✅ Loaded shape: {df.shape}")
print(f"🧾 Columns: {list(df.columns)}")

if args.target not in df.columns:
    raise ValueError(f"Target '{args.target}' not found in columns: {df.columns.tolist()}")

# Clean up obvious bad rows
df = df.dropna(how="all")
df = df.reset_index(drop=True)

y = df[args.target]
X = df.drop(columns=[args.target])

# Coerce numerics & one-hot categorical
for c in X.columns:
    if X[c].dtype == "object":
        # try to coerce numeric-looking strings; if many NaNs result, treat as categorical
        coerced = pd.to_numeric(X[c].str.replace(",", ""), errors="coerce")
        if coerced.notna().mean() >= 0.8:
            X[c] = coerced
X = pd.get_dummies(X, drop_first=True)

# Guard against empty or tiny matrices
if X.shape[0] < 5 or X.shape[1] == 0:
    raise ValueError(f"Not enough data to train: rows={X.shape[0]}, cols={X.shape[1]}")

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=args.n_estimators,
    max_depth=None if args.max_depth <= 0 else args.max_depth,
    random_state=42,
    n_jobs=-1,
)
print("🌲 Training RandomForestRegressor …")
model.fit(Xtr, ytr)

from pathlib import Path
model_path = Path(args.output) / "model.joblib"
dump(model, model_path)
print(f"💾 Model saved: {model_path}")

y_pred = model.predict(Xte)
metrics = {
    "rows": int(len(df)),
    "features": int(X.shape[1]),
    "r2": float(r2_score(yte, y_pred)),
    "mse": float(mean_squared_error(yte, y_pred)),
    "n_estimators": int(args.n_estimators),
    "max_depth": int(args.max_depth),
}
metrics_path = Path(args.metrics_out) / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f)
print(f"📊 Metrics: {metrics}")
