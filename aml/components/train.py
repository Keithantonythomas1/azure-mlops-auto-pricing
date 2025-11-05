import argparse, os, json
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Robust CSV loader (handles file or folder mount)
def load_csv_from_uri(path: str) -> pd.DataFrame:
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".csv"):
                    return pd.read_csv(os.path.join(root, f))
    raise FileNotFoundError(f"No CSV found at or under: {path}")

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

# ✅ Load CSV
df = load_csv_from_uri(args.data)
print(f"✅ Loaded data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

if args.target not in df.columns:
    raise ValueError(f"Target column '{args.target}' not found in {df.columns.tolist()}")

# ✅ Split data
y = df[args.target]
X = df.drop(columns=[args.target])
X = pd.get_dummies(X, drop_first=True)  # handle categorical
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model
model = RandomForestRegressor(
    n_estimators=args.n_estimators,
    max_depth=None if args.max_depth <= 0 else args.max_depth,
    random_state=42,
    n_jobs=-1,
)
model.fit(Xtr, ytr)

# ✅ Save model
dump(model, os.path.join(args.output, "model.joblib"))

# ✅ Evaluate and log metrics
y_pred = model.predict(Xte)
metrics = {
    "rows": len(df),
    "features": X.shape[1],
    "r2": r2_score(yte, y_pred),
    "mse": mean_squared_error(yte, y_pred),
    "n_estimators": args.n_estimators,
    "max_depth": args.max_depth,
}
with open(os.path.join(args.metrics_out, "metrics.json"), "w") as f:
    json.dump(metrics, f)
print(f"📊 Metrics: {metrics}")
