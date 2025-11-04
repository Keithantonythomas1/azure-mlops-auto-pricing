import argparse, os, json, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

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

df = pd.read_csv(args.data)
y = df[args.target]
X = df.drop(columns=[args.target])

model = RandomForestRegressor(
    n_estimators=args.n_estimators,
    max_depth=None if args.max_depth <= 0 else args.max_depth,
    random_state=42,
)
model.fit(X, y)

dump(model, os.path.join(args.output, "model.joblib"))

metrics = {"n_estimators": args.n_estimators, "max_depth": args.max_depth, "train_rows": int(len(df))}
with open(os.path.join(args.metrics_out, "metrics.json"), "w") as f:
    json.dump(metrics, f)
