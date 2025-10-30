# scripts/train.py (snippet)
import json, argparse, pathlib
# ... your training code, compute metrics such as rmse, r2, accuracy ...
p = argparse.ArgumentParser()
p.add_argument("--data")
p.add_argument("--n_estimators", type=int)
p.add_argument("--max_depth", type=int)
p.add_argument("--out")
p.add_argument("--metrics")  # NEW
args = p.parse_args()

# after training & evaluation
metrics = {"rmse": float(rmse), "r2": float(r2), "accuracy": float(acc)}

pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
with open(args.metrics, "w") as f:
    json.dump(metrics, f, indent=2)
