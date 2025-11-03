# src/scoring/score.py
import json, os, joblib
from pathlib import Path

def init():
    global model
    model_path = Path(os.getenv("AZUREML_MODEL_DIR", ".")) / "model.joblib"
    model = joblib.load(model_path)

def run(raw):
    body = json.loads(raw)
    # Expect {"inputs":[{"features":[...]}]}
    feats = [row["features"] for row in body["inputs"]]
    preds = model.predict(feats)
    return {"predictions": [float(p) for p in preds]}
