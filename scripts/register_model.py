# scripts/register_model.py (snippet)
import json, argparse, pathlib
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)   # path to best_model.pkl
p.add_argument("--out", required=True)     # NEW: single-file uri output
args = p.parse_args()

mlc = MLClient(DefaultAzureCredential(),
               subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
               resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
               workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"])

# register using MLflow format or custom — example uses path model/
model_name = "auto-pricing"
registered = mlc.models.create_or_update({
    "name": model_name,
    "path": args.model,       # adjust if you package a folder
    "type": "custom_model",   # or "mlflow_model" if applicable
})

pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"name": registered.name, "version": str(registered.version)}, f, indent=2)
