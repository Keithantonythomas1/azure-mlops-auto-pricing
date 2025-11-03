from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
args = parser.parse_args()

sub = os.environ.get("AZURE_SUBSCRIPTION_ID") or ""
rg  = os.environ.get("AZURE_RESOURCE_GROUP") or ""
ws  = os.environ.get("AZURE_ML_WORKSPACE") or ""

ml = MLClient(DefaultAzureCredential(), sub, rg, ws)
m = ml.models.create_or_update(
      Model(name="auto-pricing-model", path=args.model_path, type="custom_model",
            description="Auto pricing model trained via CI/CD"))
print(f"Registered: {m.name} v{m.version}")
