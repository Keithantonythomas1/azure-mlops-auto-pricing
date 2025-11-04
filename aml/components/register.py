import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model

p = argparse.ArgumentParser()
p.add_argument("--model_dir", type=str, required=True)
p.add_argument("--model_name", type=str, required=True)
args = p.parse_args()

# Use workspace defaults set by the workflow (RG/WS/Region)
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

model = Model(
    name=args.model_name,
    path=args.model_dir,
    type="custom_model",
    description="Auto pricing model",
)
result = ml_client.models.create_or_update(model)
print(f"Registered model: {result.name}:{result.version}")
