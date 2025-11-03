import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model
from pathlib import Path
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--model_name", required=True)
    args = p.parse_args()

    cred = DefaultAzureCredential()
    ml_client = MLClient(
        credential=cred,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )

    path = Path(args.model_dir)
    mdl = Model(
        name=args.model_name,
        path=str(path),
        description="Auto-pricing RandomForest (trained via GitHub Actions)",
        type="custom_model",
    )
    mdl = ml_client.models.create_or_update(mdl)
    print("Registered model:", mdl.name, mdl.version)

if __name__ == "__main__":
    main()
