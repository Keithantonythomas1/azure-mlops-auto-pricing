import argparse
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

def resolve_workspace_from_env():
    # These are injected by AzureML into job containers
    sub = os.environ.get("AZUREML_ARM_SUBSCRIPTION") or os.environ.get("AZUREML_SUBSCRIPTION_ID")
    rg  = os.environ.get("AZUREML_ARM_RESOURCEGROUP") or os.environ.get("AZUREML_RESOURCE_GROUP")
    ws  = os.environ.get("AZUREML_ARM_WORKSPACE_NAME") or os.environ.get("AZUREML_WORKSPACE_NAME")
    if not all([sub, rg, ws]):
        raise RuntimeError("Could not resolve workspace from job environment variables.")
    return sub, rg, ws

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    subscription_id, resource_group, workspace = resolve_workspace_from_env()

    cred = DefaultAzureCredential()
    ml_client = MLClient(cred, subscription_id, resource_group, workspace)

    model = Model(
        name=args.model_name,
        path=args.model_dir,     # folder produced by train step
        type="custom_model",
        description="Auto pricing regressor",
        version=None             # let AML version it automatically
    )
    created = ml_client.models.create_or_update(model)
    print(f"Registered model: {created.name} v{created.version}")

if __name__ == "__main__":
    main()
