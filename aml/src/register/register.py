import argparse
import os
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model


def get_ml_client() -> MLClient:
    # These are set automatically in AML jobs
    sub = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    rg  = os.environ.get("AZUREML_ARM_RESOURCEGROUP")
    ws  = os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
    if not (sub and rg and ws):
        raise RuntimeError("Missing AML workspace environment variables.")
    cred = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    return MLClient(credential=cred, subscription_id=sub, resource_group_name=rg, workspace_name=ws)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    client = get_ml_client()

    # Register as a path-based model (simple and robust)
    model_path = Path(args.model_dir).as_posix()

    model = Model(
        name=args.model_name,
        path=model_path,
        description="Auto Pricing model (RandomForest)",
        type="custom_model",
        tags={"source": "pipeline", "framework": "scikit-learn"},
    )
    created = client.models.create_or_update(model)
    print(f"Registered model: {created.name} v{created.version} (path={model_path})")


if __name__ == "__main__":
    main()
