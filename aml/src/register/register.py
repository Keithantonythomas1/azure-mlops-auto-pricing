import argparse, os, subprocess, sys
from pathlib import Path

# Ensure SDK is available in the curated env
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "azure-ai-ml>=1.15.0", "azure-identity>=1.16.0"])

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

def get_ml_client() -> MLClient:
    sub = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    rg  = os.environ.get("AZUREML_ARM_RESOURCEGROUP")
    ws  = os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
    if not (sub and rg and ws):
        raise RuntimeError("Missing AML workspace env vars.")
    cred = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    return MLClient(credential=cred, subscription_id=sub, resource_group_name=rg, workspace_name=ws)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--model_name", required=True)
    args = p.parse_args()

    client = get_ml_client()
    model_path = Path(args.model_dir).as_posix()

    model = Model(
        name=args.model_name,
        path=model_path,
        type="custom_model",
        description="Auto Pricing model (RandomForest)",
        tags={"source": "pipeline", "framework": "scikit-learn"},
    )
    created = client.models.create_or_update(model)
    print(f"Registered model: {created.name} v{created.version} (path={model_path})")

if __name__ == "__main__":
    main()
