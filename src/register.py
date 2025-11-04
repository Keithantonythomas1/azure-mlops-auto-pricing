import argparse
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_name", required=True, default="auto-pricing-model")
    args = parser.parse_args()

    # Workspace auth via managed identity / OIDC (DefaultAzureCredential)
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    model_path = os.path.join(args.model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Expected model file not found at {model_path}")

    model = Model(
        name=args.model_name,
        path=args.model_dir,      # folder with model files
        type="mlflow_model",      # or "custom_model" if you prefer; folder is fine
        description="Auto pricing RandomForest model",
        tags={"registered-by": "aml-pipeline"}
    )
    model = ml_client.models.create_or_update(model)
    print(f"Registered model: {model.name} v{model.version}")

if __name__ == "__main__":
    main()
