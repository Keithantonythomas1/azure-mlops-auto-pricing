import argparse
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="auto-pricing-model")
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    # In AzureML jobs, DefaultAzureCredential uses the job's managed identity
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    registered = ml_client.models.create_or_update(
        Model(
            name=args.model_name,
            path=str(model_path),
            type="custom_model",   # (alternatives: mlflow_model, triton_model)
            description="RandomForest model for auto pricing",
            tags={"pipeline": "auto_pricing_pipeline"},
        )
    )
    print(f"Registered model: {registered.name} v{registered.version}")


if __name__ == "__main__":
    main()
