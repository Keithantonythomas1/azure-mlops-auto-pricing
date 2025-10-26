import os
import json
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def load_cfg():
    # Expect env-based substitution if running in GH Actions
    cfg_path = os.environ.get("AML_CONFIG_PATH", "configs/aml_config.json")
    with open(cfg_path, "r") as f:
        raw = f.read()
    # naive env substitution for secrets format ${{ secrets.VAR }}
    # Using concrete values from configs/aml_config.json (already set).
    return json.loads(raw)

if __name__ == "__main__":
    cfg = load_cfg()
    cred = DefaultAzureCredential()
    mlclient = MLClient(credential=cred,
                        subscription_id=cfg["subscription_id"],
                        resource_group_name=cfg["resource_group"],
                        workspace_name=cfg["workspace_name"])
    # Register from local artifact (best_model.pkl) as a generic model asset
    model_path = os.environ.get("MODEL_PATH", "outputs/best_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.environ.get("MODEL_PATH_FALLBACK", "outputs/model.pkl")
    name = os.environ.get("MODEL_NAME", "auto-pricing-rf")
    version = os.environ.get("MODEL_VERSION")  # optionally inject
    from azure.ai.ml.entities import Model
    model = Model(
        name=name,
        path=model_path,
        type="mlflow_model" if model_path.endswith("mlflow") else "custom_model",
        description="Random Forest model for vehicle price prediction (luxury & non-luxury segments).",
        tags={"project":"auto-pricing-mlops"}
    )
    registered = mlclient.models.create_or_update(model)
    print(f"Registered model: {registered.name}:{registered.version}")