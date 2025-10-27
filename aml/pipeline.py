# aml/pipeline.py
import os
import json
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, dsl, command
from azure.ai.ml.entities import Environment


# ==========================================================
# 1️⃣ Load AML Configuration
# ==========================================================
def load_cfg() -> dict:
    """
    Loads AML workspace configuration from configs/aml_config.json.
    Expected keys:
      {
        "subscription_id": "...",
        "resource_group": "...",
        "workspace_name": "...",
        "compute": "Keith-Compute"
      }
    """
    cfg_path = os.environ.get("AML_CONFIG_PATH", "configs/aml_config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)


cfg = load_cfg()


# ==========================================================
# 2️⃣ Connect to Azure ML Workspace
# ==========================================================
cred = DefaultAzureCredential()
ml_client = MLClient(
    credential=cred,
    subscription_id=cfg["subscription_id"],
    resource_group_name=cfg["resource_group"],
    workspace_name=cfg["workspace_name"],
)

# ==========================================================
# 3️⃣ Define Environment
# ==========================================================
env = Environment(
    name="auto-pricing-env",
    conda_file="env/conda.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

compute_target = cfg.get("compute", "Keith-Compute")


# ==========================================================
# 4️⃣ Define Components
# ==========================================================

# --- Data Prep Component ---
prep_component = command(
    name="prep",
    display_name="Data Preparation",
    environment=env,
    code=".",  # directory containing scripts/
    command="python scripts/prep.py",
    compute=compute_target,
    inputs={"data_file": Input(type="uri_file", path="azureml:UsedCars:1")},
    environment_variables={"DATA_PATH": "${{inputs.data_file}}"},
    outputs={"prep_outputs": Output(type="uri_folder")},
)

# --- Training Component ---
train_component = command(
    name="train",
    display_name="Train Model",
    environment=env,
    code=".",
    command="python scripts/train.py",
    compute=compute_target,
    inputs={
        "INPUT_DIR": Input(type="uri_folder"),
        "N_ESTIMATORS": 200,
        "MAX_DEPTH": 12,
    },
    outputs={"train_outputs": Output(type="uri_folder")},
)

# --- Hyperparameter Tuning Component ---
tune_component = command(
    name="tune",
    display_name="Hyperparameter Tuning",
    environment=env,
    code=".",
    command="python scripts/tune.py",
    compute=compute_target,
    inputs={"INPUT_DIR": Input(type="uri_folder")},
    outputs={"tune_outputs": Output(type="uri_folder")},
)

# --- Model Registration Component ---
register_component = command(
    name="register",
    display_name="Register Model",
    environment=env,
    code=".",
    command="python scripts/register_model.py",
    compute=compute_target,
    inputs={"MODEL_PATH": Input(type="uri_file")},
)


# ==========================================================
# 5️⃣ Build the Pipeline (DSL)
# ==========================================================
@dsl.pipeline(compute=compute_target, description="Auto Pricing MLOps Pipeline")
def auto_pricing_pipeline():
    # Step 1: Prep
    prep = prep_component()

    # Step 2: Train (input = prep output)
    train = train_component(
        INPUT_DIR=Input(path=prep.outputs.prep_outputs, type="uri_folder"),
        N_ESTIMATORS=200,
        MAX_DEPTH=12,
    )

    # Step 3: Tune (input = prep output)
    tune = tune_component(
        INPUT_DIR=Input(path=prep.outputs.prep_outputs, type="uri_folder")
    )

    # Step 4: Register (input = tune output file)
    reg = register_component(
        MODEL_PATH=Input(
            type="uri_file",
            # This expression resolves *at runtime* inside the pipeline
            path="${{jobs.tune.outputs.tune_outputs}}/best_model.pkl",
        )
    )

    # Optional outputs (for visibility in AML Studio)
    return {
        "prep_outputs": prep.outputs.prep_outputs,
        "train_outputs": train.outputs.train_outputs,
        "tune_outputs": tune.outputs.tune_outputs,
    }


# ==========================================================
# 6️⃣ Submit the Pipeline
# ==========================================================
if __name__ == "__main__":
    print("Submitting Auto Pricing pipeline to Azure ML ...")
    pipeline_job = auto_pricing_pipeline()
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name="auto-pricing-e2e"
    )

    print(f"✅ Submitted pipeline job: {pipeline_job.name}")

    # Stream logs if running in CI
    try:
        from azure.ai.ml._azure_environ import _is_in_ci

        if _is_in_ci():
            ml_client.jobs.stream(pipeline_job.name)
    except Exception:
        pass
