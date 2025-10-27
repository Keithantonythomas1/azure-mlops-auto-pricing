# aml/pipeline.py
import os
import json

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, dsl, command
from azure.ai.ml.entities import Environment


# -----------------------------
# Load AML config from repo
# -----------------------------
def load_cfg() -> dict:
    """
    Reads configs/aml_config.json unless AML_CONFIG_PATH is set.
    The file is expected to contain:
      {
        "subscription_id": "...",
        "resource_group": "...",
        "workspace_name": "...",
        "compute": "Keith-Compute"   # optional; defaults to 'cpu-cluster'
      }
    """
    cfg_path = os.environ.get("AML_CONFIG_PATH", "configs/aml_config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)


cfg = load_cfg()

# -----------------------------
# MLClient (auth via DefaultAzureCredential)
# -----------------------------
cred = DefaultAzureCredential()
ml_client = MLClient(
    credential=cred,
    subscription_id=cfg["subscription_id"],
    resource_group_name=cfg["resource_group"],
    workspace_name=cfg["workspace_name"],
)

# -----------------------------
# Runtime environment
# -----------------------------
env = Environment(
    name="auto-pricing-env",
    conda_file="env/conda.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

compute_target = cfg.get("compute", "cpu-cluster")

# -----------------------------
# Components
# -----------------------------

# 1) Prep
prep_component = command(
    name="prep",
    display_name="Data Prep",
    environment=env,
    code=".",  # repo root
    command="python scripts/prep.py",
    compute=compute_target,
    inputs={
        # Azure ML data asset (uri_file) created in your workspace
        "data_file": Input(type="uri_file", path="azureml:UsedCars:1")
    },
    # make raw path available to the script if it uses DATA_PATH
    environment_variables={"DATA_PATH": "${{inputs.data_file}}"},
    # Let AML place outputs automatically (avoids path warnings/collisions)
    outputs={"prep_outputs": Output(type="uri_folder")},
)

# 2) Train
train_component = command(
    name="train",
    display_name="Train Baseline",
    environment=env,
    code=".",
    command="python scripts/train.py",
    compute=compute_target,
    inputs={
        # INPUT_DIR will be wired in the pipeline using Input(path=...)
        "INPUT_DIR": Input(type="uri_folder"),
        "N_ESTIMATORS": 200,
        "MAX_DEPTH": 12,
    },
    outputs={"train_outputs": Output(type="uri_folder")},
)

# 3) Tune
tune_component = command(
    name="tune",
    display_name="Hyperparameter Tuning",
    environment=env,
    code=".",
    command="python scripts/tune.py",
    compute=compute_target,
    inputs={
        # INPUT_DIR will be wired in the pipeline using Input(path=...)
        "INPUT_DIR": Input(type="uri_folder"),
    },
    outputs={
        # Complete tuning artifacts
        "tune_outputs": Output(type="uri_folder"),
        # (optional) If your tune.py writes a dedicated best model file,
        # you can expose it as a separate uri_file output instead.
        # "best_model": Output(type="uri_file"),
    },
)

# 4) Register
# NOTE:
# We keep the expression using the *parent* context so this input resolves at
# pipeline build time to the *tune* job’s output folder and file.
register_component = command(
    name="register",
    display_name="Register Model",
    environment=env,
    code=".",
    command="python scripts/register_model.py",
    compute=compute_target,
    inputs={
        # If your best model file is written as `best_model.pkl` in tune_outputs,
        # pass that concrete path using pipeline expression. Do not override this
        # in the pipeline body with a raw NodeOutput.
        "MODEL_PATH": Input(
            type="uri_file",
            path="${{parent.jobs.tune.outputs.tune_outputs}}/best_model.pkl",
        )
    },
)

# -----------------------------
# Pipeline definition (DSL)
# -----------------------------
@dsl.pipeline(compute=compute_target, description="Auto pricing MLOps pipeline")
def auto_pricing_pipeline():
    # Run prep
    prep = prep_component()

    # Feed prep -> train using Input(path=...) so AML treats it as a valid pipeline input
    train = train_component(
        INPUT_DIR=Input(path=prep.outputs.prep_outputs, type="uri_folder"),
        N_ESTIMATORS=200,
        MAX_DEPTH=12,
    )

    # Feed prep -> tune similarly
    tune = tune_component(
        INPUT_DIR=Input(path=prep.outputs.prep_outputs, type="uri_folder")
    )

    # DO NOT override MODEL_PATH with a NodeOutput here. The register component
    # already contains a pipeline expression that points at the tune output file.
    reg = register_component()

    # Optionally expose top-level pipeline outputs (useful for debugging)
    return {
        "prep_outputs": prep.outputs.prep_outputs,
        "train_outputs": train.outputs.train_outputs,
        "tune_outputs": tune.outputs.tune_outputs,
    }


# -----------------------------
# Submit
# -----------------------------
if __name__ == "__main__":
    pipeline_job = auto_pricing_pipeline()
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name="auto-pricing-e2e"
    )
    print(f"Submitted pipeline job: {pipeline_job.name}")

    # In CI, stream logs until completion
    try:
        from azure.ai.ml._azure_environ import _is_in_ci

        if _is_in_ci():
            ml_client.jobs.stream(pipeline_job.name)
    except Exception:
        pass
