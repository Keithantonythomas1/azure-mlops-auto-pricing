# aml/pipeline.py
import os
import json
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, Input, Output, command
from azure.ai.ml.entities import Environment

# ----------------------------
# 1) Load AML config
# ----------------------------
def load_cfg():
    cfg_path = os.environ.get("AML_CONFIG_PATH", "configs/aml_config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)

cfg = load_cfg()

cred = DefaultAzureCredential()
ml_client = MLClient(
    cred,
    subscription_id=cfg["subscription_id"],
    resource_group_name=cfg["resource_group"],
    workspace_name=cfg["workspace_name"],
)

# ----------------------------
# 2) Reusable environment
# ----------------------------
env = Environment(
    name="auto-pricing-env",
    conda_file="env/conda.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

compute_target = cfg.get("compute", "Keith-Compute")  # make sure this exists and is running

# ----------------------------
# 3) Components (commands)
#    IMPORTANT: only reference IO paths with ${{inputs.*}} and ${{outputs.*}}
# ----------------------------

# PREP: reads a workspace data asset and writes a folder output
prep_component = command(
    name="prep",
    display_name="Data Prep",
    environment=env,
    code=".",  # repo root
    compute=compute_target,
    inputs={
        "data_file": Input(type="uri_file", path="azureml:UsedCars:1"),
    },
    outputs={
        "prep_dir": Output(type="uri_folder"),
    },
    # NOTE: Use placeholders for IO paths
    command=(
        "python scripts/prep.py "
        "--data ${{inputs.data_file}} "
        "--out  ${{outputs.prep_dir}}"
    ),
)

# TRAIN: consumes prep output; writes a model (or any artifacts) to its output folder
train_component = command(
    name="train",
    display_name="Train Baseline",
    environment=env,
    code=".",
    compute=compute_target,
    inputs={
        "data_dir": Input(type="uri_folder"),  # we’ll wire prep.prep_dir into this in the pipeline
        "n_estimators": 200,
        "max_depth": 12,
    },
    outputs={
        "train_dir": Output(type="uri_folder"),
    },
    command=(
        "python scripts/train.py "
        "--data ${{inputs.data_dir}} "
        "--n_estimators ${{inputs.n_estimators}} "
        "--max_depth ${{inputs.max_depth}} "
        "--out  ${{outputs.train_dir}}"
    ),
)

# TUNE: consumes prep output; writes a folder that includes best_model.pkl
tune_component = command(
    name="tune",
    display_name="Hyperparameter Tuning",
    environment=env,
    code=".",
    compute=compute_target,
    inputs={
        "data_dir": Input(type="uri_folder"),
    },
    outputs={
        "best_dir": Output(type="uri_folder"),  # this folder must contain best_model.pkl
    },
    command=(
        "python scripts/tune.py "
        "--data ${{inputs.data_dir}} "
        "--out  ${{outputs.best_dir}}"
    ),
)

# REGISTER: consumes the *folder* from tune and reads best_model.pkl *inside* the command
register_component = command(
    name="register",
    display_name="Register Model",
    environment=env,
    code=".",
    compute=compute_target,
    inputs={
        # NOTE: we pass the *folder* NodeOutput here, not a string path
        "best_dir": Input(type="uri_folder"),
    },
    command=(
        # The script should accept a --model argument and register it
        "python scripts/register_model.py "
        "--model ${{inputs.best_dir}}/best_model.pkl"
    ),
)

# ----------------------------
# 4) Pipeline: wire NodeOutputs, not strings
# ----------------------------
@dsl.pipeline(
    compute=compute_target,
    description="Auto pricing MLOps pipeline (prep -> train -> tune -> register).",
)
def auto_pricing_pipeline():
    prep = prep_component()
    train = train_component(data_dir=prep.outputs.prep_dir)
    tune  = tune_component(data_dir=prep.outputs.prep_dir)
    reg   = register_component(best_dir=tune.outputs.best_dir)

    # optional: expose outputs for debugging/inspection
    return {
        "prep_dir": prep.outputs.prep_dir,
        "train_dir": train.outputs.train_dir,
        "best_dir": tune.outputs.best_dir,
    }

# ----------------------------
# 5) Submit
# ----------------------------
if __name__ == "__main__":
    print("Submitting Auto Pricing pipeline to Azure ML ...")
    pipeline_job = auto_pricing_pipeline()
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name="auto-pricing-e2e",
    )
    print(f"Submitted pipeline job: {pipeline_job.name}")

    # Optional: stream logs when running in CI
    try:
        from azure.ai.ml._azure_environ import _is_in_ci
        if _is_in_ci():
            ml_client.jobs.stream(pipeline_job.name)
    except Exception:
        pass
