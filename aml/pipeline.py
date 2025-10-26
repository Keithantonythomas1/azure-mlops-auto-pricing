import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, dsl, command
from azure.ai.ml.entities import Environment
import json

# Load AML config from env or file
def load_cfg():
    cfg_path = os.environ.get("AML_CONFIG_PATH", "configs/aml_config.json")
    with open(cfg_path, "r") as f:
        raw = f.read()
    # Using concrete values from configs/aml_config.json (already set).
    return json.loads(raw)

cfg = load_cfg()

cred = DefaultAzureCredential()
ml_client = MLClient(cred,
                     subscription_id=cfg["subscription_id"],
                     resource_group_name=cfg["resource_group"],
                     workspace_name=cfg["workspace_name"])

env = Environment(
    name="auto-pricing-env",
    conda_file="env/conda.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)

compute_target = cfg.get("compute", "cpu-cluster")

prep_component = command(
    name="prep",
    display_name="Data Prep",
    environment=env,
    code=".",
    command="python scripts/prep.py",
    compute=compute_target,
    inputs={
        "data_file": Input(type="uri_file", path="azureml:UsedCars:1")
    },
    environment_variables={"DATA_PATH": "${{inputs.data_file}}"},
    outputs={
        "prep_outputs": Output(type="uri_folder", path="azureml://datastores/workspaceblobstore/paths/auto-pricing/outputs/prep")
    }
)

train_component = command(
    name="train",
    display_name="Train Baseline",
    environment=env,
    code=".",
    command="python scripts/train.py",
    compute=compute_target,
    inputs={
        "INPUT_DIR": Input(path="${{parent.jobs.prep.outputs.prep_outputs}}", type="uri_folder"),
        "N_ESTIMATORS": 200,
        "MAX_DEPTH": 12
    },
    outputs={
        "train_outputs": Output(type="uri_folder", path="azureml://datastores/workspaceblobstore/paths/auto-pricing/outputs/train")
    }
)

tune_component = command(
    name="tune",
    display_name="Hyperparameter Tuning",
    environment=env,
    code=".",
    command="python scripts/tune.py",
    compute=compute_target,
    inputs={
        "INPUT_DIR": Input(path="${{parent.jobs.prep.outputs.prep_outputs}}", type="uri_folder")
    },
    outputs={
        "tune_outputs": Output(type="uri_folder", path="azureml://datastores/workspaceblobstore/paths/auto-pricing/outputs/tune")
    }
)

register_component = command(
    name="register",
    display_name="Register Model",
    environment=env,
    code=".",
    command="python scripts/register_model.py",
    compute=compute_target,
    inputs={
        "MODEL_PATH": Input(path="${{parent.jobs.tune.outputs.tune_outputs}}/best_model.pkl", type="uri_file")
    }
)

@dsl.pipeline(compute=compute_target, description="Auto pricing MLOps pipeline")
def auto_pricing_pipeline():
    prep = prep_component()
    train = train_component()
    train.inputs["INPUT_DIR"] = prep.outputs["prep_outputs"]
    tune = tune_component()
    tune.inputs["INPUT_DIR"] = prep.outputs["prep_outputs"]
    reg  = register_component()
    reg.inputs["MODEL_PATH"] = tune.outputs["tune_outputs"]

    return {
        "prep_outputs": prep.outputs["prep_outputs"],
        "train_outputs": train.outputs["train_outputs"],
        "tune_outputs": tune.outputs["tune_outputs"]
    }

if __name__ == "__main__":
    pipeline_job = auto_pricing_pipeline()
    pipeline_job = ml_client.jobs.create_or_update(pipeline_job, experiment_name="auto-pricing-e2e")
    print(f"Submitted pipeline job: {pipeline_job.name}")
    # Optionally, wait for completion:
    from azure.ai.ml._azure_environ import _is_in_ci
    if _is_in_ci():
        ml_client.jobs.stream(pipeline_job.name)