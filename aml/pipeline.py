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

# ----------------------------
# 3) Define individual steps
# ----------------------------

# ---- Data Prep ----
prep_job = command(
    code="./scripts",
    command="python prep.py --data ${{inputs.data}} --out ${{outputs.prep_dir}}",
    inputs={
        "data": Input(type="uri_file")
    },
    outputs={
        "prep_dir": Output(type="uri_folder", mode="rw_mount")
    },
    environment=env,
    compute=cfg["compute_target"],
    display_name="prep",
)

# ---- Train ----
train_job = command(
    code="./scripts",
    command=(
        "python train.py "
        "--data_dir ${{inputs.data_dir}} "
        "--train_dir ${{outputs.train_dir}} "
        "--metrics ${{outputs.metrics}} "
        "--n_estimators 200 "
        "--max_depth 12"
    ),
    inputs={
        "data_dir": Input(type="uri_folder"),
    },
    outputs={
        "train_dir": Output(type="uri_folder", mode="rw_mount"),
        "metrics": Output(type="uri_folder", mode="rw_mount"),
    },
    environment=env,
    compute=cfg["compute_target"],
    display_name="train",
)

# ---- Tune ----
tune_job = command(
    code="./scripts",
    command=(
        "python tune.py "
        "--data_dir ${{inputs.data_dir}} "
        "--best_dir ${{outputs.best_dir}}"
    ),
    inputs={
        "data_dir": Input(type="uri_folder"),
    },
    outputs={
        "best_dir": Output(type="uri_folder", mode="rw_mount"),
    },
    environment=env,
    compute=cfg["compute_target"],
    display_name="tune",
)

# ---- Register Model ----
reg_job = command(
    code="./scripts",
    command="python reg.py --best_dir ${{inputs.best_dir}} --model_info ${{outputs.model_info}}",
    inputs={
        "best_dir": Input(type="uri_folder"),
    },
    outputs={
        "model_info": Output(type="uri_folder", mode="rw_mount"),
    },
    environment=env,
    compute=cfg["compute_target"],
    display_name="reg",
)

# ----------------------------
# 4) Pipeline Definition
# ----------------------------
@dsl.pipeline(
    compute=cfg["compute_target"],
    description="Auto pricing full MLOps pipeline",
)
def auto_pricing_pipeline(data_path: Input):
    prep_step = prep_job(data=data_path)
    train_step = train_job(data_dir=prep_step.outputs.prep_dir)
    tune_step = tune_job(data_dir=prep_step.outputs.prep_dir)
    reg_step = reg_job(best_dir=tune_step.outputs.best_dir)
    return {
        "prep_dir": prep_step.outputs.prep_dir,
        "train_dir": train_step.outputs.train_dir,
        "metrics": train_step.outputs.metrics,
        "best_dir": tune_step.outputs.best_dir,
        "model_info": reg_step.outputs.model_info,
    }

# ----------------------------
# 5) Submit Pipeline
# ----------------------------
pipeline_job = auto_pricing_pipeline(data_path=Input(path="data/used_cars.csv", type="uri_file"))
pipeline_job = ml_client.jobs.create_or_update(pipeline_job)
print(f"✅ Pipeline submitted successfully! Job name: {pipeline_job.name}")

