# aml/pipeline.py
import os
import json
import shutil
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, Input, Output, command
from azure.ai.ml.entities import Environment

# ----------------------------
# 1) Load AML config (allows env override)
# ----------------------------
def load_cfg():
    # Prefer explicit path via env, else default file
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

compute_target = cfg.get("compute", "Keith-Compute")

# ----------------------------
# 3) Components
# ----------------------------

# PREP
prep_component = command(
    name="prep",
    display_name="Data Prep",
    environment=env,
    code=".",
    compute=compute_target,
    inputs={
        # Make sure this data asset exists in your workspace
        "data_file": Input(type="uri_file", path=cfg.get("data_asset", "azureml:UsedCars:1")),
    },
    outputs={
        "prep_dir": Output(type="uri_folder"),
    },
    command=(
        "python scripts/prep.py "
        "--data ${{inputs.data_file}} "
        "--out  ${{outputs.prep_dir}}"
    ),
)

# TRAIN — add a tiny single-file metrics output
train_component = command(
    name="train",
    display_name="Train Baseline",
    environment=env,
    code=".",
    compute=compute_target,
    inputs={
        "data_dir": Input(type="uri_folder"),
        "n_estimators": 200,
        "max_depth": 12,
    },
    outputs={
        "train_dir": Output(type="uri_folder"),
        "metrics":   Output(type="uri_file"),   # NEW
    },
    command=(
        "python scripts/train.py "
        "--data ${{inputs.data_dir}} "
        "--n_estimators ${{inputs.n_estimators}} "
        "--max_depth ${{inputs.max_depth}} "
        "--out  ${{outputs.train_dir}} "
        "--metrics ${{outputs.metrics}}"        # NEW
    ),
)

# TUNE
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
        "best_dir": Output(type="uri_folder"),
    },
    command=(
        "python scripts/tune.py "
        "--data ${{inputs.data_dir}} "
        "--out  ${{outputs.best_dir}}"
    ),
)

# REGISTER — add a tiny single-file model_info output
register_component = command(
    name="register",
    display_name="Register Model",
    environment=env,
    code=".",
    compute=compute_target,
    inputs={
        "best_dir": Input(type="uri_folder"),
    },
    outputs={
        "model_info": Output(type="uri_file"),  # NEW
    },
    command=(
        "python scripts/register_model.py "
        "--model ${{inputs.best_dir}}/best_model.pkl "
        "--out   ${{outputs.model_info}}"       # NEW
    ),
)

# ----------------------------
# 4) Pipeline wiring
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

    # Expose outputs so we can download them after completion
    return {
        "prep_dir":   prep.outputs.prep_dir,
        "train_dir":  train.outputs.train_dir,
        "metrics":    train.outputs.metrics,     # NEW
        "best_dir":   tune.outputs.best_dir,
        "model_info": reg.outputs.model_info,    # NEW
    }

# ----------------------------
# 5) Submit and export small JSONs for CI
# ----------------------------
if __name__ == "__main__":
    print("Submitting Auto Pricing pipeline to Azure ML ...")
    pipeline_job = auto_pricing_pipeline()
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name="auto-pricing-e2e",
    )
    print(f"Submitted pipeline job: {pipeline_job.name}")

    # Stream logs & wait to finish
    try:
        ml_client.jobs.stream(pipeline_job.name)
    except Exception as e:
        print(f"Stream failed (continuing): {e}")

    # Write a lightweight run_info.json for the CI summary
    with open("run_info.json", "w") as f:
        json.dump({"job_name": pipeline_job.name}, f, indent=2)

    # Download the two tiny single-file outputs into the workspace
    # They’ll be placed under ./azureml/{output_name}/...
    out_dir = Path(".").resolve()

    def _download_single(output_name: str, dest_name: str):
        try:
            ml_client.jobs.download(
                name=pipeline_job.name,
                output_name=output_name,
                download_path=str(out_dir),
                overwrite=True,
            )
            # AzureML writes to ./<output_name>/ (file keeps its original filename)
            # Find the first file and copy it to the expected CI filename.
            root = out_dir / output_name
            if root.exists():
                # Grab the first file inside (there should be exactly one for uri_file)
                for p in root.rglob("*"):
                    if p.is_file():
                        shutil.copy2(p, out_dir / dest_name)
                        print(f"Wrote {dest_name} from {p}")
                        return True
            print(f"{output_name} not found or empty")
            return False
        except Exception as e:
            print(f"Failed to download {output_name}: {e}")
            return False

    _download_single("metrics",    "metrics.json")
    _download_single("model_info", "model_info.json")

    print("Done.")
