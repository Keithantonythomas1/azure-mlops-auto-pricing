# aml/pipeline.py
import os
import json
import argparse
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, Input, Output, command
from azure.ai.ml.entities import Environment


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_cfg() -> dict:
    """Load defaults from configs/aml_config.json if present."""
    cfg_path = os.environ.get("AML_CONFIG_PATH", "configs/aml_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}


def value_or_env(cfg: dict, key: str, env: str, default: Optional[str] = None) -> Optional[str]:
    """Config value with env override and default fallback."""
    return os.environ.get(env) or cfg.get(key) or default


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    # ---------- Defaults from file / env ----------
    cfg = load_cfg()

    default_sub = value_or_env(cfg, "subscription_id", "AML_SUBSCRIPTION_ID")
    default_rg = value_or_env(cfg, "resource_group", "AML_RESOURCE_GROUP")
    default_ws = value_or_env(cfg, "workspace_name", "AML_WORKSPACE")
    default_compute = value_or_env(cfg, "compute_target", "AML_COMPUTE")
    default_exp = value_or_env(cfg, "experiment_name", "AML_EXPERIMENT", "auto-pricing-e2e")

    # KEY: your data asset from the screenshots
    # azureml Named Asset URI:
    #   azureml:UsedCars:1
    default_data_asset = value_or_env(cfg, "data_asset", "AML_DATA_ASSET", "azureml:UsedCars:1")

    parser = argparse.ArgumentParser("Submit Azure ML pipeline")
    parser.add_argument("--subscription", default=default_sub)
    parser.add_argument("--resource-group", dest="resource_group", default=default_rg)
    parser.add_argument("--workspace", default=default_ws)
    parser.add_argument("--compute", default=default_compute)
    parser.add_argument("--experiment-name", dest="experiment_name", default=default_exp)
    parser.add_argument("--data-asset", dest="data_asset", default=default_data_asset)
    # Optional files to write small outputs (for CI log collection etc.)
    parser.add_argument("--run-out", dest="run_out", default=None, help="Path to write run URL")
    parser.add_argument("--model-out", dest="model_out", default=None, help="Path to write model info folder name")
    parser.add_argument("--metrics-out", dest="metrics_out", default=None, help="Path to write metrics folder name")

    args = parser.parse_args()

    # ---------- Connect to workspace ----------
    missing = [k for k, v in {
        "subscription": args.subscription,
        "resource_group": args.resource_group,
        "workspace": args.workspace,
        "compute": args.compute
    }.items() if not v]
    if missing:
        raise SystemExit(f"Missing required config values: {', '.join(missing)}")

    cred = DefaultAzureCredential()
    ml_client = MLClient(
        credential=cred,
        subscription_id=args.subscription,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace,
    )

    # ---------- Reusable environment ----------
    env = Environment(
        name="auto-pricing-env",
        conda_file="env/conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )

    # ---------- Step: Data Prep ----------
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
        compute=args.compute,
        display_name="prep",
    )

    # ---------- Step: Train ----------
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
        compute=args.compute,
        display_name="train",
    )

    # ---------- Step: Tune ----------
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
        compute=args.compute,
        display_name="tune",
    )

    # ---------- Step: Register ----------
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
        compute=args.compute,
        display_name="reg",
    )

    # ---------- Pipeline DAG ----------
    @dsl.pipeline(compute=args.compute, description="Auto pricing full MLOps pipeline")
    def auto_pricing_pipeline(data_file: Input):
        prep_step = prep_job(data=data_file)
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

    # ---------- Bind the Azure ML data asset ----------
    # From your screenshots:
    #   Named asset: UsedCars (version 1)
    #   Asset URI  : azureml:UsedCars:1
    data_input = Input(type="uri_file", path=args.data_asset)

    pipeline_job = auto_pricing_pipeline(data_file=data_input)
    pipeline_job.experiment_name = args.experiment_name

    # ---------- Submit ----------
    submitted = ml_client.jobs.create_or_update(pipeline_job)
    run_url = submitted.studio_url
    print(f"\nSubmitted pipeline job:")
    print(f"  Name: {submitted.name}")
    print(f"  URL : {run_url}\n")

    # Optional: write small artifacts for CI
    if args.run_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.run_out)), exist_ok=True)
        with open(args.run_out, "w") as f:
            f.write(run_url)

    # These are folder names produced by the pipeline; write the logical names if requested
    if args.model_out:
        with open(args.model_out, "w") as f:
            f.write("model_info")
    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            f.write("metrics")


if __name__ == "__main__":
    main()
