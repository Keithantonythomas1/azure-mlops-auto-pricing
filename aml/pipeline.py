# aml/pipeline.py
import os
import json
import argparse
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, Input, Output, command
from azure.ai.ml.entities import Environment


# ----------------------------
# 1) Load AML config (optional file)
# ----------------------------
def load_cfg():
    cfg_path = os.environ.get("AML_CONFIG_PATH", "configs/aml_config.json")
    if Path(cfg_path).exists():
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}


def get_arg_parser():
    p = argparse.ArgumentParser("Submit Azure ML pipeline")
    p.add_argument("--subscription", type=str, default=None)
    p.add_argument("--resource-group", "--resource_group", dest="resource_group", type=str, default=None)
    p.add_argument("--workspace", type=str, default=None)
    p.add_argument("--compute", type=str, default=None)
    p.add_argument("--experiment-name", "--experiment_name", dest="experiment_name", type=str, default="auto-pricing-e2e")
    p.add_argument("--data-asset", "--data_asset", dest="data_asset", type=str, default=None)

    # optional output summaries for CI
    p.add_argument("--run-out", "--run_out", dest="run_out", type=str, default="run_info.json")
    p.add_argument("--model-out", "--model_out", dest="model_out", type=str, default="model_info.json")
    p.add_argument("--metrics-out", "--metrics_out", dest="metrics_out", type=str, default="metrics.json")
    return p


def main():
    # ----- config + args -----
    cfg = load_cfg()
    args = get_arg_parser().parse_args()

    subscription_id = (
        args.subscription
        or cfg.get("subscription_id")
        or os.environ.get("AML_SUBSCRIPTION_ID")
    )
    resource_group = (
        args.resource_group
        or cfg.get("resource_group")
        or os.environ.get("AML_RESOURCE_GROUP")
    )
    workspace_name = (
        args.workspace
        or cfg.get("workspace_name")
        or os.environ.get("AML_WORKSPACE_NAME")
    )

    # Accept several keys for compute, with env fallback
    compute_name = (
        args.compute
        or cfg.get("compute_target")
        or cfg.get("compute")
        or os.environ.get("AML_COMPUTE_NAME")
    )
    if not compute_name:
        raise KeyError(
            "Compute target not provided. Set --compute, "
            "or add 'compute_target' (or 'compute') in configs/aml_config.json, "
            "or set env AML_COMPUTE_NAME."
        )

    # Data asset to use (default to your AML data asset)
    data_asset_uri = (
        args.data_asset
        or cfg.get("data_asset_uri")
        or os.environ.get("AML_DATA_ASSET")
        or "azureml:UsedCars@latest"
    )

    # ----- client -----
    cred = DefaultAzureCredential()
    ml_client = MLClient(
        cred,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
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
        inputs={"data": Input(type="uri_file")},
        outputs={"prep_dir": Output(type="uri_folder", mode="rw_mount")},
        environment=env,
        compute=compute_name,
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
        inputs={"data_dir": Input(type="uri_folder")},
        outputs={
            "train_dir": Output(type="uri_folder", mode="rw_mount"),
            "metrics": Output(type="uri_folder", mode="rw_mount"),
        },
        environment=env,
        compute=compute_name,
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
        inputs={"data_dir": Input(type="uri_folder")},
        outputs={"best_dir": Output(type="uri_folder", mode="rw_mount")},
        environment=env,
        compute=compute_name,
        display_name="tune",
    )

    # ---- Register Model ----
    reg_job = command(
        code="./scripts",
        command="python reg.py --best_dir ${{inputs.best_dir}} --model_info ${{outputs.model_info}}",
        inputs={"best_dir": Input(type="uri_folder")},
        outputs={"model_info": Output(type="uri_folder", mode="rw_mount")},
        environment=env,
        compute=compute_name,
        display_name="reg",
    )

    # ----------------------------
    # 4) Pipeline Definition
    # ----------------------------
    @dsl.pipeline(
        compute=compute_name,
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

    # Create the pipeline input directly from the AML data asset
    data_input = Input(type="uri_file", path=data_asset_uri)

    pipeline_job = auto_pricing_pipeline(data_path=data_input)
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=args.experiment_name
    )

    # ----------------------------
    # 5) Write small JSON summaries for CI
    # ----------------------------
    run_info = {
        "name": pipeline_job.name,
        "id": pipeline_job.id,
        "status": pipeline_job.status,
        "experiment_name": args.experiment_name,
        "studio_url": getattr(pipeline_job, "studio_url", None),
    }
    Path(args.run_out).write_text(json.dumps(run_info, indent=2))

    # We don't have model metadata here; put a simple placeholder.
    model_info = {
        "note": "Model registration happens inside the 'reg' step. "
                "Inspect the child job outputs for full details."
    }
    Path(args.model_out).write_text(json.dumps(model_info, indent=2))

    # Optional placeholder metrics (extend later to fetch real child-run metrics)
    metrics = {
        "note": "Training metrics are written by the 'train' step into its output folder "
                "and logged to MLflow. This file is a CI placeholder."
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))

    print("Submitted pipeline:", pipeline_job.name)
    if getattr(pipeline_job, "studio_url", None):
        print("Studio URL:", pipeline_job.studio_url)


if __name__ == "__main__":
    main()

