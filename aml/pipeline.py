# aml/pipeline.py
import argparse
import json
import os
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, Input, Output, command
from azure.ai.ml.entities import Environment


# ----------------------------
# Helpers
# ----------------------------
def get_ml_client(subscription: str, resource_group: str, workspace: str) -> MLClient:
    cred = DefaultAzureCredential()
    return MLClient(
        cred,
        subscription_id=subscription,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )


def write_json(path: str, payload: dict):
    Path(path).write_text(json.dumps(payload, indent=2))


# ----------------------------
# Reusable environment
# ----------------------------
# NOTE: Do NOT set pipeline-level compute; each node sets compute itself.
ENV = Environment(
    name="auto-pricing-env",
    conda_file="env/conda.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)


# ----------------------------
# Component commands
# ----------------------------
def make_prep_job(compute_name: str):
    return command(
        code="./scripts",
        command="python prep.py --data ${{inputs.data}} --out ${{outputs.prep_dir}}",
        inputs={"data": Input(type="uri_file")},
        outputs={"prep_dir": Output(type="uri_folder", mode="rw_mount")},
        environment=ENV,
        compute=compute_name,
        display_name="prep",
    )


def make_train_job(compute_name: str):
    # train.py requires: --data_dir, --train_dir (required), --metrics (required)
    return command(
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
        environment=ENV,
        compute=compute_name,
        display_name="train",
    )


def make_tune_job(compute_name: str):
    # tune.py requires: --data_dir, --best_dir (required)
    return command(
        code="./scripts",
        command=(
            "python tune.py "
            "--data_dir ${{inputs.data_dir}} "
            "--best_dir ${{outputs.best_dir}}"
        ),
        inputs={"data_dir": Input(type="uri_folder")},
        outputs={"best_dir": Output(type="uri_folder", mode="rw_mount")},
        environment=ENV,
        compute=compute_name,
        display_name="tune",
    )


def make_reg_job(compute_name: str):
    return command(
        code="./scripts",
        command="python reg.py --best_dir ${{inputs.best_dir}} --model_info ${{outputs.model_info}}",
        inputs={"best_dir": Input(type="uri_folder")},
        outputs={"model_info": Output(type="uri_folder", mode="rw_mount")},
        environment=ENV,
        compute=compute_name,
        display_name="reg",
    )


# ----------------------------
# Pipeline definition
# ----------------------------
# IMPORTANT: No `compute=...` at the pipeline decorator level (fixes InvalidExpression parent.compute)
@dsl.pipeline(description="Auto pricing full MLOps pipeline")
def auto_pricing_pipeline(data_path: Input, compute_name: str):
    prep_step = make_prep_job(compute_name)(data=data_path)
    train_step = make_train_job(compute_name)(data_dir=prep_step.outputs.prep_dir)
    tune_step = make_tune_job(compute_name)(data_dir=prep_step.outputs.prep_dir)
    reg_step = make_reg_job(compute_name)(best_dir=tune_step.outputs.best_dir)

    return {
        "prep_dir": prep_step.outputs.prep_dir,
        "train_dir": train_step.outputs.train_dir,
        "metrics": train_step.outputs.metrics,
        "best_dir": tune_step.outputs.best_dir,
        "model_info": reg_step.outputs.model_info,
    }


# ----------------------------
# CLI / submission
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser("Submit Azure ML pipeline")
    p.add_argument("--subscription", required=True)
    p.add_argument("--resource-group", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--compute", required=True)
    # optional; accepted for parity with workflow but not used directly here
    p.add_argument("--location", required=False)

    # outputs for the CI summary/artifacts
    p.add_argument("--run-out", default="run_info.json")
    p.add_argument("--model-out", default="model_info.json")
    p.add_argument("--metrics-out", default="metrics.json")
    return p.parse_args()


def main():
    args = parse_args()

    ml_client = get_ml_client(
        subscription=args.subscription,
        resource_group=args.resource_group,
        workspace=args.workspace,
    )

    # Your workspace data asset (from screenshots): name=UsedCars, version=1
    # Use as an AML Input so jobs get a stable, server-side URI.
    data_input = Input(type="uri_file", path="azureml:UsedCars:1")

    # Build the pipeline (pass compute via input param to avoid parent.compute binding)
    pipe = auto_pricing_pipeline(data_path=data_input, compute_name=args.compute)
    pipe.settings.default_compute = None  # explicit, just in case

    # Give the run a readable name
    pipe.experiment_name = "auto-pricing-e2e"

    submitted = ml_client.jobs.create_or_update(pipe)
    print(f"Submitted pipeline job: {submitted.name}")
    print(f"RunId: {submitted.name}")
    print(
        "Web View: "
        f"https://ml.azure.com/runs/{submitted.name}"
        f"?wsid=/subscriptions/{args.subscription}"
        f"/resourcegroups/{args.resource_group}"
        f"/workspaces/{args.workspace}"
    )

    # Persist lightweight artifacts for the CI summary
    write_json(
        args.run_out,
        {
            "run_id": submitted.name,
            "status": "Submitted",
            "studio_url": (
                "https://ml.azure.com/runs/"
                f"{submitted.name}"
                f"?wsid=/subscriptions/{args.subscription}"
                f"/resourcegroups/{args.resource_group}"
                f"/workspaces/{args.workspace}"
            ),
        },
    )

    # We can't block until completion here; write placeholders for model/metrics
    write_json(args.model_out, {"model_uri": None})
    write_json(args.metrics_out, {"metrics_uri": None})
    print("Wrote run info to", args.run_out)
    print("Wrote model info to", args.model_out)
    print("Wrote metrics info to", args.metrics_out)


if __name__ == "__main__":
    main()
