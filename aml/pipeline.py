# aml/pipeline.py
import os
import json
import argparse
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, Input, Output, command
from azure.ai.ml.entities import Environment

# ----------------------------
# Helpers
# ----------------------------
def write_json(path: str, payload: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

# ----------------------------
# Reusable environment
# ----------------------------
env = Environment(
    name="auto-pricing-env",
    conda_file="env/conda.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

# ----------------------------
# Define individual steps
# ----------------------------
# ---- Data Prep ----
prep_job = command(
    code="./scripts",
    command="python prep.py --data ${{inputs.data}} --out ${{outputs.prep_dir}}",
    inputs={"data": Input(type="uri_file")},
    outputs={"prep_dir": Output(type="uri_folder", mode="rw_mount")},
    environment=env,
    compute="${{parent.compute}}",   # resolved from pipeline compute
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
    compute="${{parent.compute}}",
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
    compute="${{parent.compute}}",
    display_name="tune",
)

# ---- Register Model ----
reg_job = command(
    code="./scripts",
    command="python reg.py --best_dir ${{inputs.best_dir}} --model_info ${{outputs.model_info}}",
    inputs={"best_dir": Input(type="uri_folder")},
    outputs={"model_info": Output(type="uri_folder", mode="rw_mount")},
    environment=env,
    compute="${{parent.compute}}",
    display_name="reg",
)

# ----------------------------
# Pipeline Definition
# ----------------------------
@dsl.pipeline(description="Auto pricing full MLOps pipeline")
def auto_pricing_pipeline(data_path: Input, compute: str):
    # NOTE: we pass compute down via parent context
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
# CLI & Submission
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser("Submit Azure ML pipeline")
    p.add_argument("--subscription", required=False,
                   default=os.getenv("AZURE_SUBSCRIPTION_ID", "f7583458-6f5b-4491-add9-f827568d2957"))
    p.add_argument("--resource-group", required=False,
                   default=os.getenv("AZURE_RESOURCE_GROUP", "MSFT-AI-Class"))
    p.add_argument("--workspace", required=False,
                   default=os.getenv("AZURE_ML_WORKSPACE", "MyFirstWorkSpace"))
    p.add_argument("--compute", required=False, default=os.getenv("AZURE_ML_COMPUTE", "Keith-Compute"))
    p.add_argument("--experiment-name", required=False, default="auto-pricing-e2e")

    # Use the registered data asset shown in your screenshots
    p.add_argument("--data-asset", required=False, default="azureml:UsedCars:1")

    # These are for CI artifacts
    p.add_argument("--run-out", required=False, default="run.json")
    p.add_argument("--model-out", required=False, default="model_info.json")
    p.add_argument("--metrics-out", required=False, default="metrics.json")

    # Accept but ignore to keep current workflow compatible
    p.add_argument("--location", required=False, help="(Ignored) kept for backwards compatibility")

    return p.parse_args()


def main():
    args = parse_args()

    # Auth & client
    cred = DefaultAzureCredential()
    ml_client = MLClient(
        credential=cred,
        subscription_id=args.subscription,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace,
    )

    # Build pipeline using the data asset already in the workspace
    data_input = Input(type="uri_file", path=args.data-asset)
    pipeline_job = auto_pricing_pipeline(
        data_path=data_input,
        compute=args.compute,
    )
    pipeline_job.settings.default_compute = args.compute
    pipeline_job.experiment_name = args.experiment-name if hasattr(args, "experiment-name") else args.experiment_name

    # Submit
    submitted = ml_client.jobs.create_or_update(pipeline_job)
    print(f"Submitted pipeline job: {submitted.name}")
    ml_client.jobs.stream(submitted.name)

    # Refresh to read final outputs
    final_job = ml_client.jobs.get(submitted.name)

    # Prepare simple outputs for CI summary
    run_payload = {
        "job_name": final_job.name,
        "status": final_job.status,
        "experiment_name": getattr(final_job, "experiment_name", args.experiment_name),
        "studio_url": getattr(final_job, "services", {}).get("Studio", {}).get("endpoint", None)
        if hasattr(final_job, "services") else None,
        "outputs": {
            k: (getattr(v, "uri", None) if hasattr(v, "uri") else None)
            for k, v in getattr(final_job, "outputs", {}).items()
        },
    }
    write_json(args.run_out, run_payload)

    # Best-effort model/metrics files for downstream steps
    outputs = getattr(final_job, "outputs", {})
    model_uri = outputs.get("model_info").uri if "model_info" in outputs else None
    metrics_uri = outputs.get("metrics").uri if "metrics" in outputs else None

    write_json(args.model_out, {"model_info_uri": model_uri})
    write_json(args.metrics_out, {"metrics_uri": metrics_uri})

    print(f"Wrote run info to {args.run_out}")
    print(f"Wrote model info to {args.model_out}")
    print(f"Wrote metrics info to {args.metrics_out}")


if __name__ == "__main__":
    main()
