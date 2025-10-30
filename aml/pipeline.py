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
    compute="${{parent.compute}}",
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
    p.add_argument(
        "--subscription",
        dest="subscription",
        default=os.getenv("AZURE_SUBSCRIPTION_ID", "f7583458-6f5b-4491-add9-f827568d2957"),
    )
    p.add_argument(
        "--resource-group",
        dest="resource_group",
        default=os.getenv("AZURE_RESOURCE_GROUP", "MSFT-AI-Class"),
    )
    p.add_argument(
        "--workspace",
        dest="workspace",
        default=os.getenv("AZURE_ML_WORKSPACE", "MyFirstWorkSpace"),
    )
    p.add_argument(
        "--compute",
        dest="compute",
        default=os.getenv("AZURE_ML_COMPUTE", "Keith-Compute"),
    )
    p.add_argument(
        "--experiment-name",
        dest="experiment_name",
        default="auto-pricing-e2e",
    )
    # Registered data asset (from your screenshots)
    p.add_argument(
        "--data-asset",
        dest="data_asset",
        default="azureml:UsedCars:1",
    )

    # CI artifact paths
    p.add_argument("--run-out", dest="run_out", default="run.json")
    p.add_argument("--model-out", dest="model_out", default="model_info.json")
    p.add_argument("--metrics-out", dest="metrics_out", default="metrics.json")

    # Keep for backward compatibility; ignored
    p.add_argument("--location", dest="location", required=False)

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

    # Use the registered data asset in the workspace
    data_input = Input(type="uri_file", path=args.data_asset)

    pipeline_job = auto_pricing_pipeline(
        data_path=data_input,
        compute=args.compute,
    )
    pipeline_job.settings.default_compute = args.compute
    pipeline_job.experiment_name = args.experiment_name

    submitted = ml_client.jobs.create_or_update(pipeline_job)
    print(f"Submitted pipeline job: {submitted.name}")
    try:
        ml_client.jobs.stream(submitted.name)
    except Exception as e:
        # Streaming isn't critical; continue even if it fails
        print(f"Streaming failed (non-fatal): {e}")

    final_job = ml_client.jobs.get(submitted.name)

    # Build simple artifact payloads
    outputs = getattr(final_job, "outputs", {}) or {}
    def _safe_uri(key):
        try:
            return outputs[key].uri if key in outputs and hasattr(outputs[key], "uri") else None
        except Exception:
            return None

    run_payload = {
        "job_name": final_job.name,
        "status": getattr(final_job, "status", None),
        "experiment_name": args.experiment_name,
        "studio_url": getattr(getattr(final_job, "services", {}), "get", lambda *_: None)("Studio", {}).get("endpoint", None)
        if hasattr(final_job, "services") else None,
        "outputs": {
            "prep_dir": _safe_uri("prep_dir"),
            "train_dir": _safe_uri("train_dir"),
            "metrics": _safe_uri("metrics"),
            "best_dir": _safe_uri("best_dir"),
            "model_info": _safe_uri("model_info"),
        },
    }

    write_json(args.run_out, run_payload)
    write_json(args.model_out, {"model_info_uri": _safe_uri("model_info")})
    write_json(args.metrics_out, {"metrics_uri": _safe_uri("metrics")})

    print(f"Wrote run info to {args.run_out}")
    print(f"Wrote model info to {args.model_out}")
    print(f"Wrote metrics info to {args.metrics_out}")


if __name__ == "__main__":
    main()
