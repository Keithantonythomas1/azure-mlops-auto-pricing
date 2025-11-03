import argparse, os, sys
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model_name", required=True)
    args = ap.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"model_dir not found: {args.model_dir}")

    # Use federated credentials (OIDC in Actions) or SP if configured
    cred = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    mlc = MLClient.from_config(credential=cred)  # uses AML config from job context

    model = Model(
        name=args.model_name,
        path=args.model_dir,
        type="custom_model",          # we saved joblib; keep as custom folder
        description="Auto pricing model registered by pipeline",
        tags={"source": "pipeline", "framework": "sklearn"},
    )
    out = mlc.models.create_or_update(model)
    print(f"✅ Registered model: {out.name} v{out.version}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        raise
