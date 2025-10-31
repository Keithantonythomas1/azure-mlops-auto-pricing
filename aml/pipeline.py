# pipelines/pipeline.yml  — Azure ML CLI v2 pipeline job
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
type: pipeline
name: auto_pricing_pipeline
display_name: auto_pricing_pipeline
experiment_name: auto-pricing-e2e

# Pipeline-level inputs
inputs:
  # If your data asset is a versioned Azure ML asset, you can wire it here
  # or keep setting it from GitHub Actions with --set inputs.data='@azureml:UsedCars:1'
  data:
    type: uri_folder
    path: azureml:UsedCars:1

# Optional: collect outputs at the pipeline level if you want to bind between steps
outputs: {}

# ---- Jobs (replace component file paths with yours) ----
jobs:

  train:
    type: command
    component: file:./components/train.yml
    # ✅ Explicit compute, replacing any `${{ parent.compute }}`
    compute: azureml:Keith-Compute
    inputs:
      # ✅ Valid binding: parent.inputs
      data: ${{ parent.inputs.data }}
    outputs:
      model_out:
        type: mlflow_model

  evaluate:
    type: command
    component: file:./components/evaluate.yml
    compute: azureml:Keith-Compute
    inputs:
      model_in: ${{ parent.jobs.train.outputs.model_out }}
      data: ${{ parent.inputs.data }}

  register:
    type: command
    component: file:./components/register.yml
    compute: azureml:Keith-Compute
    inputs:
      model_in: ${{ parent.jobs.train.outputs.model_out }}

