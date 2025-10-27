# MLOps on Azure — Streamlining Auto Vehicle Pricing

Automate vehicle price prediction with an Azure Machine Learning (Azure ML) **end-to-end MLOps pipeline**: data prep, model training, tuning with MLflow logging, registration, and CI/CD via **GitHub Actions**.

> **Project Goal**: Improve pricing accuracy and operational efficiency for a Las Vegas auto dealership by replacing manual, disconnected pricing steps with a reproducible, scalable pipeline.

---

## Repo Structure

```
.
├── aml/
│   └── pipeline.py                # Azure ML v2 pipeline: prep → train → tune → register
├── configs/
│   └── aml_config.json            # (optional) Workspace/compute defaults
├── env/
│   └── conda.yml                  # Environment for Azure ML jobs
├── notebooks/
│   └── Week-17_Project_FullCode_Notebook.ipynb
├── scripts/
│   ├── prep.py                    # Data cleaning & feature engineering
│   ├── train.py                   # Baseline model training + MLflow logging
│   ├── tune.py                    # Hyperparameter tuning + MLflow logging
│   └── register_model.py          # Register best model in Azure ML registry
├── .github/workflows/
│   └── ci-cd.yml                  # CI/CD workflow (submit AML pipeline on push/PR)
├── data/                          # (optional) Local sample data
├── .gitignore
└── README.md
```

---

## Quick Start

### 1) Create Azure resources
- Azure ML workspace
- Compute cluster (e.g., `cpu-cluster`)
- (Optional) Storage account & data in blob

### 2) Create a Service Principal (SP) and capture its JSON
```bash
az ad sp create-for-rbac   --name "gh-mlops-auto-pricing-sp"   --role contributor   --scopes /subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.MachineLearningServices/workspaces/<WS_NAME>
```

Copy the JSON output; you'll paste it into a GitHub **Actions secret** named **`AZURE_CREDENTIALS`**.

### 3) Create a new GitHub repo and push this code
```bash
git init
git remote add origin https://github.com/<you>/azure-mlops-auto-pricing.git
git add .
git commit -m "Init: Azure ML MLOps starter"
git branch -M main
git push -u origin main
```

### 4) Add GitHub Actions **Secrets** (Repo → Settings → Secrets and variables → Actions)
- `AZURE_CREDENTIALS` : the JSON from `az ad sp create-for-rbac`
- `AML_SUBSCRIPTION_ID`
- `AML_RESOURCE_GROUP`
- `AML_WORKSPACE_NAME`
- `AML_COMPUTE_NAME` (e.g., `cpu-cluster`)
- `AML_LOCATION` (e.g., `eastus`)

### 5) Trigger CI/CD
Commit any change (e.g., tweak `scripts/train.py`) and push.  
The workflow **builds environment**, **logs in to Azure**, and **submits the pipeline** defined in `aml/pipeline.py`.  
You can monitor runs in **Azure ML Studio → Jobs**.

### 6) Notebook
At the top of `notebooks/Week-17_Project_FullCode_Notebook.ipynb`, add a Markdown cell linking to your **public GitHub repo URL**.

---

## Business Output & Rubric Coverage

- **End-to-end Pipeline (17 pts):** Components for prep, train, tune (MLflow), register; executed as a single Azure ML pipeline.
- **GitHub Actions (18 pts):** Python scripts + YAML workflow; hierarchical structure.
- **Execute Actions (5 pts):** Secrets + run on push/PR.
- **Validate CI/CD (5 pts):** Edit `prep.py` or `train.py`, push, observe re-run and new model registration.
- **Sample Output (5 pts):** Upload screenshots of GitHub Actions run and Azure ML pipeline run.
- **Insights (2 pts):** See `train.py`/`tune.py` MLflow logs and summarize in your notebook.
- **Presentation/Quality (8 pts):** Organized repo; commented code; clean flow.

---

## Data Schema (expected columns)

- `Segment` (luxury / non-luxury)
- `Kilometers_Driven`
- `Mileage` (km/l)
- `Engine` (cc)
- `Power` (BHP)
- `Seats`
- `Price` (target, in lakhs)

You can place a CSV into `data/vehicles.csv` locally, or read from Blob/Datastore in Azure ML.

---

## Your Azure settings (pre-filled)
- **Subscription ID**: `f7583458-6f5b-4491-add9-f827568d2957`
- **Resource Group**: `MSFT-AI-Class`
- **Workspace**: `MyFirstWorkSpace`
- **Location**: `eastus`
- **Compute**: `Keith-Compute`

> The pipeline (`aml/pipeline.py`) targets compute **Keith-Compute** and region **eastus**.

### If you prefer GitHub Secrets
You can still set repo secrets and the workflow will use them if present. Otherwise it reads `configs/aml_config.json`.---

## Data Input (bound to Azure ML Data asset)
This pipeline reads the dataset from the **workspace data asset**:

- **Name**: `UsedCars`
- **Version**: `1`
- **Datastore**: `workspaceblobstore`
- **Path**: `UI/.../used_cars.csv`

In `aml/pipeline.py`, the prep step declares:
```python
Input(type="uri_file", path="azureml:UsedCars:1")
```
and passes it to `prep.py` via the `DATA_PATH` environment variable. No local CSV is required.
