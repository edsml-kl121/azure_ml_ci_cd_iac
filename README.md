# Azure ML CI/CD with Bicep

A minimal end-to-end MLOps example that shows **IaC + CI/CD** using Azure Machine Learning, Bicep, and GitHub Actions.

The model: binary classification on the sklearn diabetes dataset (high vs. low disease progression).

---

## Architecture

```
PR → CI (validate Bicep)
          │
merge to main
          │
          ▼
    ┌─────────────────────────────┐
    │  Job 1 – Dev  (automatic)  │
    │  1. Deploy infra (Bicep)   │
    │  2. AML pipeline           │
    │     data_prep → train      │
    │  3. Register model         │
    │  4. Deploy dev endpoint    │
    └─────────────┬───────────────┘
                  │
        ⏸ Manual approval gate
          (GitHub Environment)
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Job 2 – Prod (gated)      │
    │  1. Deploy infra (Bicep)   │
    │  2. AML pipeline           │
    │     data_prep → train      │
    │  3. Register model         │
    │  4. Deploy prod endpoint   │
    └─────────────────────────────┘
```

Two isolated Azure resource groups, one AML workspace each:
| Environment | Resource Group | AML Workspace |
|---|---|---|
| Dev | `rg-aml-dev` | `aml-dev-workspace` |
| Prod | `rg-aml-prod` | `aml-prod-workspace` |

Each workspace includes: Storage Account · Key Vault · Application Insights · Container Registry.

---

## Repository layout

```
├── .github/workflows/
│   ├── ci.yml           # PR: lint & validate Bicep
│   └── cd.yml           # push to main: Dev deploy → gated → Prod deploy
├── infra/
│   ├── main.bicep       # All Azure ML infrastructure
│   ├── dev.bicepparam
│   └── prod.bicepparam
├── src/pipeline/
│   ├── pipeline.py      # Submit AML pipeline + register model
│   ├── deploy.py        # Create managed online endpoint + deploy model
│   └── components/
│       ├── conda.yml            # Shared conda environment
│       ├── data_prep/
│       │   ├── component.yml
│       │   └── data_prep.py     # Load sklearn diabetes → train/test CSVs
│       └── train/
│           ├── component.yml
│           └── train.py         # RandomForest + MLflow autolog
└── requirements.txt
```

---

## Prerequisites

- Azure subscription with Contributor access
- Azure CLI (`az`) ≥ 2.57
- Python 3.10+
- GitHub repository (fork or create from this code)

---

## One-time setup

### 1 – Create a service principal with OIDC (federated identity)

```bash
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
APP_ID=$(az ad app create --display-name "aml-cicd" --query appId -o tsv)
az ad sp create --id "$APP_ID"

# Contributor on the subscription so it can create resource groups
az role assignment create \
  --role Contributor \
  --assignee "$APP_ID" \
  --scope "/subscriptions/$SUBSCRIPTION_ID"

# Federated credential – trusts the main branch of your repo
az ad app federated-credential create --id "$APP_ID" --parameters '{
  "name": "github-main",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:edsml-kl121/azure_ml_ci_cd_iac:ref:refs/heads/main",
  "audiences": ["api://AzureADTokenExchange"]
}'

# Also add a credential for pull requests (used by ci.yml)
az ad app federated-credential create --id "$APP_ID" --parameters '{
  "name": "github-pr",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:edsml-kl121/azure_ml_ci_cd_iac:pull_request",
  "audiences": ["api://AzureADTokenExchange"]
}'

TENANT_ID=$(az account show --query tenantId -o tsv)
echo "AZURE_CLIENT_ID=$APP_ID"
echo "AZURE_TENANT_ID=$TENANT_ID"
echo "AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID"
```

### 2 – Add GitHub Secrets

In your repo: **Settings → Secrets and variables → Actions → New repository secret**

| Secret | Value |
|---|---|
| `AZURE_CLIENT_ID` | App (client) ID from step 1 |
| `AZURE_TENANT_ID` | Tenant ID from step 1 |
| `AZURE_SUBSCRIPTION_ID` | Subscription ID from step 1 |

### 3 – Create the Production environment with approval gate

**Settings → Environments → New environment** → name it `production`

Enable **Required reviewers** and add yourself (or your team).

This is the gate: GitHub will pause the `deploy-prod` job and send a notification to reviewers before continuing.

---

## How it works

### CI (`ci.yml`) – on every Pull Request

1. `az bicep build` compiles the template and catches syntax errors.
2. `az deployment group validate` does a dry-run against Azure (no resources created).

### CD (`cd.yml`) – on every push to `main`

**Dev job (automatic)**
1. `az deployment group create` deploys `infra/main.bicep` with dev parameters.
   Bicep is idempotent — re-running it only updates what changed.
2. `pipeline.py` submits an AML pipeline with two components:
   - `data_prep` – loads sklearn diabetes data, creates binary label, splits 80/20.
   - `train` – fits a `RandomForestClassifier`, logs accuracy + AUC via MLflow.
3. The trained model is registered in the Dev AML workspace.
4. `deploy.py` creates (or updates) a managed online endpoint and routes 100 % traffic to the `blue` deployment.

**Production job (gated)**

Identical steps, but the job is blocked by the GitHub Environment protection rule until an approver clicks **Approve** in the GitHub UI. After approval the same code runs against the Prod workspace.

### AML pipeline components

Each component is a self-contained Python script + `component.yml` spec.
They run on **serverless compute** — no cluster to provision or pay for at rest.

```
data_prep.py  →  train.csv / test.csv  →  train.py  →  model (MLflow)
```

The model is logged with `mlflow.sklearn.autolog()` and saved in MLflow format,
which lets AML deploy it to a managed endpoint with zero scoring-script boilerplate.

---

## Run locally (optional)

```bash
pip install -r requirements.txt

# Deploy dev infra
az deployment group create \
  --resource-group rg-aml-dev \
  --template-file  infra/main.bicep \
  --parameters     infra/dev.bicepparam

# Run the pipeline
python src/pipeline/pipeline.py \
  --subscription-id <sub_id> \
  --resource-group  rg-aml-dev \
  --workspace-name  aml-dev-workspace

# Deploy the endpoint
python src/pipeline/deploy.py \
  --subscription-id <sub_id> \
  --resource-group  rg-aml-dev \
  --workspace-name  aml-dev-workspace \
  --endpoint-name   diabetes-endpoint-dev
```

---

## Key concepts illustrated

| Concept | Where |
|---|---|
| **IaC** | `infra/main.bicep` — all Azure resources declared, repeatable |
| **CI** | `ci.yml` — Bicep validated on every PR, no manual checks needed |
| **CD** | `cd.yml` — infra + model deployed automatically on merge |
| **Gated approval** | `environment: production` in `cd.yml` + GitHub Environment reviewers |
| **AML pipeline** | `pipeline.py` — data_prep → train as reusable components |
| **MLflow tracking** | `train.py` — metrics and model logged automatically |
| **Managed endpoint** | `deploy.py` — one-line REST endpoint, no server management |
