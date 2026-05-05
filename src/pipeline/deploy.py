"""
Deploys the latest registered 'diabetes-classifier' model to an Azure ML
managed online endpoint.

Usage:
    python src/pipeline/deploy.py \
        --subscription-id <sub_id> \
        --resource-group  <rg> \
        --workspace-name  <ws_name> \
        --endpoint-name   diabetes-endpoint-dev   # or -prod
"""
import argparse
import json
import os
import urllib.request

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    ProbeSettings,
)
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import ClientAssertionCredential, DefaultAzureCredential


def _get_github_oidc_token() -> str:
    """Fetch a fresh GitHub OIDC assertion from the Actions token endpoint."""
    url = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
    request_token = os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]
    sep = "&" if "?" in url else "?"
    req = urllib.request.Request(
        f"{url}{sep}audience=api://AzureADTokenExchange",
        headers={"Authorization": f"bearer {request_token}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read())["value"]


def _build_credential():
    """Return the best available credential for the current environment.

    In GitHub Actions with OIDC (id-token: write), uses ClientAssertionCredential
    backed by the GitHub OIDC token endpoint.  The callback is invoked on every
    token refresh, so long-running pollers (deployment wait) never hit
    AADSTS700024 assertion-expiry errors.

    Locally, falls back to DefaultAzureCredential (Azure CLI, VS Code, etc.).
    """
    oidc_url = os.getenv("ACTIONS_ID_TOKEN_REQUEST_URL")
    if os.getenv("GITHUB_ACTIONS", "").lower() == "true" and oidc_url:
        print("Credential mode: github_oidc (ClientAssertionCredential)")
        return ClientAssertionCredential(
            tenant_id=os.environ["AZURE_TENANT_ID"],
            client_id=os.environ["AZURE_CLIENT_ID"],
            func=_get_github_oidc_token,
        )

    print("Credential mode: DefaultAzureCredential (local)")
    return DefaultAzureCredential(exclude_interactive_browser_credential=True)


def _print_deployment_logs(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str,
    lines: int = 300,
) -> None:
    try:
        logs = ml_client.online_deployments.get_logs(
            endpoint_name=endpoint_name,
            name=deployment_name,
            lines=lines,
        )
        print("\n===== Online deployment container logs (tail) =====")
        print(logs)
        print("===== End container logs =====\n")
    except Exception as log_err:  # noqa: BLE001
        print(f"Failed to fetch online deployment logs: {log_err}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group",  required=True)
    parser.add_argument("--workspace-name",  required=True)
    parser.add_argument("--endpoint-name",   required=True)
    args = parser.parse_args()

    ml_client = MLClient(
        _build_credential(),
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    # ── Create / update endpoint ──────────────────────────────────────────────
    endpoint = ManagedOnlineEndpoint(
        name=args.endpoint_name,
        description="Diabetes binary classification endpoint",
        auth_mode="key",
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint ready: {args.endpoint_name}")

    # ── Deploy latest model version ───────────────────────────────────────────
    model = ml_client.models.get("diabetes-classifier", label="latest")

    # AzureML's mlflow_score_script.py (entry script for no-code MLflow deployment)
    # unconditionally imports `from azureml.ai.monitoring import Collector`, but
    # MLflow autolog does not include azureml-ai-monitoring in the model's conda.yaml.
    # Specifying an explicit Environment here ensures it is always present.
    inference_env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file={
            "name": "sklearn-mlflow-inference",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.10",
                {"pip": [
                    "azureml-inference-server-http",
                    "azureml-ai-monitoring",
                    "mlflow==2.16.0",
                    "scikit-learn==1.4.2",
                    "pandas>=2.0",
                    "numpy>=1.26",
                    "joblib",
                ]},
            ],
        },
    )
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=args.endpoint_name,
        model=model,
        environment=inference_env,
        instance_type="Standard_DS3_v2",
        instance_count=1,
        liveness_probe=ProbeSettings(
            failure_threshold=30,
            success_threshold=1,
            period=100,
            initial_delay=500,
        ),
        readiness_probe=ProbeSettings(
            failure_threshold=30,
            success_threshold=1,
            period=100,
            initial_delay=500,
        ),
    )
    try:
        ml_client.online_deployments.begin_create_or_update(deployment).result()
    except ClientAuthenticationError as auth_err:
        print(
            "Authentication failed while polling deployment state. "
            "If running in GitHub Actions with OIDC, refresh azure/login "
            "immediately before this step."
        )
        _print_deployment_logs(ml_client, args.endpoint_name, "blue")
        raise auth_err
    except Exception as deploy_err:  # noqa: BLE001
        print(
            "Deployment failed. Fetching online deployment logs to help "
            "diagnose liveness/readiness probe and startup issues."
        )
        _print_deployment_logs(ml_client, args.endpoint_name, "blue")
        raise deploy_err

    # Route 100 % of traffic to this deployment
    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f"Deployed → {args.endpoint_name}/blue  (traffic 100 %)")


if __name__ == "__main__":
    main()
