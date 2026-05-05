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
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    ProbeSettings,
)
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential


def _build_credential() -> DefaultAzureCredential:
    # In CI, only exclude AzureCliCredential when workload identity is fully
    # configured. Some runners do not expose a federated token file to SDKs,
    # and in that case CLI fallback is required.
    in_github_actions = os.getenv("GITHUB_ACTIONS", "").lower() == "true"
    has_workload_identity = all(
        os.getenv(var)
        for var in ("AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_FEDERATED_TOKEN_FILE")
    )
    exclude_cli = in_github_actions and has_workload_identity

    if in_github_actions:
        mode = "workload_identity" if has_workload_identity else "azure_cli_fallback"
        print(f"Credential mode: {mode}")

    return DefaultAzureCredential(
        exclude_cli_credential=exclude_cli,
        exclude_interactive_browser_credential=True,
    )


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

    # Do NOT specify a custom environment for MLflow models.
    # AML automatically uses its curated MLflow inference environment which:
    #   - includes azureml-inference-server-http and azureml-ai-monitoring
    #   - installs the model's bundled conda.yaml (scikit-learn, mlflow, etc.)
    # Specifying an explicit conda env on top of the training base image causes
    # package conflicts that crash the container on startup.
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=args.endpoint_name,
        model=model,
        instance_type="Standard_DS3_v2",
        instance_count=1,
        # Give the container enough time to install the conda env on first start
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
