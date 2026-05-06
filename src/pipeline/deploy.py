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
import time
import urllib.request
import uuid

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    ProbeSettings,
)
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import ClientAssertionCredential, DefaultAzureCredential
from azure.mgmt.authorization import AuthorizationManagementClient
from azure.mgmt.authorization.models import RoleAssignmentCreateParameters


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

    credential = _build_credential()
    ml_client = MLClient(
        credential,
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

    # ── Grant endpoint identity Storage Blob Data Reader ─────────────────────
    # The storage-initializer sidecar downloads the model from the workspace
    # storage account using the endpoint's system-assigned managed identity.
    # With allowSharedKeyAccess=false, SAS-token fallback is blocked, so an
    # explicit RBAC assignment is required.
    endpoint_obj = ml_client.online_endpoints.get(args.endpoint_name)
    endpoint_principal_id = endpoint_obj.identity.principal_id
    workspace_obj = ml_client.workspaces.get(args.workspace_name)
    storage_id = workspace_obj.storage_account

    auth_client = AuthorizationManagementClient(credential, args.subscription_id)
    # Storage Blob Data Reader role definition ID (built-in, stable)
    _READER_ROLE = "2a2b9908-6ea1-4ae2-8e65-a410df84e7d2"
    role_def_id = (
        f"/subscriptions/{args.subscription_id}"
        f"/providers/Microsoft.Authorization/roleDefinitions/{_READER_ROLE}"
    )
    # Deterministic name so re-runs are idempotent
    assignment_name = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"{storage_id}/{endpoint_principal_id}/{_READER_ROLE}")
    )
    try:
        auth_client.role_assignments.create(
            scope=storage_id,
            role_assignment_name=assignment_name,
            parameters=RoleAssignmentCreateParameters(
                role_definition_id=role_def_id,
                principal_id=endpoint_principal_id,
                principal_type="ServicePrincipal",
            ),
        )
        print(f"Granted Storage Blob Data Reader to endpoint identity ({endpoint_principal_id}).")
        print("Waiting 60s for RBAC propagation...")
        time.sleep(60)
    except Exception as rbac_err:  # noqa: BLE001
        if "RoleAssignmentExists" in str(rbac_err) or "Conflict" in str(rbac_err):
            print("Storage Blob Data Reader already assigned (idempotent).")
        else:
            raise

    # ── Deploy latest model version ───────────────────────────────────────────
    model = ml_client.models.get("diabetes-classifier", label="latest")

    # Specifying a curated AzureML environment that includes azureml-ai-monitoring.
    # Using a curated env (azureml:// reference) avoids any local blob upload —
    # the image is already registered in the workspace.  Omitting code_configuration
    # keeps the MLflow no-code deployment path active, which picks up the model's
    # MLmodel file automatically and routes requests through mlflow_score_script.py.
    # azureml-ai-monitoring is present in the curated env so that script won't crash.
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=args.endpoint_name,
        model=model,
        environment="azureml://registries/azureml/environments/sklearn-1.5/labels/latest",
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
