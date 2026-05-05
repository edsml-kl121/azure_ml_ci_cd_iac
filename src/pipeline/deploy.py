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

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    ProbeSettings,
)
from azure.identity import DefaultAzureCredential


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group",  required=True)
    parser.add_argument("--workspace-name",  required=True)
    parser.add_argument("--endpoint-name",   required=True)
    args = parser.parse_args()

    ml_client = MLClient(
        DefaultAzureCredential(),
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
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # Route 100 % of traffic to this deployment
    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f"Deployed → {args.endpoint_name}/blue  (traffic 100 %)")


if __name__ == "__main__":
    main()
