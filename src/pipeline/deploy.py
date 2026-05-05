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
    Environment,
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

    # Explicit environment so the inference server package is always present.
    # Without this AML uses the MLflow model's bundled conda.yaml which lacks
    # azureml-inference-server-http, causing the container to 502 on startup.
    inference_env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file={
            "name": "diabetes-inference-env",
            "channels": ["conda-forge", "defaults"],
            "dependencies": [
                "python=3.10",
                {"pip": [
                    "setuptools>=69,<70",
                    "azureml-inference-server-http",
                    "azureml-ai-monitoring",   # provides azureml.ai.monitoring used by mlflow_score_script.py
                    "azureml-mlflow==1.55.0",
                    "mlflow==2.16.0",
                    "scikit-learn==1.4.2",
                    "pandas==2.2.2",
                    "numpy==1.26.4",
                    "joblib==1.4.2",
                ]},
            ],
        },
    )

    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=args.endpoint_name,
        model=model,
        environment=inference_env,
        instance_type="Standard_DS2_v2",
        instance_count=1,
        # Give the container enough time to install conda env on first start
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
