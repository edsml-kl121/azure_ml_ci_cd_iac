"""
Submits the Azure ML pipeline (data_prep → train) and registers the model.

Usage (run locally or from GitHub Actions):
    python src/pipeline/pipeline.py \
        --subscription-id <sub_id> \
        --resource-group  <rg> \
        --workspace-name  <ws_name>
"""
import argparse
from pathlib import Path

from azure.ai.ml import MLClient, load_component
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Model, ManagedIdentityConfiguration
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

COMPONENTS = Path(__file__).parent / "components"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group",  required=True)
    parser.add_argument("--workspace-name",  required=True)
    parser.add_argument("--identity-client-id", required=True,
                        help="Client ID of the UAMI attached to the workspace")
    args = parser.parse_args()

    ml_client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    # ── Load components from local YAML specs ─────────────────────────────────
    data_prep = load_component(source=COMPONENTS / "data_prep" / "component.yml")
    train     = load_component(source=COMPONENTS / "train"     / "component.yml")

    # ── Define pipeline ───────────────────────────────────────────────────────
    @pipeline(
        name="diabetes_classification",
        description="Data prep → train RandomForest on diabetes dataset",
    )
    def diabetes_pipeline():
        prep = data_prep()
        fit  = train(
            train_data=prep.outputs.train_data,
            test_data=prep.outputs.test_data,
        )
        return {"model_output": fit.outputs.model_output}

    job = diabetes_pipeline()
    job.settings.default_compute = "serverless"
    # Use the UAMI attached to the workspace so serverless nodes can access storage
    job.identity = ManagedIdentityConfiguration(client_id=args.identity_client_id)

    # ── Submit and stream logs ────────────────────────────────────────────────
    submitted = ml_client.jobs.create_or_update(
        job, experiment_name="diabetes-classification"
    )
    print(f"Pipeline submitted → {submitted.name}")
    print(f"Studio URL:          {submitted.studio_url}")
    ml_client.jobs.stream(submitted.name)         # blocks until complete

    # ── Register model ────────────────────────────────────────────────────────
    model = Model(
        path=f"azureml://jobs/{submitted.name}/outputs/model_output",
        name="diabetes-classifier",
        type=AssetTypes.MLFLOW_MODEL,
        description="Binary classifier: high vs low diabetes disease progression",
    )
    registered = ml_client.models.create_or_update(model)
    print(f"Model registered: {registered.name}  version: {registered.version}")


if __name__ == "__main__":
    main()
