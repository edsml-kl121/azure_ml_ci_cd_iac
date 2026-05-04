"""
Switches the default AML datastores (workspaceblobstore, workspacefilestore)
from AccountKey auth to credential-less (managed identity).

Required when the storage account has allowSharedKeyAccess: false.
"""
import argparse

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AzureBlobDatastore, AzureFileDatastore
from azure.identity import DefaultAzureCredential


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group",  required=True)
    parser.add_argument("--workspace-name",  required=True)
    args = parser.parse_args()

    ml_client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    for ds_name in ("workspaceblobstore", "workspacefilestore"):
        try:
            ds = ml_client.datastores.get(ds_name)
        except Exception as e:
            print(f"  {ds_name}: not found, skipping ({e})")
            continue

        # workspaceblobstore is always the default; workspacefilestore is not
        is_default = (ds_name == "workspaceblobstore")

        # Rebuild with no credentials (managed identity)
        if isinstance(ds, AzureBlobDatastore):
            updated = AzureBlobDatastore(
                name=ds.name,
                account_name=ds.account_name,
                container_name=ds.container_name,
                credentials=None,
                is_default=is_default,
            )
        elif isinstance(ds, AzureFileDatastore):
            updated = AzureFileDatastore(
                name=ds.name,
                account_name=ds.account_name,
                file_share_name=ds.file_share_name,
                credentials=None,
                is_default=is_default,
            )
        else:
            print(f"  {ds_name}: unexpected type {type(ds).__name__}, skipping")
            continue

        ml_client.datastores.create_or_update(updated)
        print(f"  {ds_name}: switched to managed-identity auth")


if __name__ == "__main__":
    main()
