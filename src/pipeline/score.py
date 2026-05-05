"""
Custom scoring script for the diabetes-classifier MLflow model.

AzureML calls init() once at container startup and run() for every
inference request. Using mlflow.pyfunc avoids the azureml-ai-monitoring
dependency that the AzureML-generated mlflow_score_script.py requires.
"""
import json
import logging
import os

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


def init():
    global model
    model_dir = os.environ["AZUREML_MODEL_DIR"]
    model = mlflow.pyfunc.load_model(model_dir)
    logger.info("Model loaded from %s", model_dir)


def run(raw_data):
    data = json.loads(raw_data)
    # Support AzureML standard payload: {"input_data": {"columns": [...], "data": [...]}}
    if isinstance(data, dict) and "input_data" in data:
        payload = data["input_data"]
        df = pd.DataFrame(payload["data"], columns=payload["columns"])
    else:
        df = pd.DataFrame(data)
    predictions = model.predict(df)
    return predictions.tolist()
