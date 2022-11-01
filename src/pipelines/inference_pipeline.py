"""Definition of the Inference Pipeline"""
import json
from typing import cast

from zenml.logger import get_logger
from zenml.pipelines import pipeline

from src.util.settings import docker_settings

logger = get_logger(__name__)


@pipeline(enable_cache=True, settings={"docker": docker_settings})
def inference_pipeline(
    dynamic_importer,
    prediction_service_loader,
    predictor,
):
    """Imports the inference data and executes the model deployment service on it"""
    inference_data = dynamic_importer()
    model_deployment_service = prediction_service_loader()
    predictor(model_deployment_service, inference_data)
