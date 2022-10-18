"""Definition of the Inference Pipeline"""
import json
from typing import cast

from util.settings import docker_settings
from zenml.logger import get_logger
from zenml.pipelines import pipeline

logger = get_logger(__name__)


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(
    dynamic_importer,
    prediction_service_loader,
    predictor,
):
    """Imports the inference data and executes the model deployment service on it"""
    inference_data = dynamic_importer()
    model_deployment_service = prediction_service_loader()
    predictor(model_deployment_service, inference_data)
