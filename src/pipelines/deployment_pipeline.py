"""Definition of the Deployment & Pipeline"""
import json
from typing import cast

import numpy as np
import pandas as pd
from zenml.integrations.seldon.model_deployers import SeldonModelDeployer
from zenml.integrations.seldon.services import SeldonDeploymentService
from zenml.logger import get_logger
from zenml.pipelines import pipeline
from zenml.steps import BaseParameters
from zenml.steps import Output
from zenml.steps import step

from src.util.path import TRAIN_DATA_PATH
from src.util.settings import docker_settings

logger = get_logger(__name__)


@step(enable_cache=False)
def dynamic_importer() -> Output(data=pd.DataFrame):
    """Downloads the latest data from a mock API."""
    data = pd.read_csv(TRAIN_DATA_PATH)
    return data


class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""

    min_f1_score: float


@step(enable_cache=False)
def deployment_trigger(
    metrics: dict[str, str],
    config: DeploymentTriggerConfig,
) -> np.bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    f1_score = metrics.get("F1 Score", config.min_f1_score)
    return f1_score > config.min_f1_score


class SeldonDeploymentLoaderStepConfig(BaseParameters):
    """Seldon deployment loader configuration
    Attributes:
        pipeline_name: name of the pipeline that deployed the Seldon prediction
            server
        step_name: the name of the step that deployed the Seldon prediction
            server
        model_name: the name of the model that was deployed
    """

    pipeline_name: str
    step_name: str
    model_name: str


@pipeline(
    name="continuous_deployment_pipeline_3",
    enable_cache=True,
    settings={"docker": docker_settings},
)
def continuous_deployment_pipeline(
    importer,
    transformer,
    trainer,
    evaluator,
    deployment_trigger,
    model_deployer,
):
    """Trains a Model and deploys it conditional on the successful execution of the deployment trigger"""
    df = importer()
    X_train, X_test, y_train, y_test = transformer(df)
    model = trainer(X_train, y_train)
    metrics = evaluator(X_test, y_test, model)
    deployment_decision = deployment_trigger(metrics=metrics)
    model_deployer(deployment_decision, model)
