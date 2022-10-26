"""Definition of the Deployment & Pipeline"""
import json
from typing import cast

import numpy as np
import pandas as pd
from zenml.integrations.constants import LIGHTGBM
from zenml.integrations.constants import SELDON
from zenml.integrations.constants import SKLEARN
from zenml.integrations.constants import XGBOOST
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


class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""

    min_accuracy: float


@step(enable_cache=False)
def dynamic_importer() -> Output(data=pd.DataFrame):
    """Downloads the latest data from a mock API."""
    data = pd.read_csv(TRAIN_DATA_PATH)
    return data


@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> np.bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return accuracy > config.min_accuracy


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


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    ingest_data,
    encode_cat_cols,
    drop_cols,
    data_splitter,
    model_trainer,
    evaluator,
    deployment_trigger,
    model_deployer,
):
    """Trains a Model and deploys it conditional on the successful execution of the deployment trigger"""
    customer_churn_df = ingest_data()
    customer_churn_df = encode_cat_cols(customer_churn_df)
    customer_churn_df = drop_cols(customer_churn_df)
    train, test = data_splitter(customer_churn_df)
    model = model_trainer(train)
    accuracy = evaluator(model, test)
    deployment_decision = deployment_trigger(accuracy=accuracy)
    model_deployer(deployment_decision, model)
