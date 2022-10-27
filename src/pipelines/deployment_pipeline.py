"""Definition of the Deployment & Pipeline"""
import json
from typing import cast

import numpy as np
import pandas as pd
from zenml.integrations.constants import LIGHTGBM
from zenml.integrations.constants import SELDON
from zenml.integrations.constants import SKLEARN
from zenml.integrations.constants import XGBOOST
from zenml.integrations.great_expectations.steps import (
    great_expectations_validator_step,
)
from zenml.integrations.great_expectations.steps import (
    GreatExpectationsValidatorParameters,
)
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


# instantiate a builtin Great Expectations data validation step
ge_validator_params = GreatExpectationsValidatorParameters(
    expectation_suite_name="",
    data_asset_name="breast_cancer_test_df",
)
ge_validator_step = great_expectations_validator_step(
    step_name="ge_validator_step",
    params=ge_validator_params,
)


@pipeline(
    name="continuous_deployment_pipeline_2",
    enable_cache=True,
    settings={"docker": docker_settings},
)
def continuous_deployment_pipeline(
    importer,
    validator,
    transformer,
    trainer,
    evaluator,
    deployment_trigger,
    model_deployer,
):
    """Trains a Model and deploys it conditional on the successful execution of the deployment trigger"""
    df, validate_data = importer()
    _ = validator(df, validate_data)
    X_train, X_test, y_train, y_test = transformer(df)
    model = trainer(X_train, y_train)
    metrics = evaluator(X_test, y_test, model)
    deployment_decision = deployment_trigger(metrics=metrics)
    model_deployer(deployment_decision, model)
