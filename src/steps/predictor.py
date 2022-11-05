"""Definition of the Predictor Step"""
import json

import numpy as np
import pandas as pd
from zenml.integrations.constants import SELDON
from zenml.integrations.constants import SKLEARN
from zenml.integrations.constants import XGBOOST
from zenml.integrations.seldon.model_deployers import SeldonModelDeployer
from zenml.integrations.seldon.services import SeldonDeploymentService
from zenml.logger import get_logger
from zenml.steps import Output
from zenml.steps import step

from src.util import columns

logger = get_logger(__name__)

INFERENCE_COLUMNS = columns.NUMERICAL


@step
def predictor(
    service: SeldonDeploymentService,
    data: pd.DataFrame,
) -> Output(predictions=np.ndarray):
    """Run an inference request against a prediction service"""

    service.start(timeout=120)  # should be a NOP if already started
    data = data.to_json(orient="split")
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    df = pd.DataFrame(data["data"], columns=INFERENCE_COLUMNS)
    data = np.array(df)
    predictions = service.predict(data)
    predictions = predictions.argmax(axis=-1)
    logger.info("Prediction: ", predictions)
    return predictions
