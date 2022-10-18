"""Definition of the Predictor Step"""
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

logger = get_logger(__name__)


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
    columns_for_df = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    data = np.array(df)
    predictions = service.predict(data)
    predictions = predictions.argmax(axis=-1)
    logger.info("Prediction: ", predictions)
    return predictions
