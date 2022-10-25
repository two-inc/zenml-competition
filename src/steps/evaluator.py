"""Evaluator step"""
from typing import Protocol

import lightgbm as lgbm
import pandas as pd
from zenml.logger import get_logger
from zenml.steps import step

from src.util import path
from src.util.tracking import get_classification_metrics

logger = get_logger(__name__)


class Classifier(Protocol):
    """Classifier Interface"""

    def predict(*args, **kwargs):
        ...

    def predict_proba(*args, **kwargs):
        ...


@step(enable_cache=False)
def evaluator(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    model: Classifier,
) -> dict[str, float]:
    """Evaluate a Classification Model

    Args:
        X_test: DataFrame with eval feature data.
        y_test: DataFrame with eval target data.
        model: Trained Classifier.

    Returns:
        dict[str,float]: Metric Results
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metric_results = get_classification_metrics(
        true=y_test, pred=y_pred, pred_proba=y_pred_proba
    )

    logger.info(f"Metric Values:\n{metric_results}")

    with open(path.METRICS_PATH, "a") as f:
        f.write(f"{metric_results}")

    return metric_results
