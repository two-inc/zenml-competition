"""Evaluator step"""
import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin
from zenml.client import Client
from zenml.logger import get_logger
from zenml.steps import step

from src.util.tracking import experiment_tracker_name
from src.util.tracking import get_classification_metrics

logger = get_logger(__name__)


@step(enable_cache=False, experiment_tracker=experiment_tracker_name)
def evaluator(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    model: ClassifierMixin,
) -> dict[str, float]:
    """Evaluate a Classification Model

    Evalautes a classifier according to some classification metrics on a holdout set
    and records those metric results in the experiment tracker

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
    mlflow.log_metrics(metric_results)

    return metric_results
