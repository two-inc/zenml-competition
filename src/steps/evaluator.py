"""Evaluator step"""
import mlflow
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from zenml.client import Client
from zenml.logger import get_logger
from zenml.steps import step

from src.materializer.types import Classifier
from src.util import path
from src.util.tracking import get_classification_metrics

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def evaluator(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    model: HistGradientBoostingClassifier,
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
    mlflow.log_metrics(metric_results)

    path.METRICS_PATH.touch(exist_ok=True)
    with open(path.METRICS_PATH, "a") as f:
        f.write(f"{metric_results}")

    return metric_results
