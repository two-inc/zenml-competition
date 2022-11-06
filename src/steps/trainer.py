"""Trainer step"""
import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin
from zenml.steps import Output
from zenml.steps import step

from src.util.tracking import experiment_tracker_name
from src.util.tracking import HGBM_PARAMS

from sklearn.experimental import enable_hist_gradient_boosting  # noreorder
from sklearn.ensemble import HistGradientBoostingClassifier


@step(enable_cache=False, experiment_tracker=experiment_tracker_name)
def trainer(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Output(model=ClassifierMixin):
    """Trains a HistGradientBoostingClassifier

    Args:
        X_train (pd.DataFrame): Training Data Features
        y_train (pd.Series): Training Data Target

    Returns:
        ClassifierMixin: Trained Classifier
    """
    model = HistGradientBoostingClassifier(**HGBM_PARAMS)
    mlflow.log_param("model_type", model.__class__.__name__)
    mlflow.log_params(HGBM_PARAMS)

    model.fit(X_train, y_train)

    return model
