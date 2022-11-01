"""Trainer step"""
import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml.client import Client
from zenml.steps import Output
from zenml.steps import step

from src.util import columns
from src.util.preprocess import SEED

experiment_tracker = Client().active_stack.experiment_tracker


@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name
)
def trainer(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Output(model=ClassifierMixin):
    """Trains a GBM Model"""
    ## Importing within step to force order of imports, necessary for Import
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier
    params = {"max_leaf_nodes": None, "max_depth": None, "random_state": SEED}

    model = HistGradientBoostingClassifier(**params)
    mlflow.log_param("model_type", model.__class__.__name__)
    mlflow.log_params(params)

    model.fit(X_train, y_train)

    return model
