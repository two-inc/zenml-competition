"""Trainer step"""
import lightgbm as lgbm
import pandas as pd
import mlflow
from zenml.steps import Output
from zenml.steps import step
from zenml.client import Client

from src.util import columns
from src.util.preprocess import get_column_indices
from src.util.preprocess import SEED
from src.util.tracking import LGBM_TRAIN_PARAMS


experiment_tracker = Client().active_stack.experiment_tracker


@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name
)
def trainer(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Output(model=lgbm.LGBMClassifier):
    """Trains a LightGBM Model"""
    model = lgbm.LGBMClassifier(
        **LGBM_TRAIN_PARAMS, random_state=SEED, n_jobs=-1
    )
    mlflow.log_param("model_type", model.__class__.__name__)
    mlflow.log_params(LGBM_TRAIN_PARAMS)

    model.fit(
        X_train,
        y_train,
        categorical_feature=get_column_indices(X_train, columns.CATEGORICAL),
    )

    return model
