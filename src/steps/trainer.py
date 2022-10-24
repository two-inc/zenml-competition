"""Trainer step"""
import pandas as pd
from zenml.steps import step, Output
import lightgbm as lgbm
from util.preprocess import SEED
from util.tracking import LGBM_TRAIN_PARAMS
from util.preprocess import get_column_indices
from util import columns

@step()
def trainer(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Output(model=lgbm.LGBMClassifier):
    """Trains a LightGBM Model"""
    model = lgbm.LGBMClassifier(**LGBM_TRAIN_PARAMS, random_state=SEED, n_jobs=-1)

    model.fit(
        X_train, 
        y_train, 
        categorical_feature=get_column_indices(
            X_train,
            columns.CATEGORICAL
        )
    )

    return model
