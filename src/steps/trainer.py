"""Trainer step"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn import preprocessing
from zenml.steps import step, Output
from zenml.pipelines import pipeline
from zenml.logger import get_logger
import lightgbm as lgbm
from src.util.env import CATEGORICAL_FEATURES


@step()
def trainer(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Output(model=ClassifierMixin):
    """Training a LightGBM model."""

    lbl = preprocessing.LabelEncoder()
    for categorical_feature in CATEGORICAL_FEATURES:
        X_train[categorical_feature] = lbl.fit_transform(
            X_train[categorical_feature].astype(str)
        )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2
    )

    model = lgbm.LGBMClassifier(is_unbalance=True, random_state=0).fit(
        X_train, y_train, categorical_feature=CATEGORICAL_FEATURES
    )

    return model
