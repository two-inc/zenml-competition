"""Transformer step"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml.steps import Output
from zenml.steps import step


@step()
def transformer(
    data: pd.DataFrame,
) -> Output(
    x_train=pd.DataFrame,
    x_test=pd.DataFrame,
    y_train=pd.Series,
    y_test=pd.Series,
):
    """Divides data into Train and Test sets.

    Args:
        data (pd.DataFrame): Raw Input DataFrame for training the Fraud model

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train and Test sets
    """
    return split_data(data)


def split_data(
    data: pd.DataFrame,
) -> Output(
    X_train=pd.DataFrame,
    X_test=pd.DataFrame,
    y_train=pd.Series,
    y_test=pd.Series,
):
    """Splits a DataFrame into Train and Test sets

    Args:
        data (pd.DataFrame): Data to be split into Train and Test sets

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train and Test sets
    """
    dataframe = data.copy()

    X = data.drop("fraud", axis=1)
    y = data["fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
