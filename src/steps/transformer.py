from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml.steps import Output
from zenml.steps import step

import pdb

@step()
def transformer(
    data: pd.DataFrame
    ) -> Output(
    x_train=np.ndarray,
    x_test=np.ndarray,
    y_train=np.ndarray,
    y_test=np.ndarray,
):
    """Divides data into Train and Test sets.

    Args:
        data (pd.DataFrame): Raw Input DataFrame for training the Fraud model

    Returns:
        Tuple[np.ndarray]: Train and Test sets
    """
    return split_data(data)

def split_data(data: pd.DataFrame) -> Output(
    X_train = np.ndarray, X_test = np.ndarray, y_train = np.ndarray, y_true = np.ndarray
):
    """Splits a DataFrame into Train and Test sets

    Args:
        data (pd.DataFrame): Data to be split into Train and Test sets

    Returns:
        Tuple[np.ndarray]: Train and Test sets
    """
    dataframe = data.copy()

    X = data.drop('fraud', axis = 1).values
    y = data['fraud'].values
    X_train, X_test, y_train, y_true = train_test_split(X, y, test_size = 0.2, random_state=42)

    return X_train, X_test, y_train, y_true