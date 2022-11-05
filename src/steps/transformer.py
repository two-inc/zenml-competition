"""Transformer step"""
import pandas as pd
from zenml.steps import Output
from zenml.steps import step

from src.util import columns
from src.util.preprocess import get_preprocessed_data
from src.util.preprocess import train_test_split_by_step


@step(enable_cache=True)
def transformer(
    data: pd.DataFrame,
) -> Output(
    X_train=pd.DataFrame,
    X_valid=pd.DataFrame,
    y_train=pd.Series,
    y_valid=pd.Series,
):
    """Applies preprocessing, feature engineering & data splitting logic to dataset

    Preprocessing:
        - Category Encoding
    Feature Engineering:
        - Moving Averages, Standard Deviations & Max of Amount Column
        - Customer/Merchant Transaction Numbers
    Data Splitting:
        - Splitting the Data via the Step column
        - First 80% of Days are used for training, rest for evaluation

    Args:
        data (pd.DataFrame): Raw Input DataFrame

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train and Test sets
    """
    preprocesed_data = get_preprocessed_data(data)

    X_train, X_valid, y_train, y_valid = train_test_split_by_step(
        data=preprocesed_data,
        step=columns.STEP,
        target=columns.TARGET,
        train_size=0.8,
    )

    return (
        X_train.loc[:, columns.NUMERICAL],
        X_valid.loc[:, columns.NUMERICAL],
        y_train,
        y_valid,
    )


@step
def baseline_and_new_data_combiner(
    baseline_data: pd.DataFrame, new_data: pd.DataFrame
) -> Output(data=pd.DataFrame):
    return pd.concat([baseline_data, new_data], axis=0)
