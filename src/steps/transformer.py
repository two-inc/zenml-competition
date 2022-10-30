"""Transformer step"""
import pandas as pd
from category_encoders.target_encoder import TargetEncoder
from zenml.steps import Output
from zenml.steps import step

from src.util import columns
from src.util.preprocess import get_preprocessed_data
from src.util.preprocess import train_test_split_by_step


@step(enable_cache=False)
def transformer(
    data: pd.DataFrame,
) -> Output(
    x_train=pd.DataFrame,
    x_test=pd.DataFrame,
    y_train=pd.Series,
    y_test=pd.Series,
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
    preprocessed_data = get_preprocessed_data(data)

    X_train, X_valid, y_train, y_valid = train_test_split_by_step(
        data=preprocessed_data,
        step=columns.STEP,
        target=columns.TARGET,
        train_size=0.8,
    )

    target_encoder = TargetEncoder(cols=columns.CATEGORICAL)
    X_train = target_encoder.fit_transform(X_train, y_train)
    X_valid = target_encoder.transform(X_valid)

    return (
        X_train.loc[:, columns.MODEL],
        X_valid.loc[:, columns.MODEL],
        y_train,
        y_valid,
    )
