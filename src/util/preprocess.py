"""Preprocess definition"""
from typing import Callable
from typing import Optional

import pandas as pd
from sklearn.ensemble._forest import ForestClassifier

from src.util import columns

cat_columns = ["customer", "age", "gender", "merchant", "category"]
drop_columns = ["zipMerchant", "zipcodeOri"]
SEED = 42


def train_test_split_by_step(
    data: pd.DataFrame,
    step: str,
    target: Optional[str] = None,
    train_size: int = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits a DataFrame by a discrete step column into train and test sets

    Data splitting procedure specific for the format of the competition
    dataset. As there is a temporal component, the model should not be tested
    by testing it on transactions that occurred prior/between the transactions
    it was originally trained on. This necessitates a temporal splitting of the data
    according to the step feature.

    Args:
        data (pd.DataFrame): DataFrame
        step (str): Step/Day column
        target (str): Target
        train_size (int, optional): Size of the training set. Defaults to 0.8.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train and Test Sets
    """
    data = data.copy()
    if not (0 < train_size < 1):
        raise ValueError(
            f"train_size argument must be between 0 and 1. train_size: {train_size}"
        )
    df_train, df_test = split_data_by_quantile(data, step, quantile=train_size)
    X_train, y_train = df_train.drop(target, axis=1), df_train.loc[:, target]
    X_test, y_test = df_test.drop(target, axis=1), df_test.loc[:, target]
    return (X_train, X_test, y_train, y_test)


def split_data_by_quantile(
    data: pd.DataFrame, split_column: str, quantile: int
) -> tuple[pd.DataFrame]:
    train_step_cutoff = data.loc[:, split_column].quantile(quantile)
    train_idx = data[data.loc[:, split_column] <= train_step_cutoff].index
    valid_idx = data[data.loc[:, split_column] > train_step_cutoff].index
    return (data.iloc[train_idx, :], data.iloc[valid_idx, :])


def get_preprocessed_data(data: pd.DataFrame) -> pd.DataFrame:
    """Generates the preprocessed Dataset

    In addition to baseline preprocessing, this function is responsible
    for feature engineering, adding these columns to the raw data:
        - Customer Transaction Number
        - Merchant Transaction Number
        - Moving Averages for the Amount by Customer, Merchant & Category
        - Moving Standard Deviations for the Amount by Customer, Merchant & Category
        - Moving Max for the Amount by Customer, Merchant & Category
        - Proportion of previous fraudulent transactions for Customer, Merchant & Category
        - Average Amount Paid for a Category of Items the previous Step

    Args:
        data (pd.DataFrame): Raw Data

    Returns:
        pd.DataFrame: Preprocessed Data
    """
    data = data.copy()
    MAX_MERCHANT_TRANSACTIONS = get_max_group_count(data["merchant"])
    MAX_CUSTOMER_TRANSACTIONS = get_max_group_count(data["customer"])
    MAX_CATEGORY_TRANSACTIONS = get_max_group_count(data["category"])

    moving_average_configs = {
        "merchant": (MAX_MERCHANT_TRANSACTIONS, 50, 10, 5, 3),
        "customer": (MAX_CUSTOMER_TRANSACTIONS, 10, 5),
        "category": (MAX_CATEGORY_TRANSACTIONS, 100, 10),
    }

    max_transaction_map = {
        "merchant": MAX_MERCHANT_TRANSACTIONS,
        "customer": MAX_CUSTOMER_TRANSACTIONS,
        "category": MAX_CATEGORY_TRANSACTIONS,
    }

    for group, windows in moving_average_configs.items():
        max_window = max(windows)
        for window in windows:
            w = "total" if window == max_window else window
            data[f"{group}_amount_ma_{w}"] = get_rolling_mean_by_group(
                data, "amount", group, window=window, min_periods=1
            )
            data[f"{group}_amount_mstd_{w}"] = get_rolling_std_by_group(
                data, "amount", group, window=window, min_periods=1
            ).fillna(0)

    data["transactions_completed"] = 1
    for group, max_transactions in max_transaction_map.items():
        data[f"{group}_amount_moving_max"] = get_rolling_max_by_group(
            data, "amount", group, window=max_transactions, min_periods=1
        )
        data[f"{group}_fraud_commited_mean"] = get_rolling_mean_by_group_lag(
            data,
            "amount",
            group,
            window=max_transactions,
            min_periods=1,
            period_shift=1,
        )
        if group != "category":
            data[f"{group}_transaction_number"] = get_rolling_sum_by_group(
                data,
                "transactions_completed",
                group,
                window=max_transactions,
                min_periods=1,
            )

    ## Amount transacted the day before
    data[
        "mean_category_amount_previous_step"
    ] = mean_category_amount_previous_step(data, "category", "step", "amount")

    data = preprocess(data, columns.CATEGORICAL, columns.DROP)
    return data


def preprocess(
    X: pd.DataFrame, cat_columns: list[str], drop_columns: list[str]
) -> pd.DataFrame:
    """Applies simple preprocessing to the 'Synthetic data from a financial payment system' Dataset"""
    X = X.copy()
    X[cat_columns] = (
        X.loc[:, cat_columns]
        .applymap(lambda x: x.strip("'"))
        .astype("category")
    )
    X = X.drop(drop_columns, axis=1)
    return X


def get_grouped_transform(
    data: pd.DataFrame, column: str, group: str, func: Callable
) -> pd.Series:
    """Applies a function as a transform on grouped data

    Args:
        data (pd.DataFrame): Data to use for transform
        column (str): Feature to apply transform on
        group (str):  Feature to group data by
        func (Callable): Function to apply as transform

    Returns:
        pd.Series: Grouped Transform
    """
    return data.groupby(group)[column].transform(func)


def get_rolling_mean_by_group(
    data: pd.DataFrame,
    column: str,
    group: str,
    window,
    min_periods,
    *args,
    **kwargs,
) -> pd.Series:
    """Computes a rolling mean on grouped data

    Args:
        data (pd.DataFrame): Data to use for rolling mean computation
        column (str): Feature to compute rolling mean on
        group (str):  Feature to group data by

    Returns:
        pd.Series: Grouped Rolling Mean Transform
    """
    return get_grouped_transform(
        data,
        column,
        group,
        lambda x: x.rolling(window, min_periods, *args, **kwargs).mean(),
    )


def get_rolling_sum_by_group(
    data: pd.DataFrame,
    column: str,
    group: str,
    window,
    min_periods,
    *args,
    **kwargs,
) -> pd.Series:
    """Computes a rolling sum on grouped data

    Args:
        data (pd.DataFrame): Data to use for rolling sum computation
        column (str): Feature to compute rolling sum on
        group (str):  Feature to group data by

    Returns:
        pd.Series: Grouped Rolling Sum Transform
    """
    return get_grouped_transform(
        data,
        column,
        group,
        lambda x: x.rolling(window, min_periods, *args, **kwargs).sum(),
    )


def get_rolling_std_by_group(
    data: pd.DataFrame,
    column: str,
    group: str,
    window,
    min_periods,
    *args,
    **kwargs,
) -> pd.Series:
    """Computes a rolling standard deviation on grouped data

    Args:
        data (pd.DataFrame): Data to use for rolling standard deviation computation
        column (str): Feature to compute rolling standard deviation on
        group (str):  Feature to group data by

    Returns:
        pd.Series: Grouped Rolling Standard Deviation Transform
    """
    return get_grouped_transform(
        data,
        column,
        group,
        lambda x: x.rolling(window, min_periods, *args, **kwargs).std(),
    )


def get_rolling_max_by_group(
    data: pd.DataFrame,
    column: str,
    group: str,
    window,
    min_periods,
    *args,
    **kwargs,
) -> pd.Series:
    """Computes a rolling max on grouped data

    Args:
        data (pd.DataFrame): Data to use for rolling max computation
        column (str): Feature to compute rolling max on
        group (str):  Feature to group data by

    Returns:
        pd.Series: Grouped Rolling Max Transform
    """
    return get_grouped_transform(
        data,
        column,
        group,
        lambda x: x.rolling(window, min_periods, *args, **kwargs).max(),
    )


def get_rolling_mean_by_group_lag(
    data: pd.DataFrame,
    column: str,
    group: str,
    window,
    min_periods,
    period_shift=1,
    *args,
    **kwargs,
) -> pd.Series:
    """Computes a shifted rolling mean on grouped data

    Args:
        data (pd.DataFrame): Data to use for shifted rolling mean computation
        column (str): Feature to compute shifted rolling mean on
        group (str):  Feature to group data by

    Returns:
        pd.Series: Shifted Grouped Rolling Mean Transform
    """
    return get_grouped_transform(
        data,
        column,
        group,
        lambda x: x.shift(period_shift)
        .rolling(window, min_periods, *args, **kwargs)
        .mean(),
    )


def get_max_group_count(data: pd.Series) -> int:
    """Gets the Maximum Value Count for a given Series"""
    return data.value_counts().max()


def mean_category_amount_previous_step(
    data: pd.DataFrame, category: str, step: str, amount: str
) -> pd.Series:
    """Gets the mean category amount of previous step"""
    data = data.loc[:, [category, step, amount]].copy()
    amount_transacted_daily_by_category = (
        data.groupby([category, step])[amount]
        .mean()
        .rename("mean_category_amount_previous_step")
        .reset_index()
    )
    amount_transacted_daily_by_category["step"] += 1
    data = data.merge(
        amount_transacted_daily_by_category,
        how="left",
        on=["category", "step"],
    )
    return data["mean_category_amount_previous_step"].fillna(0)


def get_column_indices(data: pd.DataFrame, cols: list[str]) -> list[int]:
    """Gets the indices of the columns on the passed data"""
    return [data.columns.get_loc(i) for i in cols]


def get_feature_importances(
    model: ForestClassifier, X_train: pd.DataFrame
) -> dict[str, float]:
    """Retrieves the feature importances from a tree-based model by feature"""
    return {
        col: f for col, f in zip(X_train.columns, model.feature_importances_)
    }


def print_description(data: pd.Series) -> None:
    """Prints the description of a pandas Series"""
    print(f"{data.name.title()} Overview")
    print("--------------")
    print(data.describe())
    print("--------------")
