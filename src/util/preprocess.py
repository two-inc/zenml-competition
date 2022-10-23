import datetime
from typing import Callable

import pandas as pd
from util import columns

cat_columns = ["customer", "age", "gender", "merchant", "category"]
drop_columns = ["zipMerchant", "zipcodeOri"]


def preprocess(
    X: pd.DataFrame, cat_columns: list[str], drop_columns: list[str]
) -> pd.DataFrame:
    """Preprocesses the 'Synthetic data from a financial payment system' Dataset"""
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
    return get_grouped_transform(
        data,
        column,
        group,
        lambda x: x.rolling(window, min_periods, *args, **kwargs).sum(),
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
    return get_grouped_transform(
        data,
        column,
        group,
        lambda x: x.shift(period_shift)
        .rolling(window, min_periods, *args, **kwargs)
        .mean(),
    )


def get_max_group_count(data: pd.Series) -> int:
    return data.value_counts().max()


def mean_category_amount_previous_step(
    data: pd.DataFrame, category: str, step: str, amount: str
) -> pd.Series:
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


def train_test_split_by_step(
    data: pd.DataFrame, step: str, target: str, train_size: int = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = data.copy()
    train_step_cutoff = data.loc[:, step].quantile(0.8)
    train_idx = data[data.loc[:, step] <= train_step_cutoff].index
    valid_idx = data[data.loc[:, step] > train_step_cutoff].index
    df_train = data.iloc[train_idx, :]
    df_test = data.iloc[valid_idx, :]
    X_train, y_train = df_train.drop(target, axis=1), df_train.loc[:, target]
    X_test, y_test = df_test.drop(target, axis=1), df_test.loc[:, target]
    return (X_train, X_test, y_train, y_test)


def get_column_indices(data: pd.DataFrame, columns: list[str]) -> list[int]:
    return [data.columns.get_loc(i) for i in columns]


def get_preprocessed_data(data: pd.DataFrame) -> pd.DataFrame:
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

    data["transactions_completed"] = 1
    for group, max_transactions in max_transaction_map.items():
        data[f"{group}_amount_moving_max"] = get_rolling_max_by_group(
            data, "amount", group, window=max_transactions, min_periods=1
        )
        data[f"{group}_fraud_comitted_mean"] = get_rolling_mean_by_group_lag(
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

    data[columns.CATEGORICAL] = (
        data.loc[:, columns.CATEGORICAL]
        .applymap(lambda x: x.strip("'"))
        .astype("category")
    )
    data = data.drop(drop_columns, axis=1)
    return data
