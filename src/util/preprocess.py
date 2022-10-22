import datetime

import pandas as pd

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
