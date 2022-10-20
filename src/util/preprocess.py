import pandas as pd

cat_columns = ["customer", "age", "gender", "merchant", "category"]
drop_columns = ["zipMerchant", "zipcodeOri"]


def preprocess(
    X: pd.DataFrame, categorical_columns: list[str], drop_columns: list[str]
) -> pd.DataFrame:
    """Preprocesses the 'Synthetic data from a financial payment system' Dataset"""
    X = X.copy()
    X[cat_columns] = X.loc[:, cat_columns].applymap(lambda x: x.strip("'"))
    X = X.drop(drop_columns, axis=1)
    X = X[X["amount"] > 0]
    return X


def has_comitted_fraud_before_list(fraud_list: list[int]) -> list[int]:
    """Given a list of fraud occurrences, returns a list of whether fraud has been occurred at this step, or previously"""
    has_commited_fraud_list = []
    for index in range(len(fraud_list)):
        if index == 0 & fraud_list[index] == 0:
            has_commited_fraud_list.append(0)
        elif fraud_list[index] == 1 or 1 in has_commited_fraud_list:
            has_commited_fraud_list.append(1)
        else:
            has_commited_fraud_list.append(0)
    return has_commited_fraud_list
