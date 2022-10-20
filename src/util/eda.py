import pandas as pd


def print_description(data: pd.Series) -> None:
    """Prints the description of a pandas Series"""
    print(f"{data.name.title()} Overview")
    print("--------------")
    print(data.describe())
    print("--------------")
