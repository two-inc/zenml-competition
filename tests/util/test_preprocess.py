from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
import pytest

from src.util.preprocess import preprocess


def test_preprocess():
    test = {"a": ["'a'", "'b'", "'c'"], "b": [1, 2, 3], "c": [4, 5, 6]}
    X = pd.DataFrame.from_dict(test)
    cat_columns = ["a"]
    drop_columns = ["b"]
    result = preprocess(X, cat_columns, drop_columns)
    X = X.drop("b", axis=1)
    X["a"] = X.loc[:, "a"].str.strip("'").astype("category")
    assert (X == result).all().all()
