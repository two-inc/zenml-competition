"""Testing preprocess functions"""
import pandas as pd
import pytest

from src.util.preprocess import get_column_indices
from src.util.preprocess import preprocess
from src.util.preprocess import print_description
from src.util.preprocess import train_test_split_by_step

TEST_DATA = {
    "a": ["'col1'", "'col2'", "'fraud'"],
    "col2": [1, 2, 3],
    "fraud": [4, 5, 6],
}


def test_preprocess():
    """test preprocessing function"""
    test = {"a": ["'a'", "'b'", "'c'"], "b": [1, 2, 3], "c": [4, 5, 6]}
    X = pd.DataFrame.from_dict(test)
    cat_columns = ["a"]
    drop_columns = ["b"]
    result = preprocess(X, cat_columns, drop_columns)
    X = X.drop("b", axis=1)
    X["a"] = X.loc[:, "a"].str.strip("'").astype("category")
    assert (X == result).all().all()


def test_train_test_split_by_step():
    """test train test split by step function"""
    X = pd.DataFrame.from_dict(TEST_DATA)

    result = train_test_split_by_step(X, "col2", "fraud", 0.8)
    X_train, X_test, y_train, y_test = result
    assert X_train.to_dict() == {
        "a": {0: "'col1'", 1: "'col2'"},
        "col2": {0: 1, 1: 2},
    }
    assert X_test.to_dict() == {"a": {2: "'fraud'"}, "col2": {2: 3}}
    assert y_train.to_dict() == {0: 4, 1: 5}
    assert y_test.to_dict() == {2: 6}


def test_train_test_split_by_step_invalid_train_size():
    """test train test split by step function with invalid train size"""
    X = pd.DataFrame.from_dict(TEST_DATA)

    with pytest.raises(
        ValueError,
        match="train_size argument must be between 0 and 1. train_size: 1.1",
    ):
        train_test_split_by_step(X, "col2", "fraud", 1.1)


def test_get_column_indices():
    """test to ensure get_column_indices correctly returns the column indices"""
    X = pd.DataFrame.from_dict(TEST_DATA)
    cols = get_column_indices(X, ["col2", "fraud"])
    assert cols == [1, 2]
