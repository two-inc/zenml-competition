"""Importer step"""
import google.auth
import pandas as pd
from google.cloud import storage
from zenml.logger import get_logger
from zenml.steps import Output
from zenml.steps import step

from src.util.data_access import load_baseline_data
from src.util.data_access import load_new_data

logger = get_logger(__name__)


@step()
def baseline_data_importer() -> Output(data=pd.DataFrame):
    """
    Loads the Synthetic data from a
    financial payment system dataset from GCP.

    Dataset Link:
        - https://www.kaggle.com/datasets/ealaxi/banksim1

    Returns:
        pd.DataFrame: 'Synthetic data from a financial
                       payment system' dataset
    """
    return load_baseline_data()


@step()
def new_data_importer() -> Output(data=pd.DataFrame):
    """
    Loads the Synthetic data from a
    financial payment system dataset from GCP.

    Dataset Link:
        - https://www.kaggle.com/datasets/ealaxi/banksim1

    Returns:
        pd.DataFrame: 'Synthetic data from a financial
                       payment system' dataset
    """
    return load_new_data()
