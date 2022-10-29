"""Importer step"""
import google.auth
import pandas as pd
from google.cloud import storage
from zenml.logger import get_logger
from zenml.steps import Output
from zenml.steps import step

from src.util.data_access import load_data

logger = get_logger(__name__)


@step()
def importer() -> Output(data=pd.DataFrame, validate_data=bool):
    """
    Loads the Synthetic data from a
    financial payment system dataset from GCP.

    Dataset Link:
        - https://www.kaggle.com/datasets/ealaxi/banksim1

    Returns:
        pd.DataFrame: 'Synthetic data from a financial
                       payment system' dataset
    """
    import google.auth

    credentials, project_id = google.auth.default()
    assert project_id == "zenml-competition"
    print(credentials)
    print(credentials.__dict__)
    if hasattr(credentials, "service_account_email"):
        print(credentials.service_account_email)
    else:
        print(
            "WARNING: no service account credential. User account credential?"
        )
    validate_data: bool = True
    return load_data(), validate_data
