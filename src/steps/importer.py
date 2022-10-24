"""Importer step"""
import pandas as pd
from zenml.logger import get_logger
from zenml.steps import Output
from zenml.steps import step

from src.util.data_access import load_data

logger = get_logger(__name__)


@step()
def importer() -> Output(data=pd.DataFrame):
    """Loads the raw fraud dataset from GCP."""
    return load_data()
