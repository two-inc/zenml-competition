"""
Helpers for accessing sample data etc
"""
import pandas as pd
from pandas import DataFrame
from zenml.logger import get_logger

from src.util import columns
from src.util import path
from src.util.preprocess import get_preprocessed_data

logger = get_logger(__name__)

BUCKET_URL = "http://storage.googleapis.com/zenmldata/"
DEFAULT_OBJECT_NAME = "bs140513_032310.csv"


def load_data(object_name: str = None) -> DataFrame:
    """
    Loads sample data from a GCP bucket.
    :param object_name: The name of the object within the bucket. If not specified, the default sample data is loaded.
    :return: A Pandas dataframe
    """
    object_url = BUCKET_URL + (object_name or DEFAULT_OBJECT_NAME)
    result = pd.read_csv(object_url)

    if result is None:
        raise ValueError("Sample data not found")

    if not isinstance(result, DataFrame):
        raise ValueError(
            f"Could not load sample data as a DataFrame. Loaded instead as type {result.__class__.__name__}"
        )

    return result
