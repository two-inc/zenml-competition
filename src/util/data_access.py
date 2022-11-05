"""
Helpers for accessing sample data
"""
import os

import pandas as pd
from dotenv import load_dotenv

from src.util import columns
from src.util.preprocess import split_data_by_quantile


load_dotenv()


def load_data(object_name: str = None) -> pd.DataFrame:
    """Loads sample data from a storage Bucket

    Args:
        object_name (str, optional): Name of the object in the storage bucket

    Raises:
        ValueError: Provided object name is not found in the bucket
        ValueError: Data loaded could not be loaded as a pandas DataFrame

    Returns:
        pd.DataFrame: Storage Bucket Data
    """
    object_url = os.environ.get("BUCKET_URL") + (
        object_name or os.environ.get("DEFAULT_OBJECT_NAME")
    )
    result = pd.read_csv(object_url)

    if result is None:
        raise ValueError("Sample data not found")

    if not isinstance(result, pd.DataFrame):
        raise ValueError(
            f"Could not load sample data as a DataFrame. Loaded instead as type {result.__class__.__name__}"
        )

    return result


BASELINE_DATA_PROPORTION = 0.6


def load_baseline_data() -> pd.DataFrame:
    """Loads the 'baseline' data used for training the model and serves as the baseline in the CD pipeline

    Returns:
        pd.DataFrame: Baseline Data
    """
    data = load_data()
    df_baseline, _ = split_data_by_quantile(
        data, columns.STEP, quantile=BASELINE_DATA_PROPORTION
    )
    return df_baseline


def load_new_data() -> pd.DataFrame:
    """Loads the 'new' data used for updating the baseline data and used for identifying data drift

    Returns:
        pd.DataFrame: New Data
    """
    data = load_data()
    _, df_new = split_data_by_quantile(
        data, columns.STEP, quantile=BASELINE_DATA_PROPORTION
    )
    return df_new
