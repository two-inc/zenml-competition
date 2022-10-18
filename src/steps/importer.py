import pandas as pd
from zenml.steps import step, Output
from zenml.logger import get_logger
from src.util.data_access import load_data

logger = get_logger(__name__)


@step()
def importer() -> Output(data=pd.DataFrame):
    """Loads the raw fraud dataset fro GCP."""
    data = load_data()

    return data
