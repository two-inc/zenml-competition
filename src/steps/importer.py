import pandas as pd
from zenml.steps import step, Output
from zenml.logger import get_logger

logger = get_logger(__name__)

@step()
def importer() -> Output(
    data=pd.DataFrame
    ):
    """Loads the raw dataset."""
    data = pd.read_csv('./src/data/bs140513_032310.csv')

    return data