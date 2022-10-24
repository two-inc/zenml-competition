import pytest
import pandas as pd
import numpy as np
from src.util.tracking import get_classification_metrics
from src.util.preprocess import SEED


@pytest.fixture
def X_dummy(): 
    return pd.DataFrame(np.random.randint(0,100,size=(100, 4), random_state=SEED), columns=list('ABCD'))

@pytest.fixture
def y_dummy():
    return pd.Series(np.random.randint(0,1,size=(100,1),random_state=SEED))