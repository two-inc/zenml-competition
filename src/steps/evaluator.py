import pandas as pd
import numpy as np
from zenml.steps import step, Output
from zenml.logger import get_logger
from sklearn.base import ClassifierMixin


@step(enable_cache=False)
def evaluator(model: ClassifierMixin, X_test: np.ndarray, y_test: np.ndarray) -> Output(
    recall = float 
    ):
    """Model Evaluation and ML metrics register."""
    
    y_preds = model.predict(X_test)
    recall = recall_score(y_test, y_preds, average = 'macro')

    return recall