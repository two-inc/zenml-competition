import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from zenml.steps import step, Output
from zenml.pipelines import pipeline
from zenml.steps import step, Output
from zenml.logger import get_logger


@step(enable_cache=False)
def trainer(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    # config: ModelConfig
    ) -> Output(
    model = ClassifierMixin
    ):
 
    """Training a sklearn RF model."""
    model = RandomForestClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    return model