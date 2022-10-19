"""
Training pipeline
"""
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from src.util import path

RANDOM_SEED = 0

from zenml.pipelines import pipeline
from zenml.integrations.constants import LIGHTGBM, SKLEARN


@pipeline()
def train_pipeline(importer, transformer, trainer, evaluator):
    """Fraud model Training Pipeline
    Args:

    Returns:
        metrics: dict[str,float]
    """
    df = importer()
    X_train, X_test, y_train, y_test = transformer(df)
    model = trainer(X_train, y_train)
    metrics = evaluator(X_test, y_test, model)

    return metrics
