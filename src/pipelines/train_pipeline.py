"""
Training pipeline
"""
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from src.steps import importer, evaluator, trainer, transformer
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

    with open(path.METRICS_PATH, "w") as f:
        f.write(f"Model Type: {LIGHTGBM}\n")
        f.write(f"Train Data Length: {len(X_train)}\n")
        f.write(f"Test Data Length: {len(X_test)}\n\n")
        f.write(f"{metrics}")

    return metrics

