"""
Training pipeline
"""
from zenml.pipelines import pipeline
from evidently.model_profile import Profile

from src.util.settings import docker_settings
import pandas as pd


@pipeline(enable_cache=True, settings={"docker": docker_settings})
def train_pipeline(importer, transformer, drift_detector, trainer, evaluator):
    """Fraud model Training Pipeline
    Args:

    Returns:
        metrics: dict[str,float]
    """
    df = importer()
    X_train, X_test, y_train, y_test = transformer(df)
    drift_report, _ = drift_detector(
        reference_dataset=X_train,
        comparison_dataset=X_test,
    )

    model = trainer(X_train, y_train)
    metrics = evaluator(X_test, y_test, model)

    return metrics
