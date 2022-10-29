"""
Training pipeline
"""
import pandas as pd
from zenml.pipelines import pipeline

from src.util.settings import docker_settings


@pipeline(
    name="train_pipeline_3",
    enable_cache=False,
    settings={"docker": docker_settings},
)
def train_pipeline(
    importer, transformer, profiler, validator, trainer, evaluator
):
    """Fraud model Training Pipeline
    Args:

    Returns:
        metrics: dict[str,float]
    """
    df, validate_data = importer()
    X_train, X_test, y_train, y_test = transformer(df)
    profiler(X_train)
    validator(X_train, validate_data)
    model = trainer(X_train, y_train)
    metrics = evaluator(X_test, y_test, model)

    return metrics
