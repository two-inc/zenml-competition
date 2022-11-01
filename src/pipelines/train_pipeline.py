"""
Training pipeline
"""
from zenml.pipelines import pipeline

from src.util.settings import docker_settings


@pipeline(enable_cache=True, settings={"docker": docker_settings})
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
