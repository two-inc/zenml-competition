"""
Training pipeline
"""
from zenml.pipelines import pipeline

from src.util.settings import docker_settings


@pipeline(
    name="train_pipeline_9",
    enable_cache=True,
    settings={"docker": docker_settings},
)
def train_pipeline(importer, transformer, trainer, evaluator):
    """Trains a fraud detection model

    Process:
        - Import some data
        - Transform and split it into training and holdout set
        - Train the model
        - Evaluate the performance of the model on the holdout set
    """
    df = importer()
    X_train, X_test, y_train, y_test = transformer(df)
    model = trainer(X_train, y_train)
    metrics = evaluator(X_test, y_test, model)
    return metrics
