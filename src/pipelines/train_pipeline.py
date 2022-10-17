import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from src.steps import importer,evaluator,trainer,transformer

from src.util import path
import pdb 

RANDOM_SEED = 0

from zenml.pipelines import pipeline
from zenml.integrations.constants import LIGHTGBM, SKLEARN
# from zenml.config import DockerConfiguration

# docker_config = DockerConfiguration(dockerfile="Dockerfile",
#     build_context_root=".",
#     required_integrations=[LIGHTGBM, SKLEARN])


def train_pipeline():
    """Dummy Pipeline to test CML"""
    data = pd.read_csv(path.TRAIN_DATA_PATH)
    numerical_columns = ["step", "amount"]
    target_column = "fraud"
    X = data.loc[:, numerical_columns]
    y = data.loc[:, target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_SEED
    )
    clf = DummyClassifier()
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
    }
    with open(path.METRICS_PATH, "w") as f:
        f.write(f"Model Type: {clf.__class__.__name__}\n\n")
        f.write(f"Train Data Length: {len(X_train)}\n")
        f.write(f"Test Data Length: {len(X_test)}\n\n")
        for key, val in metrics.items():
            f.write(f"{key} - {val:2f}\n")



@pipeline()#(docker_configuration=docker_config)
def train_pipeline_(importer, transformer, trainer, evaluator):
    """ Fraud model Training Pipeline
    Args:
        
    Returns:
        metrics: dict[str,float]
    """
    df = importer()
    X_train, X_test, y_train, y_test = transformer(df)
    model = trainer(X_train, y_train)
    metrics = evaluator(X_test, y_test, model)

    with open(path.METRICS_PATH, "w") as f:
        # f.write(f"Model Type: {clf.__class__.__name__}\n\n")
        # f.write(f"Train Data Length: {len(X_train)}\n")
        # f.write(f"Test Data Length: {len(X_test)}\n\n")
        
        f.write(f"{metrics}")

    return metrics


if __name__ == "__main__":
    pipeline=train_pipeline_(importer.importer(), transformer.transformer(), trainer.trainer(), evaluator.evaluator())
    pipeline.run()