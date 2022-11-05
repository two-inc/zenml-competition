"""Functions and data used for interacting with experiment tracking"""
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from zenml.client import Client

from src.util.preprocess import SEED


HGBM_PARAMS = {"max_leaf_nodes": None, "max_depth": None, "random_state": SEED}


try:
    experiment_tracker = Client().active_stack.experiment_tracker
    experiment_tracker_name = experiment_tracker.name
except AttributeError:
    experiment_tracker_name = None


def get_experiment_id(experiment_name: str) -> str:
    EXPERIMENT_ID = mlflow.get_experiment_by_name("EXPERIMENT_NAME")
    if not EXPERIMENT_ID:
        EXPERIMENT_ID = mlflow.create_experiment(experiment_name)
    return EXPERIMENT_ID


def get_classification_metrics(
    true: pd.Series, pred: pd.Series, pred_proba: pd.Series
) -> dict[str, float]:
    """Computes classification metrics

    Args:
        true (pd.Series): True Class Labels
        pred (pd.Series): Predicted Class Labels
        pred_proba (pd.Series): Predicted Positive Label Probability

    Returns:
        dict[str,float]: Classification Metrics
    """
    return {
        "ROC AUC": roc_auc_score(true, pred_proba),
        "PR AUC": average_precision_score(true, pred_proba),
        "Precision": precision_score(true, pred),
        "Recall": recall_score(true, pred),
        "F1 Score": f1_score(true, pred),
        "Brier Score": brier_score_loss(true, pred_proba),
        "Accuracy": accuracy_score(true, pred),
    }
