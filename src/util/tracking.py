"""tracking functions"""
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


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


LGBM_TRAIN_PARAMS = {
    "max_depth": -1,
    "n_estimators": 200,
    "num_leaves": 45,
    "scale_pos_weight": 1,
}
