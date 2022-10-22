from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def get_metrics(true, pred, pred_proba):
    return {
        "ROC AUC": roc_auc_score(true, pred_proba),
        "PR AUC": average_precision_score(true, pred_proba),
        "Precision": precision_score(true, pred),
        "Recall": recall_score(true, pred),
        "F1 Score": f1_score(true, pred),
        "Brier Score": brier_score_loss(true, pred_proba),
        "Accuracy": accuracy_score(true, pred),
    }
