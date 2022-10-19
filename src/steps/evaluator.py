"""Evaluator step"""
import pandas as pd
from typing import Dict
from sklearn.metrics import r2_score
from zenml.logger import get_logger
from zenml.steps import step
from zenml.integrations.constants import LIGHTGBM
import lightgbm as lgbm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn import preprocessing
from src.util.env import CATEGORICAL_FEATURES
from src.util import path

logger = get_logger(__name__)


@step(enable_cache=False)
def evaluator(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    model: lgbm.LGBMClassifier,
) -> Dict[str, float]:
    """Evaluate a LightGBM using ROC-AUC, PR-AUC & Brier Score.

    Args:
        X_test: DataFrame with eval feature data.
        y_test: DataFrame with eval target data.
        model: Trained LightGBM Classifier.

    Returns:
        dict[str,float]: Metric Results
    """
    lbl = preprocessing.LabelEncoder()
    for categorical_feature in CATEGORICAL_FEATURES:
        X_test[categorical_feature] = lbl.fit_transform(
            X_test[categorical_feature].astype(str)
        )

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metric_results = {
        "rocauc": roc_auc_score(y_test, y_pred_proba),
        "prauc": average_precision_score(y_test, y_pred_proba),
        "brier": brier_score_loss(y_test, y_pred_proba),
    }
    logger.info(f"Metric Values:\n{metric_results}")

    with open(path.METRICS_PATH, "a") as f:
        f.write(f"{metric_results}")

    return metric_results
