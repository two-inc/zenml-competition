import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.util import path

RANDOM_SEED = 0


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
        for key, val in metrics.items():
            f.write(f"{key} - {val}\n")


if __name__ == "__main__":
    train_pipeline()
