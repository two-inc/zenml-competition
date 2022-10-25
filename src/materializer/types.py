from typing import Protocol


class Classifier(Protocol):
    """Classifier Interface"""

    def predict(*args, **kwargs):
        ...

    def predict_proba(*args, **kwargs):
        ...


class TreeBasedModel(Protocol):
    """Tree-based Model Interface"""

    feature_importances_: list[float]


ALL: list[Protocol] = [Classifier, TreeBasedModel]
