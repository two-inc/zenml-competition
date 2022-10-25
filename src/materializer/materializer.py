import os
import pickle
from typing import Any
from typing import Type
from typing import Union

import lgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from zenml.artifacts import DataArtifact
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from src.materializer import types

DEFAULT_FILENAME = "SyntheticFinancialData"


class ScorecardMaterializer(BaseMaterializer):
    """Custom materializer for the Two UK Scorecard"""

    ASSOCIATED_TYPES = [
        str,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        lgb.LGBMClassifier,
    ]

    ASSOCIATED_ARTIFACT_TYPES = (DataArtifact,)

    def handle_input(
        self, data_type: Type[Any]
    ) -> Union[str, np.ndarray, pd.Series, pd.DataFrame, lgb.LGBMClassifier]:
        """
        Loads the model from the artifact and returns it.
        Args:
            data_type: The type of the model to be loaded
        """
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            obj = pickle.load(fid)
        return obj

    def handle_return(
        self,
        obj: Union[
            str, np.ndarray, pd.Series, pd.DataFrame, lgb.LGBMClassifier
        ],
    ) -> None:
        """
        Saves the model to the artifact store.
        Args:
            model: The model to be saved
        """

        super().handle_return(obj)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(obj, fid)
