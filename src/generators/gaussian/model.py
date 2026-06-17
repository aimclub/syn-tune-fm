import warnings
from typing import Type

import pandas as pd

from src.generators.sdv_base import SdvSingleTableGenerator

warnings.filterwarnings("ignore")
from sdv.single_table import GaussianCopulaSynthesizer


class GaussianCopulaGenerator(SdvSingleTableGenerator):
    """Gaussian Copula (SDV) generator."""

    @property
    def _sdv_synthesizer_cls(self) -> Type:
        return GaussianCopulaSynthesizer

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GaussianCopulaGenerator":
        super().fit(X, y)
        return self
