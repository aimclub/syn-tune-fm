from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any

class BaseModelWrapper(ABC):
    """
    Abstract wrapper for any model (TabPFN, XGBoost, CatBoost).
    Hides the differences in external library APIs.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = None

    @abstractmethod
    def fine_tune(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Gradient-based or incremental fine-tuning of existing weights.
        """
        pass

    @abstractmethod
    def fit_context(self, X: pd.DataFrame, y: pd.Series):
        """
        In-Context Learning (ICL) without updating model weights.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns hard class labels.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns class probabilities (required for LogLoss, ROC-AUC).
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Saves model weights to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Loads model weights from disk."""
        pass