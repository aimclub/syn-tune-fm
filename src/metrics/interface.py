from abc import ABC, abstractmethod
import numpy as np

class BaseMetric(ABC):
    """
    Interface for model evaluation metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric for logging (e.g., 'accuracy')."""
        pass

    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None) -> float:
        """
        Calculation of the metric value.

        Args:
            y_true: True class labels.
            y_pred: Predicted class labels (hard predictions).
            y_probs: Class probabilities (soft predictions). Mandatory for LogLoss/AUC.

        Returns:
            float: Value of the metric.
        """
        pass
