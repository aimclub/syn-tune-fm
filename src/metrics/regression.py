"""Regression metrics (continuous targets)."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.metrics.interface import BaseMetric


class RMSEMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "rmse"

    def calculate(self, y_true, y_pred, y_probs=None) -> float:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class MAEMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "mae"

    def calculate(self, y_true, y_pred, y_probs=None) -> float:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        return float(mean_absolute_error(y_true, y_pred))
