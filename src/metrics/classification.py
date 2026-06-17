from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from src.metrics.interface import BaseMetric
import numpy as np


class MacroF1Metric(BaseMetric):
    @property
    def name(self):
        return "macro_f1"

    def calculate(self, y_true, y_pred, y_probs=None):
        return f1_score(y_true, y_pred, average="macro", zero_division=0)


class BalancedAccuracyMetric(BaseMetric):
    @property
    def name(self):
        return "balanced_accuracy"

    def calculate(self, y_true, y_pred, y_probs=None):
        return balanced_accuracy_score(y_true, y_pred)

class AccuracyMetric(BaseMetric):
    @property
    def name(self):
        return "accuracy"

    def calculate(self, y_true, y_pred, y_probs=None):
        return accuracy_score(y_true, y_pred)

class RocAucMetric(BaseMetric):
    def __init__(self, multi_class='ovr'):
        self.multi_class = multi_class

    @property
    def name(self):
        return "roc_auc"

    def calculate(self, y_true, y_pred, y_probs=None):
        if y_probs is None:
            raise ValueError("ROC AUC requires probabilities (y_probs)")
        
        # Handle binary/multi-class classification
        if y_probs.shape[1] == 2:
             # For binary, take the probability of the positive class
             return roc_auc_score(y_true, y_probs[:, 1])
        else:
             return roc_auc_score(y_true, y_probs, multi_class=self.multi_class)

class LogLossMetric(BaseMetric):
    @property
    def name(self):
        return "log_loss"

    def calculate(self, y_true, y_pred, y_probs=None):
        if y_probs is None:
            raise ValueError("Log Loss requires probabilities")
        return log_loss(y_true, y_probs)
