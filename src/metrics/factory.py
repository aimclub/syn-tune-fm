from src.metrics.classification import (
    AccuracyMetric,
    BalancedAccuracyMetric,
    LogLossMetric,
    MacroF1Metric,
    RocAucMetric,
)
from src.metrics.regression import MAEMetric, RMSEMetric


class MetricFactory:
    _registry = {
        "accuracy": AccuracyMetric,
        "roc_auc": RocAucMetric,
        "log_loss": LogLossMetric,
        "macro_f1": MacroF1Metric,
        "balanced_accuracy": BalancedAccuracyMetric,
        "rmse": RMSEMetric,
        "mae": MAEMetric,
    }

    @staticmethod
    def get_metrics(metric_names: list):
        """Returns a list of initialized metric classes"""
        metrics = []
        for name in metric_names:
            if name not in MetricFactory._registry:
                print(f"Warning: Metric '{name}' not found in registry.")
                continue
            metrics.append(MetricFactory._registry[name]())
        return metrics
