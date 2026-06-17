import pandas as pd
import numpy as np
import pytest

from src.training.balancing import DataBalancer


class DummyGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, n_samples):
        X = pd.DataFrame({"cat": ["A"] * n_samples, "num": np.arange(n_samples, dtype=float)})
        y = pd.Series(np.linspace(0.0, 1.0, n_samples), name="target")
        return X, y

    def conditional_sampling(self, n_samples, feature_conditions=None, target_value=None):
        self.calls.append((n_samples, dict(feature_conditions or {}), target_value))
        val = list((feature_conditions or {}).values())[0]
        X = pd.DataFrame({"cat": [val] * n_samples, "num": np.zeros(n_samples)})
        y = pd.Series(np.linspace(0.0, 1.0, n_samples), name="target")
        return X, y


def test_regression_categorical_balance_expands_minority_categories():
    X = pd.DataFrame({"cat": ["A", "A", "A", "B"], "num": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([10.0, 11.0, 12.0, 13.0], name="target")
    gen = DummyGenerator()

    balancer = DataBalancer(
        strategy="synthetic",
        task_type="regression",
        synthetic_regression_mode="categorical_balance",
        categorical_balance_cols=["cat"],
        random_state=0,
    )
    X_bal, y_bal = balancer.balance(X, y, generator=gen)

    assert len(X_bal) == 6
    assert (X_bal["cat"] == "B").sum() == 3
    assert len(gen.calls) == 1
    assert gen.calls[0][0] == 2
    assert gen.calls[0][1] == {"cat": "B"}


def test_regression_random_over_uses_quantile_bins():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"num": rng.randn(40)})
    y = pd.Series(rng.randn(40) * 10, name="target")
    balancer = DataBalancer(strategy="random_over", task_type="regression", random_state=0)
    X_bal, y_bal = balancer.balance(X, y)
    assert len(X_bal) >= len(X)
    assert len(y_bal) == len(X_bal)


def test_regression_augment_mode_uses_joint_generate():
    X = pd.DataFrame({"cat": ["A", "B"], "num": [1.0, 2.0]})
    y = pd.Series([1.0, 2.0], name="target")
    gen = DummyGenerator()

    balancer = DataBalancer(
        strategy="synthetic",
        task_type="regression",
        synthetic_regression_mode="augment",
        synthetic_regression_augment_ratio=1.0,
        random_state=0,
    )
    X_bal, y_bal = balancer.balance(X, y, generator=gen)

    assert len(X_bal) == 4
    assert len(gen.calls) == 0
