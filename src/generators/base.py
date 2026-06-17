from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import pandas as pd

DEFAULT_LABEL_COL = "target"


def train_subset_from_conditions(
    train_df: pd.DataFrame,
    *,
    target_value: Optional[int] = None,
    feature_conditions: Optional[Dict[str, Any]] = None,
    label_col: str = DEFAULT_LABEL_COL,
) -> pd.DataFrame:
    """Filter train rows by optional label and equality constraints on columns."""
    fc = dict(feature_conditions or {})
    for col in fc:
        if col not in train_df.columns:
            raise ValueError(f"Unknown column in feature_conditions: {col!r}")

    m = pd.Series(True, index=train_df.index)
    if target_value is not None:
        if label_col not in train_df.columns:
            raise RuntimeError(f"train_df must contain label column {label_col!r}")
        m &= train_df[label_col] == target_value
    for col, val in fc.items():
        m &= train_df[col] == val

    sub = train_df.loc[m]
    if len(sub) == 0:
        raise ValueError(
            "No training rows match target_value=%r and feature_conditions=%r"
            % (target_value, fc)
        )
    return sub


def enforce_feature_conditions_on_X(
    X_syn: pd.DataFrame,
    feature_conditions: Dict[str, Any],
    *,
    label_col: str = DEFAULT_LABEL_COL,
) -> pd.DataFrame:
    """Set conditioned feature columns to exact values (copy)."""
    out = X_syn.copy()
    for col, val in feature_conditions.items():
        if col == label_col:
            continue
        if col not in out.columns:
            raise ValueError(f"Condition column {col!r} not in synthetic X columns")
        out[col] = val
    return out


class BaseDataGenerator(ABC):
    """Common interface for synthetic tabular generators (pipeline-swappable)."""

    def __init__(self, **kwargs):
        """Store Hydra-style params in self.params."""
        self.params = kwargs
        self.is_fitted = False
        self._task_type: str = "classification"

    def set_task_type(self, task_type: str) -> None:
        """classification | regression — used for sampling / post-processing."""
        self._task_type = (task_type or "classification").strip().lower()

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseDataGenerator":
        """Train on real (X, y)."""
        pass

    @abstractmethod
    def generate(self, n_samples: int = None, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Return (X_syn, y_syn)."""
        pass

    def conditional_sampling(
        self,
        n_samples: int,
        target_value: Optional[int] = None,
        feature_conditions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Conditional sampling: optional target_value and/or feature_conditions
        (equality filters). If both are empty, delegates to generate().
        """
        fc = dict(feature_conditions or {})
        if target_value is None and len(fc) == 0:
            return self.generate(n_samples, **kwargs)
        return self._generate_conditional(
            n_samples,
            target_value=target_value,
            feature_conditions=fc,
            **kwargs,
        )

    def _generate_conditional(
        self,
        n_samples: int,
        target_value: Optional[int] = None,
        feature_conditions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Override in subclasses that support conditional_sampling."""
        fc = dict(feature_conditions or {})
        if not fc and target_value is None:
            return self.generate(n_samples, **kwargs)
        raise NotImplementedError(
            f"{self.__class__.__name__} must override _generate_conditional for "
            "conditional_sampling."
        )

    def save(self, path: str):
        """Method to save the state of the generator (optional)."""
        pass

    def load(self, path: str):
        """Method to load the state of the generator (optional)."""
        pass
