"""
SDV single-table path: helpers + SdvSingleTableGenerator (Gaussian Copula, CTGAN, TVAE).

Inheritance: BaseDataGenerator -> SdvSingleTableGenerator -> concrete model.
Other generators inherit only BaseDataGenerator.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
from pandas.api.types import CategoricalDtype
from sdv.metadata import SingleTableMetadata

from src.generators.base import (
    BaseDataGenerator,
    DEFAULT_LABEL_COL,
    enforce_feature_conditions_on_X,
    train_subset_from_conditions,
)

warnings.filterwarnings("ignore")


def sdv_cuda_flag_from_params(params: Optional[Dict[str, Any]]) -> bool:
    """SDV synthesizers take ``cuda: bool``. Honor ``device`` in params (Hydra ++generator.params.device=...)."""
    d = str((params or {}).get("device", "")).strip().lower()
    if d in ("cpu", "false", "0", "no"):
        return False
    if d in ("cuda", "gpu", "true", "1", "yes") or d.startswith("cuda:"):
        if not torch.cuda.is_available():
            warnings.warn(
                "generator.params.device requests CUDA but torch.cuda.is_available() is False; "
                "using CPU for SDV.",
                UserWarning,
                stacklevel=3,
            )
            return False
        return True
    return bool(torch.cuda.is_available())
def sdv_fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    for col in df_clean.columns:
        if isinstance(df_clean[col].dtype, CategoricalDtype):
            df_clean[col] = df_clean[col].astype("object")
    return df_clean


def sdv_ensure_classes_presence(
    X_syn: pd.DataFrame,
    y_syn: pd.Series,
    X_real: pd.DataFrame,
    y_real: pd.Series,
    label_col: str = DEFAULT_LABEL_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    syn_df = X_syn.copy()
    syn_df[label_col] = y_syn.values
    real_df = X_real.copy()
    real_df[label_col] = y_real.values
    unique_real = y_real.unique()
    unique_syn = y_syn.unique()
    if len(unique_syn) < len(unique_real):
        missing = set(unique_real) - set(unique_syn)
        for cls in missing:
            real_samples = real_df[real_df[label_col] == cls]
            if len(real_samples) > 0:
                n_inject = min(3, len(real_samples))
                sample_to_inject = real_samples.sample(n=n_inject, replace=False)
                syn_df.iloc[:n_inject] = sample_to_inject.values
    return syn_df.drop(columns=[label_col], errors="ignore"), syn_df[label_col]


def sdv_y_labels_after_sample(
    synthetic_df: pd.DataFrame,
    X_syn: pd.DataFrame,
    train_subset: pd.DataFrame,
    target_value: Optional[int],
    label_col: str = DEFAULT_LABEL_COL,
) -> pd.Series:
    n = len(X_syn)
    if target_value is not None:
        return pd.Series([target_value] * n, name=label_col)
    if label_col in synthetic_df.columns:
        return synthetic_df[label_col]
    return pd.Series([train_subset[label_col].iloc[0]] * n, name=label_col)


def sdv_temp_synthesize_from_train_subset(
    train_subset: pd.DataFrame,
    n_samples: int,
    synthesizer_cls: Type,
    *,
    synthesizer_kwargs: Optional[Dict[str, Any]],
    seed: int,
    label_col: str,
    target_value: Optional[int],
    feature_conditions: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.Series]:
    np.random.seed(seed)
    df_ready = sdv_fix_dtypes(train_subset)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_ready)
    kw = dict(synthesizer_kwargs or {})
    temp_synth = synthesizer_cls(metadata, **kw)
    temp_synth.fit(df_ready)
    synthetic_df = temp_synth.sample(num_rows=n_samples)

    X_syn = synthetic_df.drop(columns=[label_col], errors="ignore")
    y_syn = sdv_y_labels_after_sample(
        synthetic_df, X_syn, train_subset, target_value, label_col=label_col
    )
    fc = dict(feature_conditions or {})
    if fc:
        X_syn = enforce_feature_conditions_on_X(X_syn, fc, label_col=label_col)
    if len(X_syn) > n_samples:
        X_syn, y_syn = X_syn.iloc[:n_samples], y_syn.iloc[:n_samples]
    return X_syn.reset_index(drop=True), y_syn.reset_index(drop=True)


class SdvSingleTableGenerator(BaseDataGenerator, ABC):
    """Shared SDV fit/generate/conditional (metadata, dtypes, subset temp model)."""

    def __init__(self, seed: int = 42, n_samples: Optional[int] = None, **kwargs):
        super().__init__(seed=seed, n_samples=n_samples, **kwargs)
        self.seed = seed
        self.n_samples = n_samples
        self._synthesizer = None
        self._train_df: Optional[pd.DataFrame] = None
        self._X_real: Optional[pd.DataFrame] = None
        self._y_real: Optional[pd.Series] = None

    @property
    def _label_col(self) -> str:
        return DEFAULT_LABEL_COL

    @property
    @abstractmethod
    def _sdv_synthesizer_cls(self) -> Type:
        """SDV synthesizer class (e.g. CTGANSynthesizer)."""

    def _sdv_synthesizer_kwargs(self) -> Dict[str, Any]:
        """Extra __init__ kwargs besides metadata."""
        return {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SdvSingleTableGenerator":
        lc = self._label_col
        self._train_df = X.copy()
        self._train_df[lc] = y.values
        self._X_real = X
        self._y_real = y
        df_ready = sdv_fix_dtypes(self._train_df)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df_ready)
        synth_cls = self._sdv_synthesizer_cls
        self._synthesizer = synth_cls(metadata, **self._sdv_synthesizer_kwargs())
        self._synthesizer.fit(df_ready)
        self.is_fitted = True
        return self

    def generate(self, n_samples: int = None, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.is_fitted or self._synthesizer is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        lc = self._label_col
        n = n_samples or kwargs.get("n_samples") or self.n_samples or len(self._train_df)
        seed = self.params.get("seed", self.seed)
        np.random.seed(seed)
        synthetic_df = self._synthesizer.sample(num_rows=n)
        X_syn = synthetic_df.drop(columns=[lc], errors="ignore")
        y_syn = synthetic_df[lc]
        if getattr(self, "_task_type", "classification") == "regression":
            if self._y_real is not None and self._y_real.name:
                y_syn = y_syn.rename(self._y_real.name)
        else:
            X_syn, y_syn = sdv_ensure_classes_presence(
                X_syn, y_syn, self._X_real, self._y_real, label_col=lc
            )
        if len(X_syn) > n:
            X_syn, y_syn = X_syn.iloc[:n], y_syn.iloc[:n]
        elif len(X_syn) < n:
            extra = n - len(X_syn)
            X_extra = self._X_real.sample(n=min(extra, len(self._X_real)), replace=True)
            y_extra = self._y_real.loc[X_extra.index]
            X_syn = pd.concat([X_syn, X_extra], ignore_index=True)
            y_syn = pd.concat([y_syn, y_extra], ignore_index=True)
        return X_syn, y_syn

    def conditional_sampling(
        self,
        n_samples: int,
        target_value: Optional[int] = None,
        feature_conditions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        return super().conditional_sampling(
            n_samples,
            target_value=target_value,
            feature_conditions=feature_conditions,
            **kwargs,
        )

    def _generate_conditional(
        self,
        n_samples: int,
        target_value: Optional[int] = None,
        feature_conditions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        fc = dict(feature_conditions or {})
        lc = self._label_col
        if not fc and target_value is None:
            return self.generate(n_samples, **kwargs)
        if not self.is_fitted or self._synthesizer is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        train_subset = train_subset_from_conditions(
            self._train_df,
            target_value=target_value,
            feature_conditions=fc,
            label_col=lc,
        )
        seed = self.params.get("seed", self.seed)
        return sdv_temp_synthesize_from_train_subset(
            train_subset,
            n_samples,
            self._sdv_synthesizer_cls,
            synthesizer_kwargs=self._sdv_synthesizer_kwargs(),
            seed=seed,
            label_col=lc,
            target_value=target_value,
            feature_conditions=fc,
        )
