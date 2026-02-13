import warnings
from typing import Tuple
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from src.generators.base import BaseDataGenerator

warnings.filterwarnings("ignore")

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

LABEL_COL = "target"


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    for col in df_clean.columns:
        if isinstance(df_clean[col].dtype, CategoricalDtype):
            df_clean[col] = df_clean[col].astype("object")
    return df_clean


def _ensure_classes_presence(X_syn, y_syn, X_real, y_real):
    syn_df = X_syn.copy()
    syn_df[LABEL_COL] = y_syn.values
    real_df = X_real.copy()
    real_df[LABEL_COL] = y_real.values
    unique_real = y_real.unique()
    unique_syn = y_syn.unique()
    if len(unique_syn) < len(unique_real):
        missing = set(unique_real) - set(unique_syn)
        for cls in missing:
            real_samples = real_df[real_df[LABEL_COL] == cls]
            if len(real_samples) > 0:
                n_inject = min(3, len(real_samples))
                sample_to_inject = real_samples.sample(n=n_inject, replace=False)
                syn_df.iloc[:n_inject] = sample_to_inject.values
    return syn_df.drop(columns=[LABEL_COL], errors="ignore"), syn_df[LABEL_COL]


class GaussianCopulaGenerator(BaseDataGenerator):
    """Генератор Gaussian Copula (Traditional)"""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GaussianCopulaGenerator":
        self._train_df = X.copy()
        self._train_df[LABEL_COL] = y.values
        self._X_real = X
        self._y_real = y
        self.is_fitted = True
        return self

    def generate(self, n_samples: int = None, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        n = n_samples or kwargs.get("n_samples") or self.params.get("n_samples") or len(self._train_df)
        seed = self.params.get("seed", 42)
        np.random.seed(seed)
        df_ready = _fix_dtypes(self._train_df)
        try:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df_ready)
            model = GaussianCopulaSynthesizer(metadata)
            model.fit(df_ready)
            synthetic_df = model.sample(num_rows=len(df_ready))
            X_syn = synthetic_df.drop(columns=[LABEL_COL], errors="ignore")
            y_syn = synthetic_df[LABEL_COL]
        except Exception as e:
            print(f"  [Copula Error] {e}. Fallback to bootstrap.")
            s = self._train_df.sample(frac=1, replace=True).reset_index(drop=True)
            X_syn = s.drop(columns=[LABEL_COL])
            y_syn = s[LABEL_COL]
        X_syn, y_syn = _ensure_classes_presence(X_syn, y_syn, self._X_real, self._y_real)
        if len(X_syn) > n:
            X_syn, y_syn = X_syn.iloc[:n], y_syn.iloc[:n]
        elif len(X_syn) < n:
            extra = n - len(X_syn)
            X_extra = self._X_real.sample(n=min(extra, len(self._X_real)), replace=True)
            y_extra = self._y_real.loc[X_extra.index]
            X_syn = pd.concat([X_syn, X_extra], ignore_index=True)
            y_syn = pd.concat([y_syn, y_extra], ignore_index=True)
        return X_syn, y_syn
