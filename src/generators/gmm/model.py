import random
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.generators.base import BaseDataGenerator
from src.generators.base import enforce_feature_conditions_on_X, train_subset_from_conditions

LABEL_COL = "target"


class GMMGenerator(BaseDataGenerator):
    """GMM-based generator."""

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = None,
        n_components: int = None,
        covariance_type: str = "full",
        reg_covar: float = 1e-5,
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            n_samples=n_samples,
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            **kwargs,
        )
        self.seed = seed
        self.n_samples = n_samples
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self._gmm = None
        self._full_data = None

    def _temp_gmm_n_components(self, n_rows: int) -> int:
        """GMM components for a temporary model on a subset of size n_rows (>= 2)."""
        n_components = self.n_components
        if n_components is None:
            random.seed(self.seed)
            upper = min(12, max(2, n_rows // 10 + 2))
            lower = min(2, upper)
            n_components = random.randint(lower, upper)
        return max(1, min(int(n_components), n_rows - 1))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GMMGenerator":
        self._X = X
        self._feature_names = X.columns.tolist()
        full_data = X.copy()
        full_data[LABEL_COL] = y.values
        self._full_data = full_data
        n_components = self.n_components
        if n_components is None:
            random.seed(self.seed)
            n_components = random.randint(2, min(12, len(X) // 10 + 2))
        self._gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            random_state=self.seed,
            reg_covar=self.reg_covar,
        )
        self._gmm.fit(full_data)
        self.is_fitted = True
        return self

    def generate(self, n_samples: int = None, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.is_fitted or self._gmm is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        n = n_samples or kwargs.get("n_samples") or self.n_samples or len(self._X)
        seed = self.params.get("seed", self.seed)
        np.random.seed(seed)
        X_syn_np, _ = self._gmm.sample(n)
        synthetic_df = pd.DataFrame(X_syn_np, columns=self._full_data.columns)
        for col in synthetic_df.columns:
            min_v, max_v = self._full_data[col].min(), self._full_data[col].max()
            synthetic_df[col] = synthetic_df[col].clip(min_v, max_v).round().astype(int)
        y_synthetic = synthetic_df[LABEL_COL]
        X_synthetic = synthetic_df.drop(columns=[LABEL_COL])[self._feature_names]
        return X_synthetic, y_synthetic

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
        if not fc and target_value is None:
            return self.generate(n_samples, **kwargs)
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        train_subset = train_subset_from_conditions(
            self._full_data,
            target_value=target_value,
            feature_conditions=fc,
            label_col=LABEL_COL,
        )
        n_rows = len(train_subset)
        if n_rows < 2:
            raise ValueError(
                "GMM conditional sampling needs at least 2 rows in the conditioned "
                f"subset, got {n_rows}"
            )
        n_components = self._temp_gmm_n_components(n_rows)

        temp_gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            random_state=self.seed,
            reg_covar=self.reg_covar,
        )
        temp_gmm.fit(train_subset)
        seed = self.params.get("seed", self.seed)
        np.random.seed(seed)
        X_syn_np, _ = temp_gmm.sample(n_samples)

        synthetic_df = pd.DataFrame(X_syn_np, columns=self._full_data.columns)
        for col in synthetic_df.columns:
            min_v, max_v = self._full_data[col].min(), self._full_data[col].max()
            synthetic_df[col] = synthetic_df[col].clip(min_v, max_v).round().astype(int)

        y_synthetic = synthetic_df[LABEL_COL]
        X_synthetic = synthetic_df.drop(columns=[LABEL_COL])[self._feature_names]
        if target_value is not None:
            y_synthetic = pd.Series([target_value] * len(X_synthetic), name=LABEL_COL)
        if fc:
            X_synthetic = enforce_feature_conditions_on_X(
                X_synthetic, fc, label_col=LABEL_COL
            )

        return X_synthetic.reset_index(drop=True), y_synthetic.reset_index(drop=True)
