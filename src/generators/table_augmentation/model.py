from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.generators.base import (
    BaseDataGenerator,
    enforce_feature_conditions_on_X,
    train_subset_from_conditions,
)

LABEL_COL = "target"


def _pick_augmentation_features(
    feature_cols: List[str],
    rng: np.random.RandomState,
    required_cols: Optional[List[str]] = None,
) -> List[str]:
    """Random feature fraction; required_cols are always included."""
    required = list(required_cols or [])
    if len(feature_cols) > 1:
        frac = rng.uniform(0.5, 1.0)
        n_keep = int(np.ceil(len(feature_cols) * frac))
        n_keep = max(1, min(n_keep, len(feature_cols)))
        pool = list(feature_cols)
        selected = rng.choice(pool, size=n_keep, replace=False).tolist()
        for c in required:
            if c not in selected:
                selected.append(c)
    else:
        selected = list(feature_cols)
    return selected


def _generate_task_from_dataset(
    df: pd.DataFrame,
    label: str,
    seed: int,
    reuse_original_target: bool = True,
    n_samples: int = None,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    if n_samples is None:
        n_samples = len(df)
    subset_indices = rng.choice(len(df), size=n_samples, replace=True)
    df_sub = df.iloc[subset_indices].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != label]
    n_feats = len(feature_cols)
    if n_feats > 1:
        frac = rng.uniform(0.5, 1.0)
        n_keep = int(np.ceil(n_feats * frac))
        n_keep = max(1, min(n_keep, n_feats))
        selected_features = rng.choice(feature_cols, size=n_keep, replace=False).tolist()
    else:
        selected_features = feature_cols
    if reuse_original_target:
        final_features = selected_features
    else:
        potential_cols = selected_features + [label]
        target_col = rng.choice(potential_cols).item()
        final_features = [c for c in potential_cols if c != target_col]
    return df_sub[final_features + [label]].copy()


class TableAugmentationGenerator(BaseDataGenerator):
    """Table augmentation: fit() stores data; generate() only samples (bootstrap + feature subset)."""

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = None,
        reuse_original_target: bool = True,
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            n_samples=n_samples,
            reuse_original_target=reuse_original_target,
            **kwargs,
        )
        self.seed = seed
        self.n_samples = n_samples
        self.reuse_original_target = reuse_original_target

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TableAugmentationGenerator":
        self._train_df = X.copy()
        self._train_df[LABEL_COL] = y.values
        self._label = LABEL_COL
        self._selected_columns = None
        self.is_fitted = True
        return self

    def generate(self, n_samples: int = None, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        n = n_samples or kwargs.get("n_samples") or self.n_samples or len(self._train_df)
        seed = self.params.get("seed", self.seed)
        reuse_target = self.params.get("reuse_original_target", self.reuse_original_target)
        df_synth = _generate_task_from_dataset(
            self._train_df,
            label=self._label,
            seed=seed,
            reuse_original_target=reuse_target,
            n_samples=n,
        )
        self._selected_columns = [c for c in df_synth.columns if c != self._label]
        X_syn = df_synth[self._selected_columns]
        y_syn = df_synth[self._label]
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
        if not fc and target_value is None:
            return self.generate(n_samples, **kwargs)
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        class_data = train_subset_from_conditions(
            self._train_df,
            target_value=target_value,
            feature_conditions=fc,
            label_col=LABEL_COL,
        )

        seed = self.params.get("seed", self.seed)
        rng = np.random.RandomState(seed)

        idx = rng.choice(len(class_data), size=n_samples, replace=True)
        df_subset = class_data.iloc[idx].reset_index(drop=True)

        feature_cols = [c for c in class_data.columns if c != LABEL_COL]
        cond_cols = [c for c in fc if c != LABEL_COL and c in feature_cols] if fc else None
        selected_features = _pick_augmentation_features(
            feature_cols, rng, required_cols=cond_cols
        )

        self._selected_columns = selected_features
        X_syn = df_subset[selected_features]
        y_syn = df_subset[LABEL_COL]
        if fc:
            X_syn = enforce_feature_conditions_on_X(X_syn, fc, label_col=LABEL_COL)
        if target_value is not None:
            y_syn = pd.Series([target_value] * len(X_syn), name=LABEL_COL)
        return X_syn, y_syn

    def get_selected_columns(self):
        """Feature columns used in the last generate(); use same subset for evaluation."""
        return getattr(self, "_selected_columns", None)
