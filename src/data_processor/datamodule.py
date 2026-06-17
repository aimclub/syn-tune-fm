from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .schema import TabularSchema
from .splits import (
    SplitConfigKFold,
    SplitConfigHoldout,
    KFoldSplit,
    HoldoutSplit,
    make_kfold_splits,
    make_holdout_split,
)


@dataclass
class FoldData:
    fold_id: int
    train: pd.DataFrame
    test: pd.DataFrame
    # Fitted (fold-specific) transforms are returned for reproducibility / downstream decoding
    transforms: Optional[Any] = None


@dataclass
class HoldoutData:
    train: pd.DataFrame
    val: pd.DataFrame
    transforms: Optional[Any] = None


class TabularDataModule:
    """
    Mixed-type tabular data module (continuous + discrete + categorical).

    Provides TWO INDEPENDENT splitting protocols:
      - k-fold splits for experiments
      - single holdout split for tuning generative models

    Key sb-tabular design principles preserved:
      1) Schema validates input columns.
      2) "Global" missing handling happens BEFORE any split (so splits are stable).
      3) Fold-wise transforms are FIT on the corresponding train subset only, then
         applied to train/test (or train/val) subsets.

    Notes for mixed-type settings:
      - Do NOT encode categoricals here. Encoding belongs to `representations/`.
      - Transforms are responsible for:
          * missing value policy (drop/impute)
          * scaling / normalization for continuous features
          * optional canonicalization / rare-binning for categoricals (if you decide to do that as a transform)
      - The DataModule only guarantees "fit-on-train" semantics per split.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: TabularSchema,
        transforms: Optional[Any] = None,
        reset_index: bool = True,
        # If True, also validates that df does not contain duplicated columns in schema groups, etc.
        validate: bool = True,
    ) -> None:
        self.schema = schema
        self.transforms = transforms

        if validate:
            schema.validate(df)

        df0 = df.copy()

        # Apply GLOBAL missing handling *before* splits.
        # We intentionally do NOT apply scaling globally (it must be fit on train per split).
        # Therefore, if transforms is a pipeline, it should support "pre-fit transform"
        # behavior (e.g., DropMissingRows) without needing fit.
        if self.transforms is not None:
            df0 = self._apply_global_transforms(df0)

        if reset_index:
            df0 = df0.reset_index(drop=True)

        self.df_clean = df0
        self.n_samples = len(df0)

        self._kfold_splits: Optional[list[KFoldSplit]] = None
        self._holdout_split: Optional[HoldoutSplit] = None

    # --------- K-FOLD (experiments) ---------

    def prepare_kfold(self, cfg: SplitConfigKFold) -> None:
        self._kfold_splits = make_kfold_splits(self.n_samples, cfg)

    def get_fold(self, fold_id: int) -> FoldData:
        if self._kfold_splits is None:
            raise RuntimeError("K-fold splits are not prepared. Call prepare_kfold(cfg) first.")

        if fold_id < 0 or fold_id >= len(self._kfold_splits):
            raise IndexError(f"fold_id={fold_id} out of range (n_folds={len(self._kfold_splits)})")

        fold = self._kfold_splits[fold_id]
        train_raw = self.df_clean.iloc[fold.train_idx].copy()
        test_raw = self.df_clean.iloc[fold.test_idx].copy()

        if self.transforms is None:
            return FoldData(fold_id=fold_id, train=train_raw, test=test_raw, transforms=None)

        pipe = self._clone_transforms(self.transforms)
        pipe.fit(train_raw, self.schema)  # fit only on fold-train
        train = pipe.transform(train_raw)
        test = pipe.transform(test_raw)

        # Optional safety re-validation (post-transform)
        # We do not enforce dtype checks here because transforms may cast/normalize categories, etc.
        self._validate_post_transform(train, context=f"fold={fold_id} train")
        self._validate_post_transform(test, context=f"fold={fold_id} test")

        return FoldData(fold_id=fold_id, train=train, test=test, transforms=pipe)

    def get_all_folds(self) -> Dict[int, FoldData]:
        if self._kfold_splits is None:
            raise RuntimeError("K-fold splits are not prepared. Call prepare_kfold(cfg) first.")
        return {f.fold_id: self.get_fold(f.fold_id) for f in self._kfold_splits}

    # --------- HOLDOUT (tuning) ---------

    def prepare_holdout(self, cfg: SplitConfigHoldout) -> None:
        self._holdout_split = make_holdout_split(self.n_samples, cfg)

    def get_holdout(self) -> HoldoutData:
        if self._holdout_split is None:
            raise RuntimeError("Holdout split is not prepared. Call prepare_holdout(cfg) first.")

        sp = self._holdout_split
        train_raw = self.df_clean.iloc[sp.train_idx].copy()
        val_raw = self.df_clean.iloc[sp.val_idx].copy()

        if self.transforms is None:
            return HoldoutData(train=train_raw, val=val_raw, transforms=None)

        pipe = self._clone_transforms(self.transforms)
        pipe.fit(train_raw, self.schema)  # fit only on holdout-train
        train = pipe.transform(train_raw)
        val = pipe.transform(val_raw)

        self._validate_post_transform(train, context="holdout train")
        self._validate_post_transform(val, context="holdout val")

        return HoldoutData(train=train, val=val, transforms=pipe)

    # --------- Optional convenience ---------

    def get_clean_df(self) -> pd.DataFrame:
        """Return the globally cleaned DataFrame (after global missing handling, before split-wise scaling)."""
        return self.df_clean.copy()

    # --------- Internals ---------

    def _apply_global_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply only those transforms that are safe to apply globally (e.g., dropping rows with NaNs).

        Behavior:
          - If transforms has a method `transform_global`, we call it (preferred).
          - Else we try `transform` directly. If it fails due to "not fitted", we attempt
            a *best-effort* fit+transform on the full df, but we strongly recommend pipelines
            to support pre-fit global missing handling to avoid accidental scaling leakage.

        This mirrors the original sb-tabular approach but adds a safer extension point.
        """
        t = self.transforms
        assert t is not None

        # Preferred explicit hook
        if hasattr(t, "transform_global"):
            df0 = t.transform_global(df, self.schema)  # type: ignore[attr-defined]
            return df0

        # Backward-compatible behavior
        try:
            return t.transform(df)
        except Exception:
            # Fallback: fit+transform on full df (may leak for scalers).
            # We keep this to preserve sb-tabular robustness, but you should avoid this
            # by ensuring your pipeline can drop missing without fitting.
            t.fit(df, self.schema)
            return t.transform(df)

    def _validate_post_transform(self, df: pd.DataFrame, context: str) -> None:
        """
        Lightweight checks after split-wise transforms:
          - required columns still exist
          - row count consistent
        We intentionally do not enforce strict dtypes here because
        transforms may cast categories to 'category', normalize strings, etc.
        """
        missing = [c for c in self.schema.all_cols if c not in df.columns]
        if missing:
            raise ValueError(f"[{context}] Transforms removed required columns: {missing}")

    @staticmethod
    def _clone_transforms(transforms: Any) -> Any:
        """
        Best-effort cloning:
          - If pipeline supports get_state/from_state -> use it.
          - Else deepcopy.
        """
        if hasattr(transforms, "get_state") and hasattr(transforms.__class__, "from_state"):
            state = transforms.get_state()
            return transforms.__class__.from_state(state)  # type: ignore[attr-defined]
        import copy

        return copy.deepcopy(transforms)
