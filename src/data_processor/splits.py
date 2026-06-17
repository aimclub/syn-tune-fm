
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


@dataclass(frozen=True)
class SplitConfigKFold:
    """Config for k-fold splitting used in experiments."""
    n_splits: int = 5
    shuffle: bool = True
    random_seed: int = 42


@dataclass(frozen=True)
class SplitConfigHoldout:
    """Config for a single train/val split used for tuning generative models."""
    val_size: float = 0.2
    shuffle: bool = True
    random_seed: int = 42


@dataclass(frozen=True)
class KFoldSplit:
    """One CV fold split (positional indices)."""
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class HoldoutSplit:
    """One fixed train/val split (positional indices)."""
    train_idx: np.ndarray
    val_idx: np.ndarray


def make_kfold_splits(n_samples: int, cfg: SplitConfigKFold) -> List[KFoldSplit]:
    if n_samples <= 1:
        raise ValueError("n_samples must be > 1")
    if cfg.n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    all_idx = np.arange(n_samples)
    kf = KFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_seed)

    folds: List[KFoldSplit] = []
    for fold_id, (train_idx, test_idx) in enumerate(kf.split(all_idx)):
        folds.append(
            KFoldSplit(
                fold_id=fold_id,
                train_idx=np.asarray(train_idx, dtype=int),
                test_idx=np.asarray(test_idx, dtype=int),
            )
        )
    return folds


def make_holdout_split(n_samples: int, cfg: SplitConfigHoldout) -> HoldoutSplit:
    if n_samples <= 1:
        raise ValueError("n_samples must be > 1")
    if not (0.0 < cfg.val_size < 1.0):
        raise ValueError("val_size must be in (0, 1)")

    all_idx = np.arange(n_samples)
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=cfg.val_size,
        shuffle=cfg.shuffle,
        random_state=cfg.random_seed,
    )
    return HoldoutSplit(
        train_idx=np.asarray(train_idx, dtype=int),
        val_idx=np.asarray(val_idx, dtype=int),
    )

def apply_imbalance(X: pd.DataFrame, y: pd.Series, minority_fraction: float) -> tuple[pd.DataFrame, pd.Series]:
    """
    Artificially creates a class imbalance in the dataset (supports multiclass).
    
    Parameters:
    - X (pd.DataFrame): Features.
    - y (pd.Series): Target variable.
    - minority_fraction (float): The desired fraction of the total dataset for EACH minority class.
      For example, 0.1 means that each rare class will make up exactly 10% of the resulting data.
      
    Returns:
    - Tuple[pd.DataFrame, pd.Series]: Downsampled X and y with the specified imbalance level.
    """
    class_counts = y.value_counts()
    n_classes = len(class_counts)
    
    # Imbalance is not possible if there is only one class
    if n_classes < 2:
        return X, y  
        
    k_minority = n_classes - 1 # Number of minority classes

    # 1. Validate the requested fraction
    if minority_fraction <= 0 or (k_minority * minority_fraction) >= 1.0:
        raise ValueError(
            f"Invalid minority_fraction={minority_fraction}. "
            f"For {k_minority} minority classes, their sum of fractions must be < 1.0"
        )

    # 2. Identify the majority class (the one with the most samples)
    majority_class = class_counts.idxmax()
    n_maj_current = class_counts[majority_class]

    # 3. Calculate the ideal target size for minority classes (assuming we keep all majority samples)
    # Formula: N_min = N_maj * (P / (1 - K * P))
    ideal_min_target = n_maj_current * (minority_fraction / (1.0 - k_minority * minority_fraction))

    # 4. Check if we have enough real data for all minority classes
    # Find the most "limiting" minority class
    min_available_ratio = 1.0
    for cls in class_counts.index:
        if cls != majority_class:
            if class_counts[cls] < ideal_min_target:
                ratio = class_counts[cls] / ideal_min_target
                if ratio < min_available_ratio:
                    min_available_ratio = ratio

    # 5. Compute the final class sizes (scaled down if limited by available minority samples)
    n_min_target = math.floor(ideal_min_target * min_available_ratio)
    n_maj_target = math.floor(n_maj_current * min_available_ratio)

    # 6. Collect indices to keep
    indices_to_keep = []
    
    for cls in class_counts.index:
        target_size = n_maj_target if cls == majority_class else n_min_target
        # Randomly sample the required number of elements
        idx_cls = y[y == cls].sample(n=target_size, random_state=42).index
        indices_to_keep.extend(idx_cls)

    # 7. Return the filtered data
    return X.loc[indices_to_keep], y.loc[indices_to_keep]