from typing import Optional

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from src.data_loader.base import BaseDataLoader
import pandas as pd
import numpy as np

class OpenMLDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset_id: int,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        balance: bool = False,
        target_quantile_bins: Optional[int] = None,
        hpo_js_single_column_default: Optional[str] = None,
        task_type: Optional[str] = None,
        max_rows: Optional[int] = None,
    ):
        """
        Args:
            dataset_id: OpenML dataset ID.
            target_column: Target column name.
            test_size: Test set size.
            random_state: Seed for reproducibility.
            balance: If True, performs undersampling of the majority class in the TRAIN set.
            target_quantile_bins: If set, numeric y is converted to quantile-based integer classes
                (so stratify + class-conditional synthetic balancing behave like on classification).
            hpo_js_single_column_default: Documented default for HPO single-column mode; runner reads
                cfg.dataset.params, not this attribute.
            task_type: If ``regression`` and ``target_quantile_bins`` is unset, y stays numeric (float)
                for TabPFNRegressor + SDV-style synthetic augmentation (no quantile binning).
            max_rows: If set, randomly subsample the full dataset to this many rows before split
                (useful for large OpenML sets such as diamonds).
        """
        super().__init__(target_column)
        self.dataset_id = dataset_id
        self.test_size = test_size
        self.random_state = random_state
        self.balance = balance
        self.target_quantile_bins = target_quantile_bins
        self.hpo_js_single_column_default = hpo_js_single_column_default
        self.task_type = (task_type or "classification").strip().lower()
        self.max_rows = int(max_rows) if max_rows is not None else None

    def load_xy(self):
        print(f"Fetching dataset ID {self.dataset_id} from OpenML...")
        try:
            data = fetch_openml(data_id=self.dataset_id, as_frame=True, parser='auto')
        except Exception as e:
            print(f"Error fetching auto, trying dense: {e}")
            data = fetch_openml(data_id=self.dataset_id, as_frame=True)

        X = data.data
        y = data.target
        
        # Ensure y is a Series and X is a DataFrame without y.
        if self.target_column in X.columns:
             y = X[self.target_column]
             X = X.drop(columns=[self.target_column])

        if self.max_rows is not None and len(X) > self.max_rows:
            sample_idx = X.sample(n=self.max_rows, random_state=self.random_state).index
            X = X.loc[sample_idx].reset_index(drop=True)
            y = y.loc[sample_idx].reset_index(drop=True)
            print(f"   Subsampled to max_rows={self.max_rows} (random_state={self.random_state})")

        # Numeric target: quantile bins (classification-style pipeline) OR true regression float.
        if self.target_quantile_bins is not None:
            y_num = pd.to_numeric(y, errors="coerce")
            mask = y_num.notna()
            X = X.loc[mask].reset_index(drop=True)
            y_num = y_num.loc[mask].reset_index(drop=True)
            y_binned = pd.qcut(
                y_num,
                q=int(self.target_quantile_bins),
                labels=False,
                duplicates="drop",
            )
            if y_binned.isna().any():
                y_binned = y_binned.fillna(-1).astype(int)
            y = y_binned.astype(int)
            print(
                f"   Target binned into quantile classes (q={self.target_quantile_bins}), "
                f"n_unique={y.nunique()}"
            )
        elif self.task_type == "regression":
            y_num = pd.to_numeric(y, errors="coerce")
            mask = y_num.notna()
            X = X.loc[mask].reset_index(drop=True)
            y = y_num.loc[mask].reset_index(drop=True).astype(float)
            print(
                f"   Regression target kept continuous (n={len(y)}, n_unique={y.nunique()})"
            )
        elif y.dtype == "object" or str(y.dtype) == "category":
            y = y.astype("category").cat.codes

        # Important: Assign the name explicitly so as not to lose it
        y.name = self.target_column

        stratify = self._stratify_arg(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )

        if self.balance:
            print(f"Balancing Train set (Original size: {len(X_train)})...")
            X_train, y_train = self._balance_data(X_train, y_train)
            print(f"Balanced Train set size: {len(X_train)}")
        
        return X_train, y_train, X_test, y_test

    def _balance_data(self, X, y):
        """
        Simple undersampling of the majority class to the size of the minority class.
        """
        # 1. Ensure the Series has a name before concatenating
        target_col = y.name if y.name else "target"
        y = y.rename(target_col)
        
        # 2. Concatenate X and y into a single DataFrame
        train_data = pd.concat([X, y], axis=1)
        
        # 3. Calculate class distribution
        class_counts = train_data[target_col].value_counts()
        min_class_count = class_counts.min()
        
        print(f"   Counts per class: {class_counts.to_dict()}")
        print(f"   Downsampling to {min_class_count} samples per class.")
        
        balanced_dfs = []
        for label in class_counts.index:
            df_class = train_data[train_data[target_col] == label]
            
            # If there are more examples than the minimum, resample (cut off excess)
            if len(df_class) > min_class_count:
                df_class = resample(
                    df_class, 
                    replace=False, 
                    n_samples=min_class_count, 
                    random_state=self.random_state
                )
            balanced_dfs.append(df_class)
            
        # Put back together
        balanced_data = pd.concat(balanced_dfs)
        # Shuffle the rows
        balanced_data = balanced_data.sample(frac=1, random_state=self.random_state)
        
        # Split back into X and y
        y_balanced = balanced_data[target_col]
        X_balanced = balanced_data.drop(columns=[target_col])
        
        return X_balanced, y_balanced

    @staticmethod
    def _stratify_arg(y: pd.Series):
        """sklearn stratify=None unless every class has >=2 samples."""
        vc = y.value_counts()
        if len(vc) < 2 or vc.min() < 2:
            return None
        return y