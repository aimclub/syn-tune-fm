from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd

from src.data_processor.datamodule import TabularDataModule
from src.data_processor.schema import TabularSchema


class BaseDataLoader(ABC):
    """
    Tabular loaders expose train/test feature matrices and labels via `load_xy()`.
    `load()` wraps that split into a `TabularDataModule` for the experiment pipeline.
    """

    def __init__(self, target_column: str):
        """
        Args:
            target_column (str): Target column name (label).
        """
        self.target_column = target_column

    @abstractmethod
    def load_xy(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Loads data, performs preprocessing and splitting into train/test.

        Returns:
            X_train, y_train, X_test, y_test
        """
        pass

    def _to_datamodule(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> TabularDataModule:
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        tcol = getattr(self, "target_column", None) or (
            y_train.name if y_train.name else "target"
        )
        y_train = y_train.rename(tcol)
        y_test = y_test.rename(tcol)
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)
        df_full = pd.concat([df_train, df_test], ignore_index=True)
        schema = TabularSchema.infer_from_dataframe(df_full, target_col=tcol)
        return TabularDataModule(df_full, schema, transforms=None)

    def load(self) -> TabularDataModule:
        return self._to_datamodule(*self.load_xy())

    def _validate_data(self, df: pd.DataFrame):
        """Helper method to check the presence of the target column."""
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
