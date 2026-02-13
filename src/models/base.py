from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any

class BaseModelWrapper(ABC):
    """
    Абстрактная обертка для любой модели (TabPFN, XGBoost, CatBoost).
    Скрывает различия в API библиотек.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = None

    @abstractmethod
    def fine_tune_weights(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Дообучение уже существующей модели (Gradient-based или Incremental).
        """
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Запуск ICL модели.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Возвращает жесткие метки классов (0, 1, 2...).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Возвращает вероятности классов.
        Критично для расчета LogLoss и ROC-AUC.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Сохранение весов модели на диск."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Загрузка весов модели с диска."""
        pass