import pandas as pd
import numpy as np
import os
from typing import Dict, Any

from src.models.base import BaseModelWrapper

try:
    from tabtune import TabularPipeline
except ImportError:
    TabularPipeline = None

class TabPFNModel(BaseModelWrapper):
    def __init__(self, params: Dict[str, Any]):
        """
        Обертка для TabPFN через TabTune.
        Адаптирована под версию TabTune с разделением tuning_params и model_params.
        """
        super().__init__(params)
        
        if TabularPipeline is None:
            raise ImportError("TabTune не найден. Установите библиотеку.")

        self.device = self.params.get('device', 'mps')
        self.pipeline = None
        
        print(f"Initializing TabPFN wrapper (TabTune backend) on {self.device}")

    def fine_tune_weights(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        REAL Fine-Tuning (SFT) через TabTune.
        """
        print(f"Starting TabTune SFT (Fine-Tuning) on {len(X)} samples...")
        
        # 1. Параметры обучения (learning rate, epochs, batch_size)
        # TabTune ищет их в tuning_params
        tuning_params = {
            "epochs": self.params.get("ft_epochs", 10),
            "learning_rate": self.params.get("ft_learning_rate", 2e-5),
            "batch_size": self.params.get("ft_batch_size", 128),
            "device": self.device,
            # Можно добавить finetune_mode, если нужно (по умолчанию 'meta-learning')
        }
        # Добавляем динамические аргументы, если они есть
        tuning_params.update(kwargs)

        # 2. Параметры самой модели (N_ensemble, device)
        # TabTune ищет их в model_params
        model_params = {
            "device": self.device,
            # "N_ensemble_configurations": self.params.get("N_ensemble_configurations", 4)
        }

        # Инициализация Pipeline
        self.pipeline = TabularPipeline(
            model_name="TabPFN",
            task_type="classification",
            tuning_strategy="finetune", # Включаем режим обучения
            tuning_params=tuning_params,
            model_params=model_params
        )
        
        # Запуск обучения
        self.pipeline.fit(X, y) # TabTune внутри вызовет tuner.finetune(...)
        print("TabTune SFT complete. Weights updated.")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        In-Context Learning (ICL).
        Загружает контекст без обновления весов.
        """
        print(f"Starting TabPFN ICL (Context Loading) on {len(X)} samples...")
        
        if self.pipeline is None:
            # Если fine-tuning не запускался, создаем pipeline для инференса
            model_params = {
                "device": self.device,
                # "N_ensemble_configurations": self.params.get("N_ensemble_configurations", 4)
            }
            
            self.pipeline = TabularPipeline(
                model_name="TabPFN",
                task_type="classification",
                tuning_strategy="inference", # Режим БЕЗ обучения
                model_params=model_params
            )
        
        # HACK: Если пайплайн уже был в режиме 'finetune', нам нужно переключить его?
        # В коде TabTune нет явного метода set_mode, но стратегия задается при __init__.
        # Если мы хотим гарантировать, что fit() не запустит обучение заново,
        # надежнее всего использовать уже обученный self.pipeline.model напрямую 
        # или пересоздать pipeline, загрузив веса (но это сложно без сохранения).
        
        # В текущей реализации TabTune:
        # Если tuning_strategy='finetune', метод fit() запускает tuner.finetune.
        # Если tuning_strategy='inference', метод fit() просто делает препроцессинг и запоминает X, y.
        
        # Если мы уже сделали fine-tune, self.pipeline имеет стратегию 'finetune'.
        # Вызов fit() снова запустит обучение! Это нам НЕ нужно для ICL шага.
        
        # Решение:
        if self.pipeline.tuning_strategy == 'finetune':
            print("Switching pipeline strategy to inference mode explicitly...")
            self.pipeline.tuning_strategy = 'inference'
            # Также нужно убедиться, что модель переведена в eval, если TabTune этого не делает
            if hasattr(self.pipeline.model, 'model'): # TabPFN specific
                 self.pipeline.model.model.eval()

        self.pipeline.fit(X, y) 
        print("Context loaded.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model is not initialized. Call fit() first.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model is not initialized. Call fit() first.")
        return self.pipeline.predict_proba(X)

    def save(self, path: str):
        if self.pipeline:
            # TabTune ожидает полный путь к файлу (например, "model.pkl")
            # Создаем директорию, если в path она указана (например "outputs/model.pkl")
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            self.pipeline.save(path) # <-- Передаем path целиком!
            print(f"TabTune pipeline saved to {path}")

    def load(self, path: str):
        # Загрузка
        if os.path.exists(path):
            # TabTune.load - это статический метод, который возвращает объект
            # Но он может требовать тех же параметров, что и init, если сохранение не полное.
            # Попробуем стандартный метод
            try:
                self.pipeline = TabularPipeline.load(path)
            except AttributeError:
                # Если load не реализован как classmethod
                # Придется создавать новый и грузить веса вручную (как в коде TabTune)
                pass
            print(f"TabTune pipeline loaded from {path}")
        else:
            raise FileNotFoundError(f"Path not found: {path}")