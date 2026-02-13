from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from src.data_loader.base import BaseDataLoader
import pandas as pd
import numpy as np

class OpenMLDataLoader(BaseDataLoader):
    def __init__(self, dataset_id: int, target_column: str, test_size: float = 0.2, 
                 random_state: int = 42, balance: bool = False):
        """
        Args:
            dataset_id: ID датасета на OpenML.
            target_column: Имя целевой колонки.
            test_size: Размер тестовой выборки.
            random_state: Сид для воспроизводимости.
            balance: Если True, делает undersampling мажоритарного класса в TRAIN выборке.
        """
        super().__init__(target_column)
        self.dataset_id = dataset_id
        self.test_size = test_size
        self.random_state = random_state
        self.balance = balance

    def load(self):
        print(f"Fetching dataset ID {self.dataset_id} from OpenML...")
        try:
            data = fetch_openml(data_id=self.dataset_id, as_frame=True, parser='auto')
        except Exception as e:
            print(f"Error fetching auto, trying dense: {e}")
            data = fetch_openml(data_id=self.dataset_id, as_frame=True)

        X = data.data
        y = data.target
        
        # Убедимся, что y - это Series, а X - DataFrame без y.
        if self.target_column in X.columns:
             y = X[self.target_column]
             X = X.drop(columns=[self.target_column])
        
        # Приводим y к кодам классов (0..N-1)
        if y.dtype == 'object' or str(y.dtype) == 'category':
             y = y.astype('category').cat.codes

        # Важно: Присваиваем имя явно, чтобы не потерять его
        y.name = self.target_column

        # Сначала делим на Train/Test (Test должен оставаться реальным и несбалансированным)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        if self.balance:
            print(f"Balancing Train set (Original size: {len(X_train)})...")
            X_train, y_train = self._balance_data(X_train, y_train)
            print(f"Balanced Train set size: {len(X_train)}")
        
        return X_train, y_train, X_test, y_test

    def _balance_data(self, X, y):
        """
        Простой Undersampling мажоритарного класса до размера миноритарного.
        """
        # 1. Гарантируем, что у Series есть имя перед объединением
        target_col = y.name if y.name else "target"
        y = y.rename(target_col)
        
        # 2. Объединяем X и y в один DataFrame
        train_data = pd.concat([X, y], axis=1)
        
        # 3. Считаем распределение классов
        class_counts = train_data[target_col].value_counts()
        min_class_count = class_counts.min()
        
        print(f"   Counts per class: {class_counts.to_dict()}")
        print(f"   Downsampling to {min_class_count} samples per class.")
        
        balanced_dfs = []
        for label in class_counts.index:
            df_class = train_data[train_data[target_col] == label]
            
            # Если примеров больше чем минимум, делаем resample (срезаем лишнее)
            if len(df_class) > min_class_count:
                df_class = resample(
                    df_class, 
                    replace=False, 
                    n_samples=min_class_count, 
                    random_state=self.random_state
                )
            balanced_dfs.append(df_class)
            
        # Собираем обратно
        balanced_data = pd.concat(balanced_dfs)
        # Перемешиваем строки
        balanced_data = balanced_data.sample(frac=1, random_state=self.random_state)
        
        # Разделяем обратно на X и y
        y_balanced = balanced_data[target_col]
        X_balanced = balanced_data.drop(columns=[target_col])
        
        return X_balanced, y_balanced