import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from src.data_loader.base import BaseDataLoader

class CSVDataLoader(BaseDataLoader):
    def __init__(self, train_path: str, target_column: str, test_path: str = None, 
                 test_size: float = 0.2, random_state: int = 42, balance: bool = False,
                 sep: str = ','):
        """
        Args:
            train_path: Path to the training CSV file.
            target_column: Name of the target column.
            test_path: Path to the test CSV file (optional). If None, train_path is split.
            test_size: Size of the test set if splitting from train_path.
            random_state: Random state for reproducibility.
            balance: If True, performs undersampling of the majority class in the TRAIN set.
            sep: Delimiter for the CSV file.
        """
        super().__init__(target_column)
        self.train_path = train_path
        self.test_path = test_path
        self.test_size = test_size
        self.random_state = random_state
        self.balance = balance
        self.sep = sep

    def load_xy(self):
        print(f"Loading data from {self.train_path}...")
        
        if not os.path.exists(self.train_path):
             raise FileNotFoundError(f"Training file not found at: {self.train_path}")
        
        df = pd.read_csv(self.train_path, sep=self.sep)
        self._validate_data(df)

        if self.test_path:
             print(f"Loading test data from {self.test_path}...")
             if not os.path.exists(self.test_path):
                  raise FileNotFoundError(f"Test file not found at: {self.test_path}")
             
             df_test = pd.read_csv(self.test_path, sep=self.sep)
             self._validate_data(df_test)
             
             # Split into X and y
             X_train = df.drop(columns=[self.target_column])
             y_train = df[self.target_column]
             X_test = df_test.drop(columns=[self.target_column])
             y_test = df_test[self.target_column]
             
        else:
             print(f"Splitting data into train/test with test_size={self.test_size}...")
             X = df.drop(columns=[self.target_column])
             y = df[self.target_column]
             
             X_train, X_test, y_train, y_test = train_test_split(
                 X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
             )

        # Encode target if necessary
        if y_train.dtype == 'object' or str(y_train.dtype) == 'category':
             y_train = y_train.astype('category').cat.codes
             y_test = y_test.astype('category').cat.codes

        # Ensure y is a Series with a name
        y_train.name = self.target_column
        y_test.name = self.target_column

        if self.balance:
            print(f"Balancing Train set (Original size: {len(X_train)})...")
            X_train, y_train = self._balance_data(X_train, y_train)
            print(f"Balanced Train set size: {len(X_train)}")
        
        return X_train, y_train, X_test, y_test

    def _balance_data(self, X, y):
        """
        Simple Undersampling of the majority class to the size of the minority class.
        """
        target_col = y.name if y.name else "target"
        y = y.rename(target_col)
        
        train_data = pd.concat([X, y], axis=1)
        
        class_counts = train_data[target_col].value_counts()
        min_class_count = class_counts.min()
        
        print(f"   Counts per class: {class_counts.to_dict()}")
        print(f"   Downsampling to {min_class_count} samples per class.")
        
        balanced_dfs = []
        for label in class_counts.index:
            df_class = train_data[train_data[target_col] == label]
            
            if len(df_class) > min_class_count:
                df_class = resample(
                    df_class, 
                    replace=False, 
                    n_samples=min_class_count, 
                    random_state=self.random_state
                )
            balanced_dfs.append(df_class)
            
        balanced_data = pd.concat(balanced_dfs)
        balanced_data = balanced_data.sample(frac=1, random_state=self.random_state)
        
        y_balanced = balanced_data[target_col]
        X_balanced = balanced_data.drop(columns=[target_col])
        
        return X_balanced, y_balanced
