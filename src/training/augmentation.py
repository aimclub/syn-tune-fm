import pandas as pd

class DataAugmenter:
    """
    Handles data mixing strategies (e.g., combining real and synthetic data).
    """
    def __init__(self, strategy: str = "concat"):
        self.strategy = strategy

    def mix(self, X_syn: pd.DataFrame, y_syn: pd.Series, X_real: pd.DataFrame, y_real: pd.Series):
        """
        Mixes synthetic and real datasets based on the defined strategy.
        """
        if self.strategy == "concat":
            print(f"Augmentation: Concatenating {len(X_syn)} synthetic and {len(X_real)} real samples.")
            X_combined = pd.concat([X_syn, X_real], ignore_index=True)
            y_combined = pd.concat([y_syn, y_real], ignore_index=True)
            return X_combined, y_combined
        
        # Future strategies: 'curriculum', 'weighted', etc.
        raise ValueError(f"Unknown augmentation strategy: {self.strategy}")