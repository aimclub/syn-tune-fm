import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import os
import sys

# --- Imports: Metrics ---
# Ensure you have src/metrics/factory.py and src/metrics/classification.py created
from src.metrics.factory import MetricFactory

# --- Imports: Models ---
from src.models.tabpfn_wrp import TabPFNModel

# --- Imports: Data Loaders ---
# Assuming you will populate src/data_loader/openml_loader.py
# If you haven't yet, you can use the BaseDataLoader as a placeholder or see the implementation below
try:
    from src.data_loader.openml_loader import OpenMLDataLoader
except ImportError:
    print("Warning: OpenMLDataLoader not found. Please implement it in src/data_loader/")
    OpenMLDataLoader = None

# --- Imports: Generators ---
# As you implement new generators (CTGAN, LLM, etc.), import them here.
# from src.generators.wrapper_ctgan import CTGANGenerator
# from src.generators.wrapper_llm import GreatLLMGenerator
# from src.generators.wrapper_gaussian import GaussianCopulaGenerator

class ExperimentRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        # 1. Setup Metrics
        # Reads the list of metric names from config (e.g., [accuracy, roc_auc])
        if 'metrics' in self.cfg.evaluation:
            self.metrics = MetricFactory.get_metrics(self.cfg.evaluation.metrics)
        else:
            self.metrics = []
            print("Warning: No metrics defined in config.")

    def _get_data_loader(self):
        """Factory method to get the data loader based on config."""
        name = self.cfg.dataset.name
        params = self.cfg.dataset.params
        
        if name == 'openml' or 'dataset_id' in params:
            if OpenMLDataLoader is None:
                raise ImportError("OpenMLDataLoader is not implemented or imported.")
            return OpenMLDataLoader(**params)
        
        # Add other loaders here (e.g., LocalCSVLoader)
        raise ValueError(f"Unknown dataset loader: {name}")

    def _get_generator(self):
        """Factory method to get the generator based on config."""
        name = self.cfg.generator.name
        params = self.cfg.generator.params

        # Logic to select the generator. 
        # You will uncomment these lines as you implement the wrapper files.
        
        # if name == 'gaussian':
        #     return GaussianCopulaGenerator(**params)
        # elif name == 'ctgan':
        #     return CTGANGenerator(**params)
        # elif name == 'llm_great':
        #     return GreatLLMGenerator(**params)
        
        raise ValueError(f"Generator '{name}' is not yet implemented in the runner factory.")

    def _get_model(self):
        """Factory method to get the model based on config."""
        name = self.cfg.model.name
        params = self.cfg.model.params

        if name == 'tabpfn':
            return TabPFNModel(params)
        
        # Future extension:
        # elif name == 'xgboost':
        #     return XGBoostModel(params)
            
        raise ValueError(f"Unknown model: {name}")

    def run(self):
        print(f"--- Starting Experiment: {self.cfg.dataset.name} + {self.cfg.generator.name} ---")

        # ---------------------------------------------------------
        # 1. Load Real Data
        # ---------------------------------------------------------
        print("\n[1/5] Loading Real Data...")
        loader = self._get_data_loader()
        X_train_real, y_train_real, X_test_real, y_test_real = loader.load()
        
        print(f"      Real Training Data Shape: {X_train_real.shape}")
        print(f"      Test Data Shape: {X_test_real.shape}")

        # ---------------------------------------------------------
        # 2. Train Generator & Generate Synthetic Data
        # ---------------------------------------------------------
        print(f"\n[2/5] Initializing Generator: {self.cfg.generator.name}")
        
        # Temporary check to prevent crash if you haven't implemented generators yet
        try:
            generator = self._get_generator()
            
            print("      Fitting generator on real train data...")
            generator.fit(X_train_real, y_train_real)
            
            n_samples = self.cfg.generator.get('n_samples', len(X_train_real))
            print(f"      Generating {n_samples} synthetic samples...")
            X_syn, y_syn = generator.generate(n_samples=n_samples)
            
            # Save synthetic data for debugging/analysis
            X_syn['target'] = y_syn
            X_syn.to_csv("synthetic_data_debug.csv", index=False)
            # Drop target back out for training
            X_syn = X_syn.drop(columns=['target'])
            
        except ValueError as e:
            print(f"      [SKIP] Generator step skipped due to error or missing implementation: {e}")
            print("      ! FALLBACK: Using Real Data for training to test the pipeline flow !")
            X_syn, y_syn = X_train_real, y_train_real

        # ---------------------------------------------------------
        # 3. Initialize & Train (Fine-tune) Model
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # 3. Initialize & Train (Fine-tune) Model
        # ---------------------------------------------------------
        print(f"\n[3/5] Initializing Model: {self.cfg.model.name}")
        model = self._get_model()
        
        print(f"      Fine-tuning model on {len(X_syn)} samples...")
        
        # Если модель поддерживает fine_tune_weights, используем его (для SFT)
        # Иначе используем обычный fit (для XGBoost/CatBoost)
        if hasattr(model, 'fine_tune_weights'):
            print("      >>> Triggering SFT (Gradient-based Fine-Tuning)...")
            # Передаем параметры обучения из конфига, если нужно
            ft_params = {
                'ft_epochs': self.cfg.model.get('params', {}).get('ft_epochs', 10),
                'ft_learning_rate': self.cfg.model.get('params', {}).get('ft_learning_rate', 2e-5)
            }
            model.fine_tune_weights(X_syn, y_syn, **ft_params)
        else:
            model.fit(X_syn, y_syn)

        # ---------------------------------------------------------
        # 4. Save Model
        # ---------------------------------------------------------
        print("\n[4/5] Saving Model...")
        # Hydra changes the working directory to outputs/date/time/, so we save locally
        save_path = "finetuned_model.pkl"
        model.save(save_path)
        print(f"      Model saved to: {os.getcwd()}/{save_path}")

        # ---------------------------------------------------------
        # 5. Evaluate on REAL Test Data
        # ---------------------------------------------------------
        print("\n[5/5] Evaluating on Real Test Data...")
        
        # Get predictions
        y_pred = model.predict(X_test_real)
        try:
            y_probs = model.predict_proba(X_test_real)
        except NotImplementedError:
            y_probs = None
            print("      Warning: Model does not support predict_proba")

        # Calculate metrics
        results = {}
        for metric in self.metrics:
            try:
                val = metric.calculate(y_test_real, y_pred, y_probs)
                results[metric.name] = val
                print(f"      >>> {metric.name}: {val:.4f}")
            except Exception as e:
                print(f"      Error calculating {metric.name}: {e}")

        return results
