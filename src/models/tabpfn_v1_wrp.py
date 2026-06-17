import pandas as pd
import numpy as np
import os
import torch
import joblib
from typing import Dict, Any

# --- MONKEY PATCH FOR COMPATIBILITY OF TABPFN V1 WITH NEW PYTORCH ---
import typing
import torch.nn.modules.transformer

# 1. Insert Optional into the internal PyTorch module
torch.nn.modules.transformer.Optional = typing.Optional

# 2. Patch torch.load for PyTorch 2.6+ to allow loading objects
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Force disable secure loading only for weights
    kwargs['weights_only'] = False 
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# 3. Patch torch.tensor for MPS (Apple Silicon) to intercept float64
_original_tensor = torch.tensor
def _mps_safe_tensor(data, *args, **kwargs):
    # If the input is a numpy array of format float64, convert it to float32
    if isinstance(data, np.ndarray) and data.dtype == np.float64:
        data = data.astype(np.float32)
    return _original_tensor(data, *args, **kwargs)
torch.tensor = _mps_safe_tensor
# -----------------------------------------------------------------
from tabpfn import TabPFNClassifier
from src.models.base import BaseModelWrapper

class TabPFNModelV1(BaseModelWrapper):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.device = self.params.get('device', 'cpu')
        self.n_ensemble = self.params.get('n_ensemble', 1)
        
        print(f"Initializing TabPFN v1 on {self.device}")
        self.classifier = TabPFNClassifier(device=self.device, N_ensemble_configurations=self.n_ensemble)
        self.is_fitted = False

    def get_pytorch_model(self) -> torch.nn.Module:
        if hasattr(self.classifier, 'model'):
            # Extract the neural network itself
            if isinstance(self.classifier.model, tuple):
                pytorch_model = self.classifier.model[2]
            else:
                pytorch_model = self.classifier.model
                
            # Forcibly send the model weights to our device (MPS/CUDA)
            pytorch_model.to(self.device)
            return pytorch_model
            
        raise AttributeError("Could not find internal PyTorch model.")

    def forward_pass(self, batch_X: torch.Tensor, batch_y: torch.Tensor):
        model = self.get_pytorch_model()
        
        seq_len = batch_X.shape[0]
        half = seq_len // 2
        
        if half == 0:
            raise ValueError("Batch size too small for split.")
            
        # 1. MANDATORY NORMALIZATION (Saves gradients from exploding)
        mean = batch_X.mean(dim=0, keepdim=True)
        std = batch_X.std(dim=0, keepdim=True) + 1e-8
        batch_X = (batch_X - mean) / std
            
        # 2. TABPFN v1 FEATURE PADDING (Up to 100 columns)
        num_features = batch_X.shape[1]
        if num_features < 100:
            import torch.nn.functional as F
            batch_X = F.pad(batch_X, (0, 100 - num_features), value=0.0)
            
        x_3d = batch_X.unsqueeze(1)
        y_3d = batch_y.unsqueeze(1).float()
        
        # No NaNs! single_eval_pos itself blocks data leakage through Attention.
        src = (x_3d, y_3d)
        
        logits = model(src, single_eval_pos=half)
        
        if isinstance(logits, tuple):
            logits = logits[0]
            
        if logits.dim() == 3:
            logits = logits.squeeze(1)
            
        if logits.shape[0] == seq_len - half:
            return logits, batch_y[half:]
        elif logits.shape[0] == seq_len:
            return logits[half:], batch_y[half:]
        else:
            return logits, batch_y[-logits.shape[0]:]

    def fine_tune(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        pass 

    def fit_context(self, X: pd.DataFrame, y: pd.Series):
        print(f"Starting TabPFN ICL (Context Loading) on {len(X)} samples...")
        
        # Cast X to float32 for MPS
        X_32 = X.astype(np.float32) if hasattr(X, 'astype') else X
        
        # Cast y to int32 for MPS (64-bit ints also crash Apple Silicon)
        y_32 = y.astype(np.int32) if hasattr(y, 'astype') else y
        
        self.classifier.fit(X_32, y_32)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_float32 = X.astype(np.float32) if hasattr(X, 'astype') else X
        preds = self.classifier.predict(X_float32)
        
        # --- SAFE INVERSION (Only for binary tasks) ---
        if hasattr(self.classifier, 'classes_') and len(self.classifier.classes_) == 2:
            # Check if predictions are really only 0 and 1
            if set(np.unique(preds)).issubset({0, 1}):
                preds = 1 - preds
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_float32 = X.astype(np.float32) if hasattr(X, 'astype') else X
        probs = self.classifier.predict_proba(X_float32)
        
        # --- SAFE INVERSION ---
        if probs.shape[1] == 2:
            probs = probs[:, [1, 0]]
        return probs

    def save(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self.classifier, path)

    def load(self, path: str):
        self.classifier = joblib.load(path)
        self.is_fitted = True