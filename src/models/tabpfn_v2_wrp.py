import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
from tabpfn.model_loading import ModelSource, prepend_cache_path

from src.models.base import BaseModelWrapper


class TabPFNModelV2(BaseModelWrapper):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.device = self.params.get("device", "cpu")
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.task_type = self.params.get("task_type", "classification")
        self.n_estimators = self.params.get("n_estimators", 4)

        self.is_fitted = False
        self.model = None

    def _use_latest_hf_model(self) -> bool:
        """If True, use library defaults (model_path='auto' → latest TabPFN, e.g. 2.6).

        If False (default), pin Prior-Labs/tabpfn_2_5 checkpoints (TabPFN v2.5 weights).
        That avoids pulling tabpfn_2_6, but a one-time license / HF auth step may still
        appear for any Prior-Labs download in recent tabpfn builds — use HF_TOKEN, etc.
        """
        v = str(self.params.get("tabpfn_model_version", "v2.5")).strip().lower()
        if v in ("v2.5", "2.5", "v2_5"):
            return False
        if v in ("auto", "latest"):
            return True
        raise ValueError(
            "tabpfn_model_version must be 'v2.5' (default) or 'auto' (latest default model)."
        )

    def _finetune_extra_estimator_kwargs(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {"n_estimators": self.n_estimators}
        if self._use_latest_hf_model():
            return extra
        if self.task_type == "regression":
            extra["model_path"] = prepend_cache_path(
                ModelSource.get_regressor_v2_5().default_filename
            )
        else:
            extra["model_path"] = prepend_cache_path(
                ModelSource.get_classifier_v2_5().default_filename
            )
        return extra

    def fine_tune(self, X: pd.DataFrame, y: pd.Series, epochs: int = 10, learning_rate: float = 1e-5):
        """Uses the native fine-tuning mechanism (SFT) from newer versions of TabPFN."""
        print(f"      [V2 Mode] Initializing native FinetunedTabPFN on {self.device}...")
        extra_kw = self._finetune_extra_estimator_kwargs()
        if self.task_type == "regression":
            self.model = FinetunedTabPFNRegressor(
                device=self.device,
                epochs=epochs,
                learning_rate=learning_rate,
                extra_regressor_kwargs=extra_kw,
            )
        else:
            self.model = FinetunedTabPFNClassifier(
                device=self.device,
                epochs=epochs,
                learning_rate=learning_rate,
                extra_classifier_kwargs=extra_kw,
            )

        print("      [V2 Mode] Starting native Fine-Tuning loop...")

        run_id = uuid.uuid4().hex[:8]
        output_dir = Path(f"tabpfn_temp_checkpoints_{run_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.model.fit(X, y, output_dir=output_dir)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            if self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        self.is_fitted = True

    def fit_context(self, X: pd.DataFrame, y: pd.Series):
        """Standard ICL training without changing weights"""
        print(f"Starting TabPFN V2 ICL (Context Loading) on {len(X)} samples...")
        if self._use_latest_hf_model():
            if self.task_type == "regression":
                self.model = TabPFNRegressor(device=self.device, n_estimators=self.n_estimators)
            else:
                self.model = TabPFNClassifier(device=self.device, n_estimators=self.n_estimators)
        else:
            if self.task_type == "regression":
                self.model = TabPFNRegressor.create_default_for_version(
                    ModelVersion.V2_5,
                    device=self.device,
                    n_estimators=self.n_estimators,
                )
            else:
                self.model = TabPFNClassifier.create_default_for_version(
                    ModelVersion.V2_5,
                    device=self.device,
                    n_estimators=self.n_estimators,
                )

        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        X_np = X.values if isinstance(X, pd.DataFrame) else X

        batch_size = 500
        predictions = []
        for i in range(0, len(X_np), batch_size):
            batch = self.model.predict(X_np[i : i + batch_size])
            predictions.append(batch)
        return np.concatenate(predictions, axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        if self.task_type == "regression":
            raise ValueError("predict_proba is not available for regression tasks.")
        X_np = X.values if isinstance(X, pd.DataFrame) else X

        batch_size = 500
        predictions = []
        for i in range(0, len(X_np), batch_size):
            batch = self.model.predict_proba(X_np[i : i + batch_size])
            predictions.append(batch)
        return np.vstack(predictions)

    def save(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        print(f"      Saving TabPFN V2 model to {path}...")
        joblib.dump(self.model, path)

    def load(self, path: str):
        print(f"      Loading TabPFN V2 model from {path}...")
        self.model = joblib.load(path)
        self.is_fitted = True
