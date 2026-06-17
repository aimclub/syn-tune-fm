"""Tabular DDPM: scale numeric / ordinal-encode categoricals, train noise predictor, sample reverse diffusion."""
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from src.generators.base import BaseDataGenerator
from src.generators.base import enforce_feature_conditions_on_X, train_subset_from_conditions

warnings.filterwarnings("ignore")

LABEL_COL = "target"

import torch
import torch.nn as nn


def _get_num_cat_columns(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return num_cols, cat_cols


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, CategoricalDtype):
            df[col] = df[col].astype("object")
    return df


class _TabularDenoiser(nn.Module):
    """MLP predicting noise from noisy x and timestep t."""

    def __init__(self, dim: int, hidden: int = 256, n_steps: int = 100):
        super().__init__()
        self.time_embed = nn.Embedding(n_steps, hidden)
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


def _cosine_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    steps = np.arange(T + 1, dtype=np.float64) / T
    f = np.cos(((steps + s) / (1 + s)) * np.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-4, 0.999)


def _ensure_classes_presence(X_syn: pd.DataFrame, y_syn: pd.Series, X_real: pd.DataFrame, y_real: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    syn_df = X_syn.copy()
    syn_df[LABEL_COL] = y_syn.values
    real_df = X_real.copy()
    real_df[LABEL_COL] = y_real.values
    unique_real = set(y_real.unique())
    unique_syn = set(y_syn.unique())
    if len(unique_syn) < len(unique_real):
        missing = unique_real - unique_syn
        for cls in missing:
            real_samples = real_df[real_df[LABEL_COL] == cls]
            if len(real_samples) > 0:
                n_inject = min(3, len(real_samples))
                sample_to_inject = real_samples.sample(n=n_inject, replace=False)
                syn_df = pd.concat([sample_to_inject, syn_df.iloc[n_inject:]], ignore_index=True)
    return syn_df.drop(columns=[LABEL_COL], errors="ignore"), syn_df[LABEL_COL]


class TabularDiffusionGenerator(BaseDataGenerator):
    """DDPM on a flat numeric tensor after StandardScaler + OrdinalEncoder."""

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = None,
        epochs: int = 100,
        n_steps: int = 100,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        batch_size: int = 256,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            seed=seed, n_samples=n_samples, epochs=epochs,
            n_steps=n_steps, hidden_dim=hidden_dim, lr=lr, batch_size=batch_size,
            device=device, **kwargs
        )
        self.seed = seed
        self.n_samples = n_samples
        self.epochs = epochs
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._scaler_num = None
        self._encoder_cat = None
        self._num_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._cat_mappings: dict = {}
        self._dim: int = 0
        self._model: Optional[nn.Module] = None
        self._betas: Optional[np.ndarray] = None
        self._alphas_cumprod: Optional[np.ndarray] = None
        self._X_real = None
        self._y_real = None
        self._column_order: List[str] = []

    def _encode(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        X = _fix_dtypes(X)
        parts = []
        if self._num_cols:
            X_num_df = X[self._num_cols].fillna(0)
            parts.append(self._scaler_num.transform(X_num_df))
        if self._cat_cols:
            X_cat_df = X[self._cat_cols].fillna("__nan__").astype(str)
            parts.append(self._encoder_cat.transform(X_cat_df))
        y_arr = np.asarray(y).reshape(-1, 1)
        if y.dtype == "object" or str(y.dtype) == "category":
            y_codes = pd.Series(y).astype("category").cat.codes.values.reshape(-1, 1)
        else:
            y_codes = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if self._num_cols and hasattr(self._scaler_num, "scale_") and y_codes.dtype != np.float64:
            y_codes = y_codes.astype(np.float64)
        parts.append(y_codes)
        return np.hstack(parts).astype(np.float32)

    def _decode(self, Z: np.ndarray) -> Tuple[pd.DataFrame, pd.Series]:
        idx = 0
        cols = []
        if self._num_cols:
            n_num = len(self._num_cols)
            X_num = self._scaler_num.inverse_transform(
                pd.DataFrame(Z[:, idx : idx + n_num], columns=self._num_cols)
            )
            for j, c in enumerate(self._num_cols):
                cols.append(pd.Series(X_num[:, j], name=c))
            idx += n_num
        if self._cat_cols:
            n_cat = len(self._cat_cols)
            X_cat_codes = np.round(Z[:, idx:idx + n_cat]).astype(int)
            for j, c in enumerate(self._cat_cols):
                cats = self._encoder_cat.categories_[j]
                max_idx = len(cats) - 1
                code = np.clip(X_cat_codes[:, j], 0, max_idx)
                vals = np.array([str(cats[int(k)]) for k in code])
                cols.append(pd.Series(vals, name=c))
            idx += n_cat
        y_vals = Z[:, idx]
        if getattr(self, "_y_is_cat", False) and self._y_classes is not None:
            k = np.round(y_vals).astype(int)
            k = np.clip(k, 0, len(self._y_classes) - 1)
            y_series = pd.Series(np.array(self._y_classes)[k], name=LABEL_COL)
        else:
            y_series = pd.Series(y_vals.astype(np.float64), name=LABEL_COL)
        X_df = pd.concat(cols, axis=1) if cols else pd.DataFrame()
        return X_df, y_series

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabularDiffusionGenerator":
        X = _fix_dtypes(X)
        self._num_cols, self._cat_cols = _get_num_cat_columns(X)
        self._column_order = self._num_cols + self._cat_cols
        self._X_real = X
        self._y_real = y

        if y.dtype == "object" or str(y.dtype) == "category":
            self._y_is_cat = True
            self._y_classes = pd.Series(y).astype("category").cat.categories.tolist()
        elif np.issubdtype(y.dtype, np.integer):
            self._y_is_cat = True
            self._y_classes = sorted(pd.Series(y).unique().tolist())
        else:
            self._y_is_cat = False
            self._y_classes = None

        train_df = X.copy()
        train_df[LABEL_COL] = y.values
        if self._num_cols:
            self._scaler_num = StandardScaler()
            self._scaler_num.fit(train_df[self._num_cols].fillna(0))
        else:
            self._scaler_num = None
        if self._cat_cols:
            self._encoder_cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self._encoder_cat.fit(train_df[self._cat_cols].fillna("__nan__").astype(str))
            self._cat_mappings = {}
            for j, c in enumerate(self._cat_cols):
                cats = self._encoder_cat.categories_[j]
                self._cat_mappings[c] = {i: str(cats[i]) for i in range(len(cats))}
        else:
            self._encoder_cat = None

        Z = self._encode(X, y)
        self._dim = Z.shape[1]
        self._betas = _cosine_beta_schedule(self.n_steps)
        self._alphas_cumprod = np.cumprod(1 - self._betas)
        self._alphas_cumprod = np.concatenate([[1.0], self._alphas_cumprod])

        self._model = _TabularDenoiser(self._dim, hidden=self.hidden_dim, n_steps=self.n_steps).to(self.device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        torch.manual_seed(self.seed)
        self._model.train()
        for ep in range(self.epochs):
            perm = np.random.permutation(len(Z))
            for start in range(0, len(Z), self.batch_size):
                idx = perm[start:start + self.batch_size]
                x0 = torch.from_numpy(Z[idx]).float().to(self.device)
                t = torch.randint(0, self.n_steps, (x0.shape[0],), device=self.device).long()
                noise = torch.randn_like(x0, device=self.device)
                alpha = torch.from_numpy(self._alphas_cumprod[t.cpu().numpy()]).float().to(self.device).unsqueeze(1)
                x_noisy = torch.sqrt(alpha) * x0 + torch.sqrt(1 - alpha) * noise
                pred_noise = self._model(x_noisy, t)
                loss = nn.functional.mse_loss(pred_noise, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.is_fitted = True
        return self

    def generate(self, n_samples: int = None, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        n = n_samples or kwargs.get("n_samples") or self.n_samples or len(self._X_real)
        self._model.eval()
        torch.manual_seed(self.params.get("seed", self.seed))
        z_clamp = 10.0
        with torch.no_grad():
            z = torch.randn(n, self._dim, device=self.device)
            for t in reversed(range(self.n_steps)):
                t_b = torch.full((n,), t, dtype=torch.long, device=self.device)
                pred_noise = self._model(z, t_b)
                beta_t = float(self._betas[t])
                alpha_t = float(self._alphas_cumprod[t])
                alpha_t_next = float(self._alphas_cumprod[t + 1])
                denom = np.sqrt(1 - alpha_t)
                if denom < 1e-8:
                    denom = 1e-8
                z = (z - (beta_t / denom) * pred_noise) / (np.sqrt(alpha_t) + 1e-8)
                z = torch.clamp(z, -z_clamp, z_clamp)
                if t > 0:
                    sigma_t = ((1 - alpha_t_next) / (1 - alpha_t + 1e-8) * beta_t) ** 0.5
                    z = z + sigma_t * torch.randn_like(z, device=self.device)
                    z = torch.clamp(z, -z_clamp, z_clamp)
            Z_syn = z.cpu().numpy()
            if not np.isfinite(Z_syn).all():
                Z_syn = np.nan_to_num(Z_syn, nan=0.0, posinf=z_clamp, neginf=-z_clamp)
        X_syn, y_syn = self._decode(Z_syn)
        X_syn, y_syn = _ensure_classes_presence(X_syn, y_syn, self._X_real, self._y_real)
        if len(X_syn) > n:
            X_syn, y_syn = X_syn.iloc[:n], y_syn.iloc[:n]
        elif len(X_syn) < n:
            extra = n - len(X_syn)
            X_extra = self._X_real.sample(n=min(extra, len(self._X_real)), replace=True)
            y_extra = self._y_real.loc[X_extra.index]
            X_syn = pd.concat([X_syn, X_extra], ignore_index=True)
            y_syn = pd.concat([y_syn, y_extra], ignore_index=True)
        return X_syn, y_syn

    def conditional_sampling(
        self,
        n_samples: int,
        target_value: Optional[int] = None,
        feature_conditions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        return super().conditional_sampling(
            n_samples,
            target_value=target_value,
            feature_conditions=feature_conditions,
            **kwargs,
        )

    def _generate_conditional(
        self,
        n_samples: int,
        target_value: Optional[int] = None,
        feature_conditions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        fc = dict(feature_conditions or {})
        if not fc and target_value is None:
            return self.generate(n_samples, **kwargs)
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        train_df = self._X_real.copy()
        train_df[LABEL_COL] = self._y_real.values
        train_subset = train_subset_from_conditions(
            train_df,
            target_value=target_value,
            feature_conditions=fc,
            label_col=LABEL_COL,
        )
        X_subset = train_subset.drop(columns=[LABEL_COL]).reset_index(drop=True)
        y_subset = train_subset[LABEL_COL].reset_index(drop=True)

        X_syn, y_syn = self._sample_diffusion_on_subset(
            X_subset,
            y_subset,
            n_samples,
            y_constant=target_value if target_value is not None else None,
        )
        if fc:
            X_syn = enforce_feature_conditions_on_X(X_syn, fc, label_col=LABEL_COL)
        return X_syn, y_syn

    def _sample_diffusion_on_subset(
        self,
        X_subset: pd.DataFrame,
        y_subset: pd.Series,
        n_samples: int,
        y_constant: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Train a short-lived diffusion on (X_subset, y_subset) and sample."""
        if len(X_subset) == 0:
            raise ValueError("Empty subset for conditional diffusion.")

        temp_gen = TabularDiffusionGenerator(
            seed=self.seed,
            epochs=max(10, self.epochs // 2),  # Fewer epochs for subset
            n_steps=self.n_steps,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            batch_size=self.batch_size,
            device=self.device,
        )
        temp_gen._num_cols = self._num_cols
        temp_gen._cat_cols = self._cat_cols
        temp_gen._column_order = self._column_order
        temp_gen._X_real = X_subset
        temp_gen._y_real = y_subset
        temp_gen._y_is_cat = self._y_is_cat
        temp_gen._y_classes = self._y_classes
        
        temp_gen._scaler_num = self._scaler_num
        temp_gen._encoder_cat = self._encoder_cat
        temp_gen._cat_mappings = self._cat_mappings
        temp_gen._dim = self._dim
        temp_gen._betas = self._betas
        temp_gen._alphas_cumprod = self._alphas_cumprod
        
        temp_gen._model = _TabularDenoiser(temp_gen._dim, hidden=temp_gen.hidden_dim, n_steps=temp_gen.n_steps).to(temp_gen.device)
        opt = torch.optim.Adam(temp_gen._model.parameters(), lr=temp_gen.lr)
        torch.manual_seed(temp_gen.seed)
        temp_gen._model.train()
        
        Z = temp_gen._encode(X_subset, y_subset)
        for ep in range(temp_gen.epochs):
            perm = np.random.permutation(len(Z))
            for start in range(0, len(Z), temp_gen.batch_size):
                idx = perm[start:start + temp_gen.batch_size]
                x0 = torch.from_numpy(Z[idx]).float().to(temp_gen.device)
                t = torch.randint(0, temp_gen.n_steps, (x0.shape[0],), device=temp_gen.device).long()
                noise = torch.randn_like(x0, device=temp_gen.device)
                alpha = torch.from_numpy(temp_gen._alphas_cumprod[t.cpu().numpy()]).float().to(temp_gen.device).unsqueeze(1)
                x_noisy = torch.sqrt(alpha) * x0 + torch.sqrt(1 - alpha) * noise
                pred_noise = temp_gen._model(x_noisy, t)
                loss = nn.functional.mse_loss(pred_noise, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
        
        temp_gen._model.eval()
        torch.manual_seed(self.params.get("seed", self.seed))
        z_clamp = 10.0
        with torch.no_grad():
            z = torch.randn(n_samples, temp_gen._dim, device=temp_gen.device)
            for t in reversed(range(temp_gen.n_steps)):
                t_b = torch.full((n_samples,), t, dtype=torch.long, device=temp_gen.device)
                pred_noise = temp_gen._model(z, t_b)
                beta_t = float(temp_gen._betas[t])
                alpha_t = float(temp_gen._alphas_cumprod[t])
                alpha_t_next = float(temp_gen._alphas_cumprod[t + 1])
                denom = np.sqrt(1 - alpha_t)
                if denom < 1e-8:
                    denom = 1e-8
                z = (z - (beta_t / denom) * pred_noise) / (np.sqrt(alpha_t) + 1e-8)
                z = torch.clamp(z, -z_clamp, z_clamp)
                if t > 0:
                    sigma_t = ((1 - alpha_t_next) / (1 - alpha_t + 1e-8) * beta_t) ** 0.5
                    z = z + sigma_t * torch.randn_like(z, device=temp_gen.device)
                    z = torch.clamp(z, -z_clamp, z_clamp)
            Z_syn = z.cpu().numpy()
            if not np.isfinite(Z_syn).all():
                Z_syn = np.nan_to_num(Z_syn, nan=0.0, posinf=z_clamp, neginf=-z_clamp)
        
        X_syn, y_syn = temp_gen._decode(Z_syn)
        if y_constant is not None:
            y_syn = pd.Series([y_constant] * len(X_syn), name=LABEL_COL)

        return X_syn.reset_index(drop=True), y_syn.reset_index(drop=True)
