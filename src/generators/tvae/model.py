import warnings
from typing import Any, Dict, Type

import pandas as pd

from src.generators.sdv_base import SdvSingleTableGenerator, sdv_cuda_flag_from_params

warnings.filterwarnings("ignore")
from sdv.single_table import TVAESynthesizer


class TVAEGenerator(SdvSingleTableGenerator):
    """TVAE (VAE) generator."""

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = None,
        epochs: int = 300,
        batch_size: int = 500,
        embedding_dim: int = 128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale: float = 1e-5,
        loss_factor: int = 2,
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            n_samples=n_samples,
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            l2scale=l2scale,
            loss_factor=loss_factor,
            **kwargs,
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.compress_dims = tuple(compress_dims)
        self.decompress_dims = tuple(decompress_dims)
        self.l2scale = l2scale
        self.loss_factor = loss_factor

    @property
    def _sdv_synthesizer_cls(self) -> Type:
        return TVAESynthesizer

    def _sdv_synthesizer_kwargs(self) -> Dict[str, Any]:
        # SDV TVAESynthesizer ``cuda`` follows generator.params.device or auto if unset.
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "embedding_dim": self.embedding_dim,
            "compress_dims": self.compress_dims,
            "decompress_dims": self.decompress_dims,
            "l2scale": self.l2scale,
            "loss_factor": self.loss_factor,
            "cuda": sdv_cuda_flag_from_params(self.params),
            "verbose": False,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TVAEGenerator":
        super().fit(X, y)
        return self
