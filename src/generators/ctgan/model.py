import warnings
from typing import Any, Dict, Type

import pandas as pd

from src.generators.sdv_base import (
    SdvSingleTableGenerator,
    sdv_cuda_flag_from_params,
)

warnings.filterwarnings("ignore")
from sdv.single_table import CTGANSynthesizer


class CTGANGenerator(SdvSingleTableGenerator):
    """CTGAN (GAN) generator."""

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = None,
        epochs: int = 300,
        batch_size: int = 500,
        embedding_dim: int = 128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        discriminator_steps: int = 1,
        pac: int = 10,
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            n_samples=n_samples,
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            discriminator_lr=discriminator_lr,
            discriminator_steps=discriminator_steps,
            pac=pac,
            **kwargs,
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.generator_dim = tuple(generator_dim)
        self.discriminator_dim = tuple(discriminator_dim)
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.discriminator_steps = discriminator_steps
        self.pac = pac

    @property
    def _sdv_synthesizer_cls(self) -> Type:
        return CTGANSynthesizer

    def _sdv_synthesizer_kwargs(self) -> Dict[str, Any]:
        # SDV CTGANSynthesizer ``cuda`` follows generator.params.device (Hydra ++generator.params.device=cuda)
        # or auto-enables when CUDA is available if device is unset.
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "embedding_dim": self.embedding_dim,
            "generator_dim": self.generator_dim,
            "discriminator_dim": self.discriminator_dim,
            "generator_lr": self.generator_lr,
            "discriminator_lr": self.discriminator_lr,
            "discriminator_steps": self.discriminator_steps,
            "pac": self.pac,
            "cuda": sdv_cuda_flag_from_params(self.params),
            "verbose": False,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CTGANGenerator":
        super().fit(X, y)
        return self
