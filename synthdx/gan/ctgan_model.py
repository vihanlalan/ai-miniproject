"""Wrapper around the ``ctgan`` library (Xu et al., NeurIPS 2019).

CTGAN handles mixed tabular data with mode-specific normalisation for
continuous columns and a conditional vector + training-by-sampling for
discrete columns, which makes class-conditional generation of a rare class
straightforward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from synthdx.data.schema import ALL_COLUMNS, CATEGORICAL_COLUMNS, TARGET
from synthdx.gan.base import SyntheticGenerator
from synthdx.utils import get_logger, set_seed

logger = get_logger(__name__)


class CTGANGenerator(SyntheticGenerator):
    """CTGAN with the SynthDx ``generate(n, condition)`` API."""

    name = "ctgan"

    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 50,
        pac: int = 10,
        generator_dim: tuple[int, ...] = (128, 128),
        discriminator_dim: tuple[int, ...] = (128, 128),
        embedding_dim: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if batch_size % pac or batch_size % 2:
            raise ValueError("batch_size must be even and a multiple of pac")
        self.params = dict(
            epochs=epochs, batch_size=batch_size, pac=pac, generator_dim=tuple(generator_dim),
            discriminator_dim=tuple(discriminator_dim), embedding_dim=embedding_dim,
        )
        self.seed = seed
        self.model = None

    @classmethod
    def from_config(cls, cfg: dict, seed: int, epochs: int | None = None) -> CTGANGenerator:
        c = dict(cfg["gan"]["ctgan"])
        if epochs is not None:
            c["epochs"] = epochs
        return cls(seed=seed, **c)

    def fit(self, df: pd.DataFrame) -> CTGANGenerator:
        from ctgan import CTGAN

        set_seed(self.seed)
        self.model = CTGAN(cuda=torch.cuda.is_available(), verbose=False, **self.params)
        self.model.set_random_state(self.seed)
        logger.info("Training CTGAN for %d epochs on %d rows", self.params["epochs"], len(df))
        self.model.fit(df[ALL_COLUMNS].reset_index(drop=True),
                       discrete_columns=CATEGORICAL_COLUMNS + [TARGET])
        self.class_prior = float(df[TARGET].mean())
        self.loss_history = self._extract_losses()
        self._rng = np.random.RandomState(self.seed)
        self.fitted = True
        return self

    def _extract_losses(self) -> pd.DataFrame:
        lv = getattr(self.model, "loss_values", None)
        if lv is None or len(lv) == 0:
            return pd.DataFrame(columns=["epoch", "generator_loss", "discriminator_loss"])
        lv = lv.rename(columns={"Epoch": "epoch", "Generator Loss": "generator_loss",
                                "Discriminator Loss": "discriminator_loss"})
        lv = lv[["epoch", "generator_loss", "discriminator_loss"]].astype(float)
        return lv.groupby("epoch", as_index=False).mean()

    def _sample(self, n: int, condition: int | None) -> pd.DataFrame:
        # A private RNG makes the sequence of generated batches reproducible per seed.
        self.model.set_random_state(int(self._rng.randint(0, 2**31 - 1)))
        if condition is None:
            return self.model.sample(n)
        return self.model.sample(n, condition_column=TARGET, condition_value=int(condition))
