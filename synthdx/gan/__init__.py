"""Synthetic tabular data generators (CTGAN wrapper + from-scratch conditional GAN)."""

from __future__ import annotations

from synthdx.gan.base import SyntheticGenerator, enforce_schema
from synthdx.gan.cgan import ConditionalGAN
from synthdx.gan.ctgan_model import CTGANGenerator

GENERATORS = {"ctgan": CTGANGenerator, "cgan": ConditionalGAN}


def build_generator(name: str, cfg: dict, seed: int, epochs: int | None = None) -> SyntheticGenerator:
    """Instantiate a generator by name from the config."""
    if name not in GENERATORS:
        raise ValueError(f"unknown generator '{name}'; choose from {sorted(GENERATORS)}")
    return GENERATORS[name].from_config(cfg, seed=seed, epochs=epochs)


__all__ = ["CTGANGenerator", "ConditionalGAN", "GENERATORS", "SyntheticGenerator",
           "build_generator", "enforce_schema"]
