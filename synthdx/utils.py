"""Shared utilities: logging and reproducibility."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger; configures a root handler once."""
    root = logging.getLogger("synthdx")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(os.environ.get("SYNTHDX_LOG_LEVEL", "INFO"))
    return logging.getLogger(name if name.startswith("synthdx") else f"synthdx.{name}")


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass
