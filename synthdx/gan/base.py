"""Common interface and schema enforcement for synthetic-data generators."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from synthdx.data.schema import (
    ALL_COLUMNS,
    CATEGORICAL_COLUMNS,
    CATEGORY_LEVELS,
    INTEGER_NUMERIC_COLUMNS,
    NUMERIC_BOUNDS,
    TARGET,
)
from synthdx.utils import get_logger

logger = get_logger(__name__)


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce generated rows to the real schema.

    Numerics are clipped to plausible bounds and rounded like the source data;
    categorical values are snapped to the nearest valid level; the target is 0/1.
    """
    out = df[ALL_COLUMNS].copy()
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna((lo + hi) / 2).clip(lo, hi)
    for col in INTEGER_NUMERIC_COLUMNS:
        out[col] = out[col].round().astype(int)
    out["oldpeak"] = out["oldpeak"].astype(float).round(1)
    for col in CATEGORICAL_COLUMNS:
        levels = np.array(CATEGORY_LEVELS[col])
        vals = pd.to_numeric(out[col], errors="coerce").fillna(levels[0]).to_numpy(float)
        out[col] = levels[np.abs(vals[:, None] - levels[None, :]).argmin(axis=1)].astype(int)
    out[TARGET] = (pd.to_numeric(out[TARGET], errors="coerce").fillna(0) >= 0.5).astype(int)
    return out.reset_index(drop=True)


class SyntheticGenerator(ABC):
    """Interface implemented by :class:`CTGANGenerator` and :class:`ConditionalGAN`."""

    name: str = "base"

    def __init__(self) -> None:
        self.loss_history: pd.DataFrame = pd.DataFrame(
            columns=["epoch", "generator_loss", "discriminator_loss"])
        self.class_prior: float = 0.5
        self.fitted: bool = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> SyntheticGenerator:
        """Train on a real frame with features and target."""

    @abstractmethod
    def _sample(self, n: int, condition: int | None) -> pd.DataFrame:
        """Draw ``n`` raw rows (optionally conditioned on the class)."""

    def generate(self, n: int, condition: int | None = None, max_rounds: int = 20) -> pd.DataFrame:
        """Generate ``n`` schema-valid rows; if ``condition`` is given, every row
        has ``target == condition`` (rejection sampling on top of conditioning)."""
        if not self.fitted:
            raise RuntimeError(f"{self.name} generator has not been fitted")
        if n <= 0:
            return pd.DataFrame(columns=ALL_COLUMNS)
        if condition is None:
            return enforce_schema(self._sample(n, None))
        if condition not in (0, 1):
            raise ValueError("condition must be 0, 1 or None")
        kept: list[pd.DataFrame] = []
        have = 0
        for _ in range(max_rounds):
            batch = enforce_schema(self._sample(max(n - have, 16) * 2, condition))
            batch = batch[batch[TARGET] == condition]
            kept.append(batch)
            have += len(batch)
            if have >= n:
                break
        out = pd.concat(kept, ignore_index=True).head(n)
        if len(out) < n:
            raise RuntimeError(f"{self.name}: could only generate {len(out)}/{n} rows "
                               f"of class {condition}")
        return out.reset_index(drop=True)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Saved %s generator to %s", self.name, path)
        return path

    @staticmethod
    def load(path: str | Path) -> SyntheticGenerator:
        with open(path, "rb") as fh:
            return pickle.load(fh)  # noqa: S301 - only loads our own checkpoints
