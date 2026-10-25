"""Stratified splitting and imbalance simulation.

The DataFrame index is preserved as a record id throughout, so leakage between
train and test can be checked by index disjointness.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from synthdx.data.schema import TARGET
from synthdx.utils import get_logger

logger = get_logger(__name__)


@dataclass
class DataSplit:
    """Container for one seeded split.

    Attributes:
        train_pool: real training records before imbalance simulation.
        train: imbalanced real training set (what every arm starts from).
        test: held-out real test set (never augmented or resampled).
    """

    train_pool: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    seed: int


def stratified_split(
    df: pd.DataFrame, test_size: float, seed: int, target: str = TARGET
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split that keeps the original index."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[target], random_state=seed
    )
    return train, test


def make_imbalanced(
    df: pd.DataFrame, positive_fraction: float, seed: int, target: str = TARGET
) -> pd.DataFrame:
    """Downsample the positive class so it makes up ``positive_fraction`` of rows.

    All negatives are kept; ``n_pos = round(f / (1 - f) * n_neg)``. If the
    requested fraction is above the current one, the data is returned unchanged.
    """
    if not 0 < positive_fraction < 1:
        raise ValueError("positive_fraction must be in (0, 1)")
    pos = df[df[target] == 1]
    neg = df[df[target] == 0]
    n_pos = max(2, int(round(positive_fraction / (1 - positive_fraction) * len(neg))))
    if n_pos >= len(pos):
        logger.warning("Requested positive fraction not below current; no downsampling.")
        return df.copy()
    pos_kept = pos.sample(n=n_pos, random_state=seed)
    out = pd.concat([neg, pos_kept]).sample(frac=1.0, random_state=seed)
    return out


def class_balance(df: pd.DataFrame, target: str = TARGET) -> dict[str, float]:
    """Return counts and the positive share of a frame."""
    n_pos = int(df[target].sum())
    return {"n": len(df), "n_pos": n_pos, "n_neg": len(df) - n_pos,
            "pos_fraction": n_pos / max(len(df), 1)}


def prepare_splits(
    df: pd.DataFrame,
    test_size: float,
    positive_fraction: float,
    seed: int,
    imbalance_test: bool = False,
) -> DataSplit:
    """Hold out a real test set first, then imbalance the training pool."""
    train_pool, test = stratified_split(df, test_size, seed)
    train = make_imbalanced(train_pool, positive_fraction, seed)
    if imbalance_test:
        test = make_imbalanced(test, positive_fraction, seed)
    logger.debug("seed=%d train=%s test=%s", seed, class_balance(train), class_balance(test))
    return DataSplit(train_pool=train_pool, train=train, test=test, seed=seed)


def split_from_config(df: pd.DataFrame, cfg: dict, seed: int) -> DataSplit:
    """Build a :class:`DataSplit` from the ``data`` config section."""
    d = cfg["data"]
    return prepare_splits(
        df, d["test_size"], d["positive_fraction"], seed, d.get("imbalance_test", False)
    )
