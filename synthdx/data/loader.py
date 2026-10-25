"""Download, cache and clean the UCI Heart Disease (Cleveland) dataset."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from synthdx.config import resolve_path
from synthdx.data.schema import (
    CATEGORICAL_COLUMNS,
    INTEGER_NUMERIC_COLUMNS,
    NUMERIC_BOUNDS,
    RAW_COLUMNS,
    TARGET,
    ALL_COLUMNS,
)
from synthdx.utils import get_logger

logger = get_logger(__name__)

RAW_FILENAME = "processed.cleveland.data"
BUNDLED_COPY: Path = Path(__file__).resolve().parent / "resources" / RAW_FILENAME
DEFAULT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def fetch_raw_file(raw_dir: str | Path, url: str = DEFAULT_URL, timeout: float = 10) -> Path:
    """Return a local path to the raw data file.

    Order of preference: existing cache in ``raw_dir`` -> download from UCI ->
    bundled copy shipped with the package (offline fallback).
    """
    raw_dir = resolve_path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / RAW_FILENAME
    if target.exists() and target.stat().st_size > 0:
        logger.debug("Using cached dataset at %s", target)
        return target
    try:
        logger.info("Downloading dataset from %s", url)
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            payload = resp.read()
        if not payload:
            raise ValueError("empty download")
        target.write_bytes(payload)
        return target
    except Exception as exc:  # network errors, timeouts, bad URLs
        logger.warning("Download failed (%s); using bundled offline copy.", exc)
        shutil.copyfile(BUNDLED_COPY, target)
        return target


def clean_heart_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Cleveland frame.

    * ``?`` markers become NaN; the few missing ``ca``/``thal`` values are
      imputed with the column mode.
    * ``num`` (0-4) is binarised into ``target`` (disease present if > 0).
    * Columns are cast to integer types (``oldpeak`` stays float).
    """
    df = raw.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")
    missing = int(df.isna().sum().sum())
    if missing:
        logger.info("Imputing %d missing values with column modes", missing)
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    for col in INTEGER_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS:
        df[col] = df[col].round().astype(int)
    df["oldpeak"] = df["oldpeak"].astype(float).round(1)
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        df[col] = df[col].clip(lo, hi)
    return df[ALL_COLUMNS].reset_index(drop=True)


def load_heart_disease(
    raw_dir: str | Path = "data/raw", url: str = DEFAULT_URL, timeout: float = 10
) -> pd.DataFrame:
    """Load the cleaned dataset (303 rows, 13 features + binary target)."""
    path = fetch_raw_file(raw_dir, url, timeout)
    raw = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    df = clean_heart_data(raw)
    logger.info("Loaded %d records (%.1f%% positive)", len(df), 100 * df[TARGET].mean())
    return df


def load_from_config(cfg: dict) -> pd.DataFrame:
    """Convenience wrapper reading paths from the project config."""
    return load_heart_disease(
        cfg["paths"]["raw_dir"], cfg["data"]["url"], cfg["data"].get("download_timeout", 10)
    )
