"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from synthdx.config import load_config
from synthdx.data import load_heart_disease, prepare_splits


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def heart_df(tmp_path_factory) -> pd.DataFrame:
    # An unreachable URL + empty cache dir forces the offline bundled fallback,
    # so the test-suite never depends on the network.
    raw_dir = tmp_path_factory.mktemp("raw")
    return load_heart_disease(raw_dir, url="http://127.0.0.1:9/none", timeout=0.5)


@pytest.fixture(scope="session")
def split(heart_df):
    return prepare_splits(heart_df, test_size=0.3, positive_fraction=0.12, seed=0)
