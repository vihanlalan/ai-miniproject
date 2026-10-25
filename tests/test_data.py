"""Tests for data loading, cleaning, splitting and imbalance simulation."""

from __future__ import annotations

import pandas as pd
import pytest

from synthdx.config import deep_merge, load_config, resolve_path
from synthdx.data import (
    CATEGORICAL_COLUMNS,
    TARGET,
    class_balance,
    make_imbalanced,
    prepare_splits,
    split_from_config,
    stratified_split,
)
from synthdx.data.loader import clean_heart_data, fetch_raw_file
from synthdx.data.schema import ALL_COLUMNS, CATEGORY_LEVELS, RAW_COLUMNS


def test_loaded_shape_and_schema(heart_df):
    assert heart_df.shape == (303, 14)
    assert list(heart_df.columns) == ALL_COLUMNS
    assert heart_df.isna().sum().sum() == 0
    assert set(heart_df[TARGET].unique()) == {0, 1}
    for col in CATEGORICAL_COLUMNS:
        assert set(heart_df[col].unique()) <= set(CATEGORY_LEVELS[col])


def test_clean_handles_missing_markers():
    raw = pd.DataFrame(
        [[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, "?", 6, 0],
         [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, "?", 2],
         [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, 3, 7, 1]],
        columns=RAW_COLUMNS,
    )
    clean = clean_heart_data(raw)
    assert clean.isna().sum().sum() == 0
    assert clean["ca"].tolist() == [3, 3, 3]          # imputed with mode
    assert clean[TARGET].tolist() == [0, 1, 1]       # num > 0 -> disease


def test_cache_is_reused(tmp_path):
    p1 = fetch_raw_file(tmp_path, url="http://127.0.0.1:9/x", timeout=0.5)
    mtime = p1.stat().st_mtime
    p2 = fetch_raw_file(tmp_path, url="http://127.0.0.1:9/x", timeout=0.5)
    assert p1 == p2 and p2.stat().st_mtime == mtime


def test_stratified_split_no_leakage(heart_df):
    train, test = stratified_split(heart_df, 0.3, seed=1)
    assert set(train.index).isdisjoint(test.index)
    assert len(train) + len(test) == len(heart_df)
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.03


def test_split_rejects_bad_size(heart_df):
    with pytest.raises(ValueError):
        stratified_split(heart_df, 1.5, seed=0)


@pytest.mark.parametrize("fraction", [0.10, 0.12, 0.15, 0.25])
def test_imbalance_ratio(heart_df, fraction):
    out = make_imbalanced(heart_df, fraction, seed=0)
    assert out[TARGET].mean() == pytest.approx(fraction, abs=0.01)
    # every negative is kept, positives are a subset
    assert (out[TARGET] == 0).sum() == (heart_df[TARGET] == 0).sum()
    assert set(out.index) <= set(heart_df.index)


def test_imbalance_noop_and_validation(heart_df):
    out = make_imbalanced(heart_df, 0.9, seed=0)
    assert len(out) == len(heart_df)
    with pytest.raises(ValueError):
        make_imbalanced(heart_df, 0.0, seed=0)


def test_prepare_splits_train_is_subset_of_pool(split):
    assert set(split.train.index) <= set(split.train_pool.index)
    assert set(split.train_pool.index).isdisjoint(split.test.index)
    assert class_balance(split.train)["pos_fraction"] == pytest.approx(0.12, abs=0.01)
    # natural prevalence preserved in the test set
    assert 0.4 < split.test[TARGET].mean() < 0.52


def test_prepare_splits_reproducible(heart_df):
    a = prepare_splits(heart_df, 0.3, 0.12, seed=7)
    b = prepare_splits(heart_df, 0.3, 0.12, seed=7)
    c = prepare_splits(heart_df, 0.3, 0.12, seed=8)
    assert list(a.train.index) == list(b.train.index)
    assert list(a.test.index) == list(b.test.index)
    assert list(a.test.index) != list(c.test.index)


def test_imbalanced_test_option(heart_df, cfg):
    cfg2 = deep_merge(cfg, {"data": {"imbalance_test": True}})
    s = split_from_config(heart_df, cfg2, seed=0)
    assert s.test[TARGET].mean() == pytest.approx(0.12, abs=0.03)


def test_config_helpers():
    cfg = load_config(overrides={"experiments": {"seeds": [1]}})
    assert cfg["experiments"]["seeds"] == [1]
    assert cfg["data"]["test_size"] == 0.30          # untouched siblings kept
    assert resolve_path("config.yaml").exists()
    assert resolve_path(resolve_path("x")).is_absolute()
