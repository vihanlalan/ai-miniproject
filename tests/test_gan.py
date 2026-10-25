"""GAN tests: tiny training runs, schema, conditional generation, reproducibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthdx.data.schema import (
    ALL_COLUMNS,
    CATEGORICAL_COLUMNS,
    CATEGORY_LEVELS,
    NUMERIC_BOUNDS,
    TARGET,
)
from synthdx.gan import (
    ConditionalGAN,
    CTGANGenerator,
    SyntheticGenerator,
    build_generator,
    enforce_schema,
)
from synthdx.gan.cgan import TabularEncoder


def _check_schema(df: pd.DataFrame) -> None:
    assert list(df.columns) == ALL_COLUMNS
    assert df.isna().sum().sum() == 0
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        assert df[col].between(lo, hi).all(), col
    for col in CATEGORICAL_COLUMNS:
        assert set(df[col].unique()) <= set(CATEGORY_LEVELS[col]), col
        assert pd.api.types.is_integer_dtype(df[col])
    assert pd.api.types.is_integer_dtype(df["age"])
    assert pd.api.types.is_float_dtype(df["oldpeak"])
    assert set(df[TARGET].unique()) <= {0, 1}


@pytest.fixture(scope="module")
def tiny_cgan(split):
    return ConditionalGAN(epochs=3, batch_size=16, seed=0).fit(split.train)


@pytest.fixture(scope="module")
def tiny_ctgan(split):
    return CTGANGenerator(epochs=2, batch_size=20, pac=10, seed=0).fit(split.train)


class TestSchema:
    def test_enforce_schema_snaps_and_clips(self):
        raw = pd.DataFrame([{**{c: 0 for c in ALL_COLUMNS}, "age": 150.7, "cp": 2.6,
                             "thal": 5.0, "oldpeak": -1.234, "target": 0.7, "chol": None}])
        out = enforce_schema(raw)
        _check_schema(out)
        r = out.iloc[0]
        assert r["age"] == 100 and r["cp"] == 3 and r["thal"] == 6
        assert r["oldpeak"] == 0.0 and r["target"] == 1

    def test_encoder_roundtrip(self, split):
        enc = TabularEncoder().fit(split.train)
        x = enc.transform(split.train)
        assert x.shape == (len(split.train), enc.dim)
        assert x.min() >= -1 - 1e-6 and x.max() <= 1 + 1e-6
        back = enc.inverse_transform(x)
        pd.testing.assert_frame_equal(
            back[CATEGORICAL_COLUMNS].reset_index(drop=True).astype(int),
            split.train[CATEGORICAL_COLUMNS].reset_index(drop=True).astype(int))
        np.testing.assert_allclose(back["chol"], split.train["chol"], atol=1e-3)


class TestConditionalGAN:
    def test_trains_and_records_losses(self, tiny_cgan):
        assert tiny_cgan.fitted
        h = tiny_cgan.loss_history
        assert list(h["epoch"]) == [1, 2, 3]
        assert np.isfinite(h[["generator_loss", "discriminator_loss"]].to_numpy()).all()

    def test_schema(self, tiny_cgan):
        _check_schema(tiny_cgan.generate(200))

    @pytest.mark.parametrize("cls", [0, 1])
    def test_conditional(self, tiny_cgan, cls):
        out = tiny_cgan.generate(50, condition=cls)
        assert len(out) == 50 and (out[TARGET] == cls).all()
        _check_schema(out)

    def test_reproducible(self, split):
        a = ConditionalGAN(epochs=2, batch_size=16, seed=3).fit(split.train).generate(30, 1)
        b = ConditionalGAN(epochs=2, batch_size=16, seed=3).fit(split.train).generate(30, 1)
        pd.testing.assert_frame_equal(a, b)

    def test_save_load_and_checkpoint(self, split, tmp_path):
        gan = ConditionalGAN(epochs=2, batch_size=16, seed=1, checkpoint_dir=tmp_path,
                             checkpoint_every=1).fit(split.train)
        assert (tmp_path / "cgan_epoch0002.pt").exists()
        path = gan.save(tmp_path / "cgan.pkl")
        loaded = SyntheticGenerator.load(path)
        pd.testing.assert_frame_equal(gan.generate(10, 1), loaded.generate(10, 1))

    def test_errors(self, tiny_cgan):
        with pytest.raises(RuntimeError):
            ConditionalGAN().generate(5)
        with pytest.raises(ValueError):
            tiny_cgan.generate(5, condition=2)
        assert tiny_cgan.generate(0).empty


class TestCTGAN:
    def test_trains_and_schema(self, tiny_ctgan):
        assert tiny_ctgan.fitted
        assert len(tiny_ctgan.loss_history) == 2
        _check_schema(tiny_ctgan.generate(100))

    def test_conditional(self, tiny_ctgan):
        out = tiny_ctgan.generate(40, condition=1)
        assert len(out) == 40 and (out[TARGET] == 1).all()

    def test_reproducible(self, split):
        a = CTGANGenerator(epochs=1, batch_size=20, seed=5).fit(split.train).generate(20)
        b = CTGANGenerator(epochs=1, batch_size=20, seed=5).fit(split.train).generate(20)
        pd.testing.assert_frame_equal(a, b)

    def test_invalid_batch(self):
        with pytest.raises(ValueError):
            CTGANGenerator(batch_size=25, pac=10)


def test_build_generator(cfg):
    assert isinstance(build_generator("cgan", cfg, seed=0, epochs=1), ConditionalGAN)
    assert build_generator("ctgan", cfg, seed=0, epochs=1).params["epochs"] == 1
    with pytest.raises(ValueError):
        build_generator("vae", cfg, seed=0)
