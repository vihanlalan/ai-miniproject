"""Tests for ML baselines and arm construction (test set untouched)."""

from __future__ import annotations

import pandas as pd
import pytest

from synthdx.data.schema import TARGET
from synthdx.experiments.arms import (
    arm_label,
    build_arms,
    gan_augment,
    minority_deficit,
    smote_augment,
    synthetic_only,
)
from synthdx.gan import ConditionalGAN
from synthdx.ml import CLASSIFIER_NAMES, make_classifier, predict_scores, train_classifier


@pytest.mark.parametrize("name", CLASSIFIER_NAMES)
def test_classifiers_train_and_predict(split, name):
    model = train_classifier(name, split.train_pool, seed=0)
    s = predict_scores(model, split.test)
    assert s.shape == (len(split.test),) and ((s >= 0) & (s <= 1)).all()


def test_classifier_errors(split):
    with pytest.raises(ValueError):
        make_classifier("svm")
    with pytest.raises(ValueError):
        train_classifier("logistic_regression", split.train[split.train[TARGET] == 0])


@pytest.fixture(scope="module")
def gen(split):
    return ConditionalGAN(epochs=2, batch_size=16, seed=0).fit(split.train)


def test_gan_augment_balances(split, gen):
    out = gan_augment(split.train, gen)
    assert out[TARGET].mean() == pytest.approx(0.5, abs=0.01)
    assert out["_synthetic"].sum() == minority_deficit(split.train)
    assert (out.loc[out._synthetic == 1, TARGET] == 1).all()


def test_smote_augment(split):
    out = smote_augment(split.train, 3, seed=0)
    assert out[TARGET].mean() == pytest.approx(0.5, abs=0.01)
    assert out["_synthetic"].sum() == minority_deficit(split.train)


def test_synthetic_only(split, gen):
    out = synthetic_only(split.train, gen)
    assert (out["_synthetic"] == 1).all()
    assert out[TARGET].mean() == pytest.approx(0.5)


def test_build_arms_never_touches_test(split, gen):
    test_before = split.test.copy()
    arms = build_arms(split.train, {"cgan": gen}, 3, 0)
    assert set(arms) == {"A_real", "B_cgan", "C_smote", "D_cgan"}
    pd.testing.assert_frame_equal(split.test, test_before)
    # the real rows inside every augmented arm are exactly the training rows
    train_vals = split.train.reset_index(drop=True)
    for arm in ("A_real", "B_cgan", "C_smote"):
        real_part = arms[arm][arms[arm]["_synthetic"] == 0].drop(columns="_synthetic")
        pd.testing.assert_frame_equal(real_part.reset_index(drop=True), train_vals,
                                      check_dtype=False)
    assert arm_label("A_real").startswith("A:") and arm_label("X") == "X"
