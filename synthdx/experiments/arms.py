"""Construction of the training-data arms.

    A_real    real imbalanced training data only
    B_<gen>   real + GAN-generated minority rows until classes are balanced
    C_smote   real + SMOTE-NC minority rows until classes are balanced
    D_<gen>   synthetic only (TSTR): a balanced synthetic set the size of arm B

The held-out real test set is never touched by any of these functions.
"""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTENC

from synthdx.data.schema import ALL_COLUMNS, CATEGORICAL_COLUMNS, FEATURE_COLUMNS, TARGET
from synthdx.gan.base import SyntheticGenerator, enforce_schema

ARM_DESCRIPTIONS: dict[str, str] = {
    "A_real": "A: Real imbalanced data only",
    "B_ctgan": "B: Real + CTGAN synthetic minority",
    "B_cgan": "B: Real + custom cGAN synthetic minority",
    "C_smote": "C: Real + SMOTE-NC",
    "D_ctgan": "D: Synthetic only, CTGAN (TSTR)",
    "D_cgan": "D: Synthetic only, custom cGAN (TSTR)",
}


def arm_label(arm: str) -> str:
    return ARM_DESCRIPTIONS.get(arm, arm)


def minority_deficit(train: pd.DataFrame) -> int:
    """How many positive rows are needed to balance the classes."""
    n_pos = int(train[TARGET].sum())
    return max(len(train) - 2 * n_pos, 0)


def gan_augment(train: pd.DataFrame, gen: SyntheticGenerator) -> pd.DataFrame:
    """Arm B: oversample the minority class with GAN samples."""
    synth = gen.generate(minority_deficit(train), condition=1).assign(_synthetic=1)
    return pd.concat([train[ALL_COLUMNS].assign(_synthetic=0), synth], ignore_index=True)


def smote_augment(train: pd.DataFrame, k_neighbors: int = 3, seed: int = 0) -> pd.DataFrame:
    """Arm C: SMOTE-NC (handles categorical columns) up to a 1:1 ratio."""
    n_pos = int(train[TARGET].sum())
    k = max(1, min(k_neighbors, n_pos - 1))
    cat_idx = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_COLUMNS]
    sm = SMOTENC(categorical_features=cat_idx, k_neighbors=k, random_state=seed)
    x_res, y_res = sm.fit_resample(train[FEATURE_COLUMNS], train[TARGET])
    out = x_res.copy()
    out[TARGET] = y_res.to_numpy()
    out = enforce_schema(out)
    out["_synthetic"] = [0] * len(train) + [1] * (len(out) - len(train))
    return out


def synthetic_only(train: pd.DataFrame, gen: SyntheticGenerator) -> pd.DataFrame:
    """Arm D: balanced, fully synthetic training set (Train-Synthetic-Test-Real)."""
    n_per_class = int((train[TARGET] == 0).sum())
    synth = pd.concat([gen.generate(n_per_class, 0), gen.generate(n_per_class, 1)],
                      ignore_index=True)
    return synth.assign(_synthetic=1)


def build_arms(
    train: pd.DataFrame,
    generators: dict[str, SyntheticGenerator],
    smote_k: int = 3,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Return every training arm keyed by arm id."""
    arms = {"A_real": train[ALL_COLUMNS].assign(_synthetic=0).reset_index(drop=True)}
    for name, gen in generators.items():
        arms[f"B_{name}"] = gan_augment(train, gen)
    arms["C_smote"] = smote_augment(train, smote_k, seed)
    for name, gen in generators.items():
        arms[f"D_{name}"] = synthetic_only(train, gen)
    return arms
