"""Baseline classifiers: Logistic Regression, Random Forest, Gradient Boosting.

All models share one preprocessing pipeline (standardised numerics, one-hot
categoricals with the full, fixed level set) and are deliberately left
without class weighting, so the effect of data augmentation is not masked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from synthdx.data.schema import (
    CATEGORICAL_COLUMNS,
    CATEGORY_LEVELS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET,
)

CLASSIFIER_NAMES: list[str] = ["logistic_regression", "random_forest", "gradient_boosting"]
DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "expert_fixed": "Expert system (hand-written, fixed threshold)",
    "expert_calibrated": "Expert system (threshold calibrated)",
    "expert_tuned": "Expert system (rule CFs tuned)",
}


def build_preprocessor() -> ColumnTransformer:
    """Scaler for numerics + one-hot encoder for categoricals."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(categories=[CATEGORY_LEVELS[c] for c in CATEGORICAL_COLUMNS],
                              handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ])


def make_classifier(name: str, seed: int = 0) -> Pipeline:
    """Create an unfitted preprocessing + classifier pipeline."""
    if name == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
    elif name == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed,
                                     n_jobs=1)
    elif name == "gradient_boosting":
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                         random_state=seed)
    else:
        raise ValueError(f"unknown classifier '{name}'; choose from {CLASSIFIER_NAMES}")
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def train_classifier(name: str, train_df: pd.DataFrame, seed: int = 0) -> Pipeline:
    """Fit a classifier on a frame containing features and ``target``."""
    if train_df[TARGET].nunique() < 2:
        raise ValueError("training data must contain both classes")
    model = make_classifier(name, seed)
    model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET].astype(int))
    return model


def predict_scores(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Positive-class probability for each row."""
    return model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
