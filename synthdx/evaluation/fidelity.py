"""Statistical fidelity of synthetic data versus real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from synthdx.data.schema import (
    ALL_COLUMNS,
    CATEGORICAL_COLUMNS,
    CATEGORY_LEVELS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET,
)
from synthdx.ml.models import build_preprocessor


def column_fidelity(real: pd.DataFrame, synth: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Per-column KS test (numerics) and Jensen-Shannon distance (all columns).

    JS distance uses base 2, so it lies in [0, 1] (0 = identical distributions).
    Numeric columns are binned on the real data's quantiles.
    """
    rows = []
    for col in NUMERIC_COLUMNS:
        ks = ks_2samp(real[col], synth[col])
        edges = np.unique(np.quantile(real[col], np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(real[col], edges)[0] + 1e-9
        q = np.histogram(synth[col], edges)[0] + 1e-9
        rows.append({"column": col, "type": "numeric", "ks_stat": ks.statistic,
                     "ks_pvalue": ks.pvalue, "js_distance": jensenshannon(p, q, base=2)})
    for col in CATEGORICAL_COLUMNS + [TARGET]:
        levels = CATEGORY_LEVELS.get(col, [0, 1])
        p = real[col].value_counts().reindex(levels, fill_value=0).to_numpy() + 1e-9
        q = synth[col].value_counts().reindex(levels, fill_value=0).to_numpy() + 1e-9
        rows.append({"column": col, "type": "categorical", "ks_stat": np.nan,
                     "ks_pvalue": np.nan, "js_distance": jensenshannon(p, q, base=2)})
    return pd.DataFrame(rows)


def correlation_matrices(real: pd.DataFrame, synth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson correlation matrices over all columns (NaN -> 0 for constant columns)."""
    return (real[ALL_COLUMNS].corr().fillna(0), synth[ALL_COLUMNS].corr().fillna(0))


def correlation_difference(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:
    """Mean / max absolute difference between the two correlation matrices."""
    cr, cs = correlation_matrices(real, synth)
    diff = (cr - cs).abs().to_numpy()
    off = diff[~np.eye(len(diff), dtype=bool)]
    return {"corr_mean_abs_diff": float(off.mean()), "corr_max_abs_diff": float(off.max())}


def detectability(real: pd.DataFrame, synth: pd.DataFrame, seed: int = 0, folds: int = 5) -> float:
    """ROC-AUC of a random forest trained to tell real from synthetic rows.

    0.5 = indistinguishable (ideal); 1.0 = trivially detectable. Classes are
    balanced by subsampling the larger set.
    """
    n = min(len(real), len(synth))
    r = real.sample(n, random_state=seed)[ALL_COLUMNS].assign(_is_synth=0)
    s = synth.sample(n, random_state=seed)[ALL_COLUMNS].assign(_is_synth=1)
    data = pd.concat([r, s], ignore_index=True)
    model = Pipeline([("prep", build_preprocessor()),
                      ("clf", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1))])
    k = max(2, min(folds, n))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    auc = cross_val_score(model, data[FEATURE_COLUMNS + [TARGET]], data["_is_synth"],
                          cv=cv, scoring="roc_auc")
    return float(np.mean(auc))


def fidelity_summary(real: pd.DataFrame, synth: pd.DataFrame, seed: int = 0) -> dict[str, float]:
    """One-row summary used by the experiment runner."""
    cols = column_fidelity(real, synth)
    return {
        "mean_ks_stat": float(cols["ks_stat"].mean()),
        "mean_js_distance": float(cols["js_distance"].mean()),
        **correlation_difference(real, synth),
        "detectability_auc": detectability(real, synth, seed),
    }
