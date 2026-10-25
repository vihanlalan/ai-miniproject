"""Group fairness metrics.

* Demographic parity difference (DPD): max - min over groups of P(y_hat = 1 | g).
* Equal opportunity difference (EOD): max - min over groups of the true-positive
  rate P(y_hat = 1 | y = 1, g). Groups without positives are skipped.
Both are 0 for a perfectly "fair" classifier under that definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_sensitive_columns(df: pd.DataFrame, age_cutoff: int = 55) -> pd.DataFrame:
    """Add human-readable ``sex_group`` and ``age_group`` columns."""
    out = df.copy()
    out["sex_group"] = np.where(out["sex"] == 1, "male", "female")
    out["age_group"] = np.where(out["age"] >= age_cutoff, f">={age_cutoff}", f"<{age_cutoff}")
    return out


SENSITIVE_COLUMN = {"sex": "sex_group", "age_group": "age_group"}


def group_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Per-group size, accuracy, recall (TPR) and selection rate."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    rows = []
    for g in sorted(pd.unique(groups)):
        m = groups == g
        pos = m & (y_true == 1)
        rows.append({
            "group": g, "n": int(m.sum()), "n_pos": int(pos.sum()),
            "accuracy": float((y_true[m] == y_pred[m]).mean()),
            "recall": float(y_pred[pos].mean()) if pos.any() else np.nan,
            "selection_rate": float(y_pred[m].mean()),
        })
    return pd.DataFrame(rows)


def demographic_parity_difference(y_pred: np.ndarray, groups: np.ndarray) -> float:
    rates = group_metrics(np.zeros(len(y_pred)), y_pred, groups)["selection_rate"]
    return float(rates.max() - rates.min())


def equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    tprs = group_metrics(y_true, y_pred, groups)["recall"].dropna()
    return float(tprs.max() - tprs.min()) if len(tprs) > 1 else float("nan")
