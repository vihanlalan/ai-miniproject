"""Seed aggregation and paired significance tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def paired_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired comparison of ``b`` against baseline ``a`` (same seeds, same order).

    Returns the mean difference (b - a), a paired t-test p-value and a
    Wilcoxon signed-rank p-value (NaN when undefined, e.g. all differences 0).
    With only 5 seeds the Wilcoxon test cannot reach p < 0.05 (min p = 0.0625),
    so the t-test is the primary test and Wilcoxon a robustness check.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or len(a) < 2:
        raise ValueError("paired_test needs two equal-length arrays with >= 2 values")
    d = b - a
    out = {"mean_diff": float(d.mean()), "t_pvalue": float("nan"), "wilcoxon_pvalue": float("nan")}
    if np.allclose(d, d[0]):
        # zero variance: t-test undefined
        out["t_pvalue"] = 1.0 if np.isclose(d[0], 0) else 0.0
    else:
        out["t_pvalue"] = float(stats.ttest_rel(b, a).pvalue)
    if np.any(d != 0):
        out["wilcoxon_pvalue"] = float(stats.wilcoxon(b, a, zero_method="zsplit").pvalue)
    return out


def mean_std(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    """Aggregate seeds into mean and std columns (``<metric>_mean``, ``<metric>_std``)."""
    agg = df.groupby(group_cols)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()
