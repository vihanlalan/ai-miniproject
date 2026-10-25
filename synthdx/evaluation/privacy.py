"""Privacy checks for synthetic data.

* DCR (distance to closest record): for each synthetic row, the distance to
  its nearest REAL TRAINING row in a standardised one-hot space. It is
  compared with the same statistic for real hold-out rows (which the
  generator never saw): synthetic rows systematically closer than hold-out
  rows suggest memorisation.
* Exact duplicates of real training rows.
* Distance-based membership-inference attack: the attacker scores each real
  record by how close it is to the synthetic data; AUC 0.5 means the attacker
  cannot tell training members from non-members.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from synthdx.data.schema import ALL_COLUMNS, FEATURE_COLUMNS, TARGET
from synthdx.ml.models import build_preprocessor


def _embed(reference: pd.DataFrame, *frames: pd.DataFrame) -> list[np.ndarray]:
    prep = build_preprocessor().fit(reference[FEATURE_COLUMNS])

    def emb(df: pd.DataFrame) -> np.ndarray:
        x = prep.transform(df[FEATURE_COLUMNS])
        x = x.toarray() if hasattr(x, "toarray") else x
        return np.hstack([x, df[[TARGET]].to_numpy(float)])

    return [emb(f) for f in frames]


def nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1).fit(reference)
    return nn.kneighbors(query)[0][:, 0]


def exact_duplicates(synth: pd.DataFrame, real: pd.DataFrame) -> int:
    """Number of synthetic rows identical (all columns) to some real row."""
    real_rows = set(map(tuple, real[ALL_COLUMNS].to_numpy(float).round(4)))
    return int(sum(tuple(r) in real_rows for r in synth[ALL_COLUMNS].to_numpy(float).round(4)))


def dcr_stats(synth: pd.DataFrame, train: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, float]:
    """DCR of synthetic and hold-out rows to the real training set."""
    tr, sy, ho = _embed(train, train, synth, holdout)
    d_syn = nearest_distances(sy, tr)
    d_hold = nearest_distances(ho, tr)
    return {
        "dcr_synth_median": float(np.median(d_syn)),
        "dcr_synth_p05": float(np.percentile(d_syn, 5)),
        "dcr_holdout_median": float(np.median(d_hold)),
        "dcr_holdout_p05": float(np.percentile(d_hold, 5)),
        "dcr_ratio": float(np.median(d_syn) / max(np.median(d_hold), 1e-9)),
        "share_synth_closer_than_holdout_p05": float(np.mean(d_syn < np.percentile(d_hold, 5))),
    }


def membership_inference_auc(
    synth: pd.DataFrame, members: pd.DataFrame, non_members: pd.DataFrame
) -> float:
    """AUC of a distance-to-synthetic attack (members should look closer).

    The attack is run separately within each class and the AUCs are averaged
    (weighted by class size), so a different class mix between members and
    non-members cannot masquerade as leakage.
    """
    sy, mem, non = _embed(members, synth, members, non_members)
    d_mem = nearest_distances(mem, sy)
    d_non = nearest_distances(non, sy)
    y_mem, y_non = members[TARGET].to_numpy(), non_members[TARGET].to_numpy()
    aucs, weights = [], []
    for cls in (0, 1):
        dm, dn = d_mem[y_mem == cls], d_non[y_non == cls]
        if len(dm) and len(dn):
            y = np.r_[np.ones(len(dm)), np.zeros(len(dn))]
            aucs.append(roc_auc_score(y, -np.r_[dm, dn]))
            weights.append(len(dm) + len(dn))
    return float(np.average(aucs, weights=weights)) if aucs else float("nan")


def privacy_summary(synth: pd.DataFrame, train: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, float]:
    """DCR statistics, duplicates and membership-inference AUC in one dict."""
    return {
        **dcr_stats(synth, train, holdout),
        "exact_duplicates": exact_duplicates(synth, train),
        "mia_auc": membership_inference_auc(synth, train, holdout),
    }
