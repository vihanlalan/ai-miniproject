"""Data-driven calibration of the expert system.

Two steps, both fitted on a TRAINING arm only (never on the test set):

1. ``calibrate_threshold`` picks the CF decision threshold that maximises a
   metric (default: accuracy, mirroring the default objective of the ML
   baselines) on the training data.
2. ``tune_rule_cfs`` (stretch goal) re-estimates the CF of every rule that
   concludes the goal using Heckerman's probabilistic reading of a CF:

       p = P(disease | rule fired),  prior = P(disease)
       CF_emp = (p - prior) / (1 - prior)   if p >= prior
              = (p - prior) / prior          otherwise

   and shrinks it toward the hand-written CF with a pseudo-count ``k``:
       CF_new = (n * CF_emp + k * CF_hand) / (n + k),   n = times the rule fired.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from synthdx.data.schema import TARGET
from synthdx.expert.engine import InferenceEngine

METRICS: dict[str, Callable] = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "f1": lambda y, p: f1_score(y, p, zero_division=0),
}


def calibrate_threshold(scores: np.ndarray, y: np.ndarray, metric: str = "accuracy") -> float:
    """Return the CF threshold maximising ``metric`` (ties -> threshold nearest 0.2)."""
    fn = METRICS[metric]
    candidates = np.unique(np.concatenate([np.round(scores, 4), [0.2]]))
    best_t, best_v = 0.2, -np.inf
    for t in candidates:
        v = fn(y, (scores >= t).astype(int))
        if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and abs(t - 0.2) < abs(best_t - 0.2)):
            best_t, best_v = float(t), v
    return best_t


def empirical_cf(p: float, prior: float) -> float:
    """Heckerman's CF from a conditional and a prior probability."""
    if prior <= 0 or prior >= 1:
        return 0.0
    return (p - prior) / (1 - prior) if p >= prior else (p - prior) / prior


def tune_rule_cfs(
    engine: InferenceEngine, df: pd.DataFrame, shrinkage: float = 10.0
) -> dict[str, float]:
    """Re-estimate CFs of rules concluding the goal from labelled data."""
    y = df[TARGET].to_numpy()
    prior = float(y.mean())
    fired_sets = [engine.score_record(r)[1] for r in df.to_dict("records")]
    new_cfs: dict[str, float] = {}
    for rule in engine.kb.rules:
        if rule.conclusion != engine.goal:
            continue
        mask = np.array([rule.id in s for s in fired_sets])
        n = int(mask.sum())
        if n == 0:
            new_cfs[rule.id] = rule.cf
            continue
        emp = empirical_cf(float(y[mask].mean()), prior)
        cf = (n * emp + shrinkage * rule.cf) / (n + shrinkage)
        new_cfs[rule.id] = float(np.clip(cf, -0.95, 0.95))
    return new_cfs


def fit_expert(
    engine: InferenceEngine, train_df: pd.DataFrame, tune_rules: bool,
    metric: str = "accuracy", shrinkage: float = 10.0,
) -> InferenceEngine:
    """Return a calibrated copy of ``engine`` (optionally with tuned rule CFs)."""
    eng = engine
    if tune_rules:
        eng = engine.clone(kb=engine.kb.with_cfs(tune_rule_cfs(engine, train_df, shrinkage)))
    scores = eng.score(train_df)
    threshold = calibrate_threshold(scores, train_df[TARGET].to_numpy(), metric)
    return eng.clone(decision_threshold=threshold)
