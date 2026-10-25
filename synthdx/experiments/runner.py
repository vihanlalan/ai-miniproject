"""Reproducible experiment runner.

For every seed: split the data, train each GAN on the imbalanced real
training set, build arms A-D, then train/calibrate every model on every arm
and evaluate on the SAME untouched real test set. Results are written to
``results/`` as CSV/JSON.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthdx import __version__
from synthdx.config import resolve_path
from synthdx.data import load_from_config, split_from_config
from synthdx.data.schema import TARGET
from synthdx.evaluation.fairness import (
    SENSITIVE_COLUMN,
    add_sensitive_columns,
    demographic_parity_difference,
    equal_opportunity_difference,
    group_metrics,
)
from synthdx.evaluation.fidelity import column_fidelity, fidelity_summary
from synthdx.evaluation.metrics import METRIC_NAMES, classification_metrics
from synthdx.evaluation.privacy import privacy_summary
from synthdx.evaluation.stats import mean_std, paired_test
from synthdx.expert.engine import InferenceEngine
from synthdx.expert.tuning import fit_expert
from synthdx.experiments.arms import build_arms
from synthdx.gan import build_generator
from synthdx.ml.models import predict_scores, train_classifier
from synthdx.utils import get_logger, set_seed

logger = get_logger(__name__)

EXPERT_VARIANTS = ("expert_fixed", "expert_calibrated", "expert_tuned")


def _evaluate_models(
    arm_df: pd.DataFrame, test: pd.DataFrame, cfg: dict, seed: int, base_engine: InferenceEngine
) -> dict[str, dict[str, Any]]:
    """Train/calibrate every model on one arm; return scores, preds and extras."""
    out: dict[str, dict[str, Any]] = {}
    for name in cfg["experiments"]["classifiers"]:
        model = train_classifier(name, arm_df, seed)
        scores = predict_scores(model, test)
        out[name] = {"scores": scores, "pred": (scores >= 0.5).astype(int)}
    if cfg["experiments"].get("include_expert", True):
        e = cfg["expert"]
        metric = e.get("calibration_metric", "accuracy")
        shrink = e.get("tuning_shrinkage", 10)
        variants = {
            "expert_fixed": base_engine,
            "expert_calibrated": fit_expert(base_engine, arm_df, False, metric, shrink),
            "expert_tuned": fit_expert(base_engine, arm_df, True, metric, shrink),
        }
        for name, eng in variants.items():
            scores = eng.score(test)
            out[name] = {"scores": scores, "pred": (scores >= eng.decision_threshold).astype(int),
                         "threshold": eng.decision_threshold,
                         "rule_cfs": {r.id: r.cf for r in eng.kb.rules if r.conclusion == eng.goal}}
    return out


def run_seed(df: pd.DataFrame, cfg: dict, seed: int, models_dir: Path | None = None) -> dict[str, list]:
    """Run all arms for one seed and return lists of result rows."""
    set_seed(seed)
    exp = cfg["experiments"]
    split = split_from_config(df, cfg, seed)
    train, test = split.train, split.test
    rows: dict[str, list] = {k: [] for k in (
        "metrics", "predictions", "fairness_groups", "fairness_summary",
        "generator_quality", "column_fidelity", "losses", "expert_params", "arm_sizes")}

    generators = {}
    for gname in exp["generators"]:
        t0 = time.time()
        gen = build_generator(gname, cfg, seed).fit(train)
        generators[gname] = gen
        train_secs = time.time() - t0
        if models_dir is not None:
            gen.save(models_dir / f"{gname}_seed{seed}.pkl")
        for _, r in gen.loss_history.iterrows():
            rows["losses"].append({"seed": seed, "generator": gname, **r.to_dict()})
        # quality sample drawn at the training class prior, compared with real train
        sample = gen.generate(len(train))
        q = {"seed": seed, "generator": gname, "train_seconds": train_secs,
             **fidelity_summary(train, sample, seed), **privacy_summary(sample, train, test)}
        rows["generator_quality"].append(q)
        for _, r in column_fidelity(train, sample).iterrows():
            rows["column_fidelity"].append({"seed": seed, "generator": gname, **r.to_dict()})

    arms = build_arms(train, generators, exp.get("smote_k_neighbors", 3), seed)
    base_engine = InferenceEngine.from_config(cfg)
    test_s = add_sensitive_columns(test, exp.get("age_group_cutoff", 55))
    y = test[TARGET].to_numpy()

    for arm, arm_df in arms.items():
        rows["arm_sizes"].append({"seed": seed, "arm": arm, "n": len(arm_df),
                                  "n_pos": int(arm_df[TARGET].sum()),
                                  "n_synthetic": int(arm_df["_synthetic"].sum())})
        logger.info("seed %d | arm %-8s | n=%d (pos=%d, synthetic=%d)", seed, arm, len(arm_df),
                    arm_df[TARGET].sum(), arm_df["_synthetic"].sum())
        for model, res in _evaluate_models(arm_df, test, cfg, seed, base_engine).items():
            rows["metrics"].append({"seed": seed, "arm": arm, "model": model,
                                    **classification_metrics(y, res["pred"], res["scores"])})
            rows["predictions"] += [
                {"seed": seed, "arm": arm, "model": model, "record_id": int(i), "y_true": int(t),
                 "score": float(s), "pred": int(p)}
                for i, t, s, p in zip(test.index, y, res["scores"], res["pred"])]
            if "threshold" in res:
                rows["expert_params"] += [
                    {"seed": seed, "arm": arm, "model": model, "param": rid, "value": cf}
                    for rid, cf in res["rule_cfs"].items()]
                rows["expert_params"].append({"seed": seed, "arm": arm, "model": model,
                                              "param": "threshold", "value": res["threshold"]})
            for attr in exp.get("sensitive_attributes", []):
                groups = test_s[SENSITIVE_COLUMN.get(attr, attr)].to_numpy()
                gm = group_metrics(y, res["pred"], groups)
                for _, g in gm.iterrows():
                    rows["fairness_groups"].append({"seed": seed, "arm": arm, "model": model,
                                                    "attribute": attr, **g.to_dict()})
                rows["fairness_summary"].append({
                    "seed": seed, "arm": arm, "model": model, "attribute": attr,
                    "demographic_parity_diff": demographic_parity_difference(res["pred"], groups),
                    "equal_opportunity_diff": equal_opportunity_difference(y, res["pred"], groups),
                })
    return rows


def significance_table(metrics: pd.DataFrame, baseline: str = "A_real",
                       targets: tuple[str, ...] = ("recall", "f1", "balanced_accuracy", "roc_auc")
                       ) -> pd.DataFrame:
    """Paired tests of every arm against the baseline arm, per model and metric."""
    rows = []
    for (model, arm), grp in metrics.groupby(["model", "arm"]):
        if arm == baseline:
            continue
        base = metrics[(metrics.model == model) & (metrics.arm == baseline)]
        merged = base.merge(grp, on="seed", suffixes=("_a", "_b"))
        if len(merged) < 2:
            continue
        for m in targets:
            a, b = merged[f"{m}_a"].to_numpy(), merged[f"{m}_b"].to_numpy()
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            rows.append({"model": model, "arm": arm, "metric": m, "n_seeds": len(merged),
                         **paired_test(a, b)})
    return pd.DataFrame(rows)


def run_experiments(cfg: dict, save_models: bool = True) -> dict[str, pd.DataFrame]:
    """Run every seed, write all result files, and return the main tables."""
    t_start = time.time()
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = resolve_path(cfg["paths"]["models_dir"]) / "experiments" if save_models else None
    df = load_from_config(cfg)

    all_rows: dict[str, list] = {}
    for seed in cfg["experiments"]["seeds"]:
        logger.info("===== seed %d =====", seed)
        for key, vals in run_seed(df, cfg, seed, models_dir).items():
            all_rows.setdefault(key, []).extend(vals)

    tables = {k: pd.DataFrame(v) for k, v in all_rows.items()}
    metrics = tables["metrics"]
    tables["metrics_summary"] = mean_std(metrics, ["arm", "model"], METRIC_NAMES)
    tables["significance"] = significance_table(metrics)
    fs = tables["fairness_summary"]
    if len(fs):
        tables["fairness_summary_agg"] = mean_std(
            fs, ["arm", "model", "attribute"],
            ["demographic_parity_diff", "equal_opportunity_diff"])
    gq = tables["generator_quality"]
    if len(gq):
        tables["generator_quality_agg"] = mean_std(
            gq, ["generator"], [c for c in gq.columns if c not in ("seed", "generator")])

    for name, table in tables.items():
        table.to_csv(results_dir / f"{name}.csv", index=False)
    meta = {
        "synthdx_version": __version__, "python": platform.python_version(),
        "platform": platform.platform(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(time.time() - t_start, 1), "config": cfg,
        "n_records": len(df),
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    logger.info("Experiments finished in %.1fs; results in %s", meta["runtime_seconds"], results_dir)
    return tables
