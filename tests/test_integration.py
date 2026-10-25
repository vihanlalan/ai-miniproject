"""End-to-end pipeline on a tiny configuration (should finish well under 2 minutes)."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from synthdx.config import load_config
from synthdx.experiments import run as run_cli
from synthdx.experiments.report import generate_report
from synthdx.experiments.runner import run_experiments, significance_table
from synthdx.experiments.train import train_demo_models

pytestmark = pytest.mark.slow


@pytest.fixture
def tiny_cfg(tmp_path):
    return load_config(overrides={
        "paths": {"results_dir": str(tmp_path / "results"), "models_dir": str(tmp_path / "models")},
        "experiments": {"seeds": [0, 1], "classifiers": ["logistic_regression"]},
        "gan": {"ctgan": {"epochs": 2}, "cgan": {"epochs": 3}},
    })


def test_full_pipeline(tiny_cfg, tmp_path):
    t0 = time.time()
    tables = run_experiments(tiny_cfg)
    report = generate_report(tiny_cfg["paths"]["results_dir"])
    elapsed = time.time() - t0
    assert elapsed < 120, f"pipeline too slow: {elapsed:.0f}s"

    rd = tmp_path / "results"
    for name in ("metrics", "metrics_summary", "predictions", "fairness_summary",
                 "generator_quality", "significance", "losses", "arm_sizes"):
        assert (rd / f"{name}.csv").exists(), name
    for fig in ("recall_by_arm", "roc_curves", "gan_losses", "fairness_eod"):
        assert (rd / "figures" / f"{fig}.png").exists(), fig
    assert (tmp_path / "models" / "experiments" / "ctgan_seed0.pkl").exists()

    m = tables["metrics"]
    assert set(m["arm"]) == {"A_real", "B_ctgan", "B_cgan", "C_smote", "D_ctgan", "D_cgan"}
    assert set(m["model"]) == {"logistic_regression", "expert_fixed",
                               "expert_calibrated", "expert_tuned"}
    assert len(m) == 2 * 6 * 4
    # every arm is evaluated on the same test records within a seed
    preds = pd.read_csv(rd / "predictions.csv")
    ids = preds.groupby(["seed", "arm", "model"])["record_id"].apply(frozenset)
    for seed in (0, 1):
        assert ids.loc[seed].nunique() == 1
    # the fixed expert system does not depend on the training arm
    fixed = m[m.model == "expert_fixed"].groupby("seed")["recall"].nunique()
    assert (fixed == 1).all()
    text = report.read_text(encoding="utf-8")
    assert "Key findings" in text and "not for clinical use" in text


def test_significance_table_shape():
    rows = []
    for seed in range(3):
        for arm, base in (("A_real", 0.2), ("B_x", 0.5)):
            rows.append({"seed": seed, "arm": arm, "model": "m", "recall": base + seed * 0.01,
                         "f1": 0.5, "balanced_accuracy": 0.5, "roc_auc": 0.5})
    sig = significance_table(pd.DataFrame(rows))
    r = sig[sig.metric == "recall"].iloc[0]
    assert r["arm"] == "B_x" and r["mean_diff"] == pytest.approx(0.3)


def test_demo_training(tiny_cfg, tmp_path):
    summary = train_demo_models(tiny_cfg, generators=["cgan"], epochs={"cgan": 2}, n_samples=40)
    demo = tmp_path / "results" / "demo"
    assert (tmp_path / "models" / "cgan.pkl").exists()
    synth = pd.read_csv(demo / "synthetic_cgan.csv")
    assert len(synth) == 40 and synth["target"].sum() == 20
    assert "mia_auc" in summary["cgan"]


def test_cli_quick(monkeypatch, tiny_cfg, tmp_path):
    called = {}
    monkeypatch.setattr(run_cli, "run_experiments", lambda cfg: called.setdefault("cfg", cfg))
    monkeypatch.setattr(run_cli, "generate_report", lambda d: called.setdefault("dir", d))
    run_cli.main(["--quick", "--seeds", "3"])
    assert called["cfg"]["experiments"]["seeds"] == [3]
    assert called["cfg"]["gan"]["ctgan"]["epochs"] == 30
