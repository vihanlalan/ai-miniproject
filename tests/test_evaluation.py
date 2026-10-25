"""Tests for metric, fairness, statistics, fidelity and privacy functions on toy inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthdx.evaluation import (
    add_sensitive_columns,
    classification_metrics,
    column_fidelity,
    correlation_difference,
    dcr_stats,
    demographic_parity_difference,
    detectability,
    equal_opportunity_difference,
    exact_duplicates,
    fidelity_summary,
    group_metrics,
    mean_std,
    membership_inference_auc,
    paired_test,
    privacy_summary,
    roc_points,
)


class TestClassificationMetrics:
    def test_toy_confusion(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        p = np.array([1, 1, 0, 1, 0, 0, 0, 0])
        m = classification_metrics(y, p)
        assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
        assert m["accuracy"] == pytest.approx(6 / 8)
        assert m["recall"] == pytest.approx(2 / 3)
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["specificity"] == pytest.approx(4 / 5)
        assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)
        assert m["f1"] == pytest.approx(2 / 3)
        assert np.isnan(m["roc_auc"])

    def test_scores_auc(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.4, 0.35, 0.8])
        m = classification_metrics(y, (s >= 0.5).astype(int), s)
        assert m["roc_auc"] == pytest.approx(0.75)
        assert 0 < m["pr_auc"] <= 1
        fpr, tpr = roc_points(y, s)
        assert fpr[0] == 0 and tpr[-1] == 1

    def test_single_class(self):
        m = classification_metrics(np.zeros(4), np.zeros(4), np.zeros(4))
        assert m["accuracy"] == 1 and m["recall"] == 0 and np.isnan(m["roc_auc"])


class TestFairness:
    def test_hand_computed_gaps(self):
        y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        p = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        g = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
        gm = group_metrics(y, p, g).set_index("group")
        # selection: a = 2/4, b = 1/4 ; TPR: a = 1/2, b = 1/2
        assert gm.loc["a", "selection_rate"] == 0.5 and gm.loc["b", "selection_rate"] == 0.25
        assert demographic_parity_difference(p, g) == pytest.approx(0.25)
        assert equal_opportunity_difference(y, p, g) == pytest.approx(0.0)
        assert gm.loc["a", "accuracy"] == pytest.approx(0.5)

    def test_maximal_unfairness(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([1, 1, 0, 0])
        g = np.array(["m", "m", "f", "f"])
        assert demographic_parity_difference(p, g) == 1.0
        assert equal_opportunity_difference(y, p, g) == 1.0

    def test_group_without_positives(self):
        y = np.array([1, 0, 0, 0])
        p = np.array([1, 0, 1, 0])
        g = np.array(["a", "a", "b", "b"])
        assert np.isnan(group_metrics(y, p, g).set_index("group").loc["b", "recall"])
        assert np.isnan(equal_opportunity_difference(y, p, g))

    def test_sensitive_columns(self, heart_df):
        out = add_sensitive_columns(heart_df, 55)
        assert set(out["sex_group"]) == {"male", "female"}
        assert set(out["age_group"]) == {"<55", ">=55"}


class TestStats:
    def test_paired(self):
        a = np.array([0.5, 0.6, 0.55, 0.52, 0.58])
        b = a + np.array([0.10, 0.12, 0.08, 0.11, 0.09])
        r = paired_test(a, b)
        assert r["mean_diff"] == pytest.approx(0.1)
        assert r["t_pvalue"] < 0.01
        assert r["wilcoxon_pvalue"] == pytest.approx(0.0625)   # minimum for n=5

    def test_degenerate(self):
        a = np.ones(5)
        assert paired_test(a, a)["t_pvalue"] == 1.0
        assert np.isnan(paired_test(a, a)["wilcoxon_pvalue"])
        assert paired_test(a, a + 0.1)["t_pvalue"] == 0.0
        with pytest.raises(ValueError):
            paired_test([1.0], [2.0])

    def test_mean_std(self):
        df = pd.DataFrame({"arm": ["A", "A", "B"], "recall": [0.2, 0.4, 0.9]})
        out = mean_std(df, ["arm"], ["recall"]).set_index("arm")
        assert out.loc["A", "recall_mean"] == pytest.approx(0.3)
        assert out.loc["A", "recall_std"] == pytest.approx(np.std([0.2, 0.4], ddof=1))


class TestFidelityPrivacy:
    def test_identical_data(self, split):
        real = split.train_pool
        cols = column_fidelity(real, real)
        assert cols["js_distance"].max() < 1e-3
        assert cols["ks_stat"].dropna().max() == 0
        assert correlation_difference(real, real)["corr_max_abs_diff"] == pytest.approx(0)

    def test_shifted_data_is_detectable(self, split):
        real = split.train_pool
        fake = real.copy()
        fake["chol"] = fake["chol"] + 150
        fake["age"] = 30
        assert detectability(real, fake, seed=0) > 0.9
        assert detectability(real, real.sample(frac=1, random_state=1), seed=0) < 0.75
        summ = fidelity_summary(real, fake)
        assert summ["mean_ks_stat"] > 0.1

    def test_copying_is_flagged(self, split):
        train, test = split.train, split.test
        copied = train.copy()
        assert exact_duplicates(copied, train) == len(train)
        dcr = dcr_stats(copied, train, test)
        assert dcr["dcr_synth_median"] == pytest.approx(0) and dcr["dcr_ratio"] < 0.01
        assert membership_inference_auc(copied, train, test) > 0.9

    def test_independent_data_is_private(self, heart_df):
        # members and non-members drawn from the same pool; "synthetic" = a third disjoint part
        shuffled = heart_df.sample(frac=1, random_state=0)
        members, non_members, other = shuffled[:100], shuffled[100:200], shuffled[200:]
        assert exact_duplicates(other, members) == 0
        p = privacy_summary(other, members, non_members)
        assert 0.3 < p["mia_auc"] < 0.7
        assert p["dcr_ratio"] > 0.7
