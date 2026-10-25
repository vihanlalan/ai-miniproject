"""Tests for the expert system: CF math, rule parsing, inference, explanations, tuning."""

from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest
import yaml

from synthdx.expert import (
    InferenceEngine,
    RuleValidationError,
    calibrate_threshold,
    combine,
    combine_all,
    conjunction,
    derive_facts,
    explain_how,
    explain_why_not,
    fit_expert,
    format_report,
    load_knowledge_base,
    parse_knowledge_base,
    rule_conclusion_cf,
    trace_to_dataframe,
    tune_rule_cfs,
)
from synthdx.expert.facts import (
    age_band,
    bp_category,
    chol_category,
    ramp_down,
    ramp_up,
    st_depression_level,
)
from synthdx.expert.knowledge_base import DEFAULT_RULES_PATH
from synthdx.expert.tuning import empirical_cf

# A high-risk patient worked through by hand (see expected order below).
HIGH_RISK = dict(age=63, sex=1, cp=4, trestbps=150, chol=290, fbs=0, restecg=2, thalach=110,
                 exang=1, oldpeak=2.5, slope=2, ca=2, thal=7)
LOW_RISK = dict(age=35, sex=0, cp=3, trestbps=115, chol=180, fbs=0, restecg=0, thalach=185,
                exang=0, oldpeak=0.0, slope=1, ca=0, thal=3)


# ------------------------------------------------------------------ CF math
class TestCertaintyFactors:
    def test_both_positive(self):
        assert combine(0.6, 0.4) == pytest.approx(0.76)        # 0.6 + 0.4 * (1 - 0.6)

    def test_both_negative(self):
        assert combine(-0.6, -0.4) == pytest.approx(-0.76)     # -0.6 + -0.4 * (1 - 0.6)

    def test_mixed_signs(self):
        assert combine(0.6, -0.4) == pytest.approx(1 / 3)      # 0.2 / (1 - 0.4)
        assert combine(-0.6, 0.4) == pytest.approx(-1 / 3)

    def test_full_contradiction_is_unknown(self):
        assert combine(1.0, -1.0) == 0.0

    def test_identity_and_certainty(self):
        assert combine(0.0, 0.7) == pytest.approx(0.7)
        assert combine(1.0, 0.3) == pytest.approx(1.0)

    def test_combine_all_sequence(self):
        # 0.5 -> 0.75 -> 0.875
        assert combine_all([0.5, 0.5, 0.5]) == pytest.approx(0.875)
        assert combine_all([]) == 0.0

    def test_commutative_and_order_independent(self):
        assert combine(0.3, 0.8) == pytest.approx(combine(0.8, 0.3))
        assert combine(0.3, -0.8) == pytest.approx(combine(-0.8, 0.3))
        # same-sign evidence combined in any order gives one value
        perms = itertools.permutations([0.3, 0.7, 0.4])
        assert len({round(combine_all(p), 10) for p in perms}) == 1

    def test_rule_application(self):
        assert conjunction([0.9, 0.5, 0.7]) == 0.5
        assert rule_conclusion_cf(0.8, [0.9, 0.5]) == pytest.approx(0.4)
        assert rule_conclusion_cf(-0.6, [1.0]) == pytest.approx(-0.6)

    def test_validation(self):
        with pytest.raises(ValueError):
            combine(1.2, 0.1)
        with pytest.raises(ValueError):
            conjunction([])


# ------------------------------------------------------------------ facts
class TestFacts:
    def test_discretisation(self):
        assert [age_band(a) for a in (30, 50, 70)] == ["young", "middle", "senior"]
        assert [bp_category(b) for b in (110, 125, 135, 150)] == [
            "normal", "elevated", "stage1_hypertension", "stage2_hypertension"]
        assert [chol_category(c) for c in (180, 220, 260)] == ["desirable", "borderline", "high"]
        assert [st_depression_level(o) for o in (0, 1, 2.5)] == ["none", "mild", "significant"]

    def test_fuzzy_ramps(self):
        assert ramp_up(150, 125, 160) == pytest.approx(25 / 35)
        assert ramp_up(100, 125, 160) == 0 and ramp_up(200, 125, 160) == 1
        assert ramp_down(0.75, 0.65, 0.85) == pytest.approx(0.5)

    def test_derive_facts(self):
        wm = derive_facts(HIGH_RISK)
        assert wm.get("chest_pain").value == "asymptomatic"
        assert wm.get("hypertension").cf == pytest.approx(25 / 35)
        assert wm.get("elderly").cf == pytest.approx(0.65)
        assert wm.get("good_heart_rate_response").value is False
        assert wm.get("chol").value == 290 and wm.get("chol").cf == 1.0
        assert "chol" in wm and len(wm) > 20
        assert wm.cf("unknown_hypothesis") == 0.0

    def test_missing_field(self):
        rec = dict(HIGH_RISK)
        rec.pop("chol")
        with pytest.raises(KeyError):
            derive_facts(rec)


# ------------------------------------------------------------------ knowledge base
class TestKnowledgeBase:
    def test_loads_bundled_rules(self):
        kb = load_knowledge_base()
        assert 25 <= len(kb) <= 45
        assert "not medical advice" in kb.disclaimer.lower()
        for rule in kb.rules:
            assert -1 <= rule.cf <= 1 and rule.rationale
        df = kb.to_dataframe()
        assert {"id", "conditions", "cf", "rationale"} <= set(df.columns)
        assert any(c.op in (">", ">=", "<") for r in kb.rules for c in r.conditions)
        assert any(c.op == "==" and isinstance(c.value, str) for r in kb.rules for c in r.conditions)

    def test_every_premise_is_reachable(self):
        """Each fact a rule tests is either derived from the record or concluded by a rule."""
        kb = load_knowledge_base()
        derived = {f.name for f in derive_facts(HIGH_RISK).facts()}
        concluded = {r.conclusion for r in kb.rules}
        for rule in kb.rules:
            for c in rule.conditions:
                assert c.fact in derived | concluded, f"{rule.id}: {c.fact} unreachable"

    @pytest.fixture
    def raw(self):
        with open(DEFAULT_RULES_PATH, encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    @pytest.mark.parametrize("mutate, msg", [
        (lambda r: r["rules"][0].update(cf=1.5), "cf must be"),
        (lambda r: r["rules"][0]["if"][0].update(op="~"), "unknown operator"),
        (lambda r: r["rules"][1].update(id=r["rules"][0]["id"]), "duplicate"),
        (lambda r: r["rules"][0].pop("then"), "missing key"),
        (lambda r: r["rules"][0].update({"if": []}), "non-empty"),
        (lambda r: r["rules"][6]["if"][0].update(value="x"), "needs a list"),
        (lambda r: r["rules"][27]["if"][0].update(value="yes"), "true/false"),
        (lambda r: r["rules"][0]["if"][0].pop("value"), "fact/op/value"),
        (lambda r: r["rules"][0].update(then={"value": 1}), "'then'"),
    ])
    def test_validation_errors(self, raw, mutate, msg):
        bad = copy.deepcopy(raw)
        mutate(bad)
        with pytest.raises(RuleValidationError, match=msg):
            parse_knowledge_base(bad)

    def test_rejects_non_mapping(self):
        with pytest.raises(RuleValidationError):
            parse_knowledge_base({"no_rules": True})

    def test_with_cfs(self):
        kb = load_knowledge_base()
        kb2 = kb.with_cfs({"R28": 0.1})
        assert kb2["R28"].cf == 0.1 and kb["R28"].cf == 0.6


# ------------------------------------------------------------------ inference
class TestInference:
    EXPECTED_ORDER = [
        # priority 3 (findings): 2-premise rules first (specificity), then KB order
        "R05", "R23", "R01", "R03", "R04", "R09", "R12", "R15", "R16", "R18", "R19",
        "R20", "R21", "R24", "R27",
        # priority 2 (diagnosis)
        "R31", "R32", "R28", "R29", "R30", "R33", "R34", "R35",
        # priority 1 (recommendation)
        "R40",
    ]

    def test_high_risk_patient_fires_expected_rules_in_order(self):
        d = InferenceEngine().diagnose(HIGH_RISK)
        assert [t.rule_id for t in d.trace] == self.EXPECTED_ORDER
        assert d.positive and d.risk_level == "high" and d.cf > 0.9
        assert "recommend_cardiology_referral" in d.recommendations

    def test_hand_computed_intermediate_cf(self):
        d = InferenceEngine().diagnose(HIGH_RISK)
        # R05: 0.5 ; R01: 0.5 -> 0.75 ; R03: 0.5 -> 0.875 ; R04: 0.6*0.75=0.45 -> 0.93125
        assert d.working_memory["ischemia_evidence"]["cf"] == pytest.approx(0.93125, abs=1e-4)
        r04 = next(t for t in d.trace if t.rule_id == "R04")
        assert r04.premise_cf == pytest.approx(0.75) and r04.contribution == pytest.approx(0.45)
        assert r04.cf_before == pytest.approx(0.875)

    def test_low_risk_patient(self):
        d = InferenceEngine().diagnose(LOW_RISK)
        assert not d.positive and d.risk_level == "low" and d.cf < -0.5
        fired = [t.rule_id for t in d.trace]
        assert {"R36", "R37", "R38"} <= set(fired)

    def test_each_rule_fires_at_most_once(self, heart_df):
        eng = InferenceEngine()
        for rec in heart_df.head(40).to_dict("records"):
            ids = [t.rule_id for t in eng.diagnose(rec).trace]
            assert len(ids) == len(set(ids))

    def test_loop_safe_on_cyclic_rules(self):
        kb = parse_knowledge_base({"rules": [
            {"id": "A", "name": "a", "if": [{"fact": "exercise_angina", "op": "==", "value": True}],
             "then": {"fact": "x"}, "cf": 0.5},
            {"id": "B", "name": "b", "if": [{"fact": "x", "op": "is", "value": True}],
             "then": {"fact": "y"}, "cf": 0.9},
            {"id": "C", "name": "c", "if": [{"fact": "y", "op": "is", "value": True}],
             "then": {"fact": "x"}, "cf": 0.9},
        ]})
        trace = InferenceEngine(kb, goal="x").diagnose(HIGH_RISK).trace
        assert [t.rule_id for t in trace] == ["A", "B", "C"]

    def test_max_cycles_guard(self):
        eng = InferenceEngine(max_cycles=2)
        assert len(eng.diagnose(HIGH_RISK).trace) == 2

    def test_batch_scoring_and_auc(self, heart_df):
        from sklearn.metrics import roc_auc_score
        eng = InferenceEngine()
        scores = eng.score(heart_df)
        assert scores.min() >= -1 and scores.max() <= 1
        assert roc_auc_score(heart_df.target, scores) > 0.8      # rules are informative
        assert set(np.unique(eng.predict(heart_df))) <= {0, 1}

    def test_risk_bands(self):
        eng = InferenceEngine(risk_bands={"medium": 0.2, "high": 0.6})
        assert [eng.risk_level(c) for c in (-0.5, 0.3, 0.8)] == ["low", "medium", "high"]

    def test_from_config(self, cfg):
        eng = InferenceEngine.from_config(cfg)
        assert eng.premise_threshold == cfg["expert"]["premise_threshold"]
        assert len(eng.kb) == len(load_knowledge_base())


# ------------------------------------------------------------------ explanations
class TestExplanation:
    def test_how_explanation(self):
        eng = InferenceEngine()
        d = eng.diagnose(HIGH_RISK)
        how = explain_how(d, eng)
        assert len(how) == len(d.trace)
        assert how[0].startswith("Step 1: R05")
        assert "x premise CF" in how[0]
        df = trace_to_dataframe(d)
        assert list(df["rule"]) == [t.rule_id for t in d.trace]

    def test_why_not(self):
        d = InferenceEngine().diagnose(HIGH_RISK)
        ids = [m.rule_id for m in d.near_misses]
        assert "R02" in ids                                    # exang holds, cp is not typical
        r02 = next(m for m in d.near_misses if m.rule_id == "R02")
        assert r02.satisfied == 1 and r02.total == 2
        assert any("chest_pain" in f for f in r02.failed)
        assert explain_why_not(d)

    def test_full_report(self):
        eng = InferenceEngine()
        text = format_report(eng.diagnose(HIGH_RISK), eng)
        for part in ("DIAGNOSIS: Heart disease likely", "HOW", "WHY NOT", "not medical advice"):
            assert part in text
        assert "Heart disease unlikely" in format_report(eng.diagnose(LOW_RISK))


# ------------------------------------------------------------------ tuning
class TestTuning:
    def test_empirical_cf(self):
        assert empirical_cf(0.8, 0.5) == pytest.approx(0.6)
        assert empirical_cf(0.2, 0.5) == pytest.approx(-0.6)
        assert empirical_cf(0.5, 0.0) == 0.0

    def test_calibrate_threshold(self):
        scores = np.array([-0.8, -0.5, 0.1, 0.4, 0.7, 0.9])
        y = np.array([0, 0, 0, 1, 1, 1])
        t = calibrate_threshold(scores, y)
        assert ((scores >= t).astype(int) == y).all()
        t_f1 = calibrate_threshold(scores, y, metric="f1")
        assert 0.1 < t_f1 <= 0.4

    def test_tune_and_fit(self, split):
        eng = InferenceEngine()
        cfs = tune_rule_cfs(eng, split.train, shrinkage=10)
        goal_rules = {r.id for r in eng.kb.rules if r.conclusion == "heart_disease"}
        assert set(cfs) == goal_rules
        assert all(-0.95 <= v <= 0.95 for v in cfs.values())
        fitted = fit_expert(eng, split.train, tune_rules=True)
        assert fitted.kb["R28"].cf == pytest.approx(cfs["R28"])
        assert -1 <= fitted.decision_threshold <= 1
        plain = fit_expert(eng, split.train, tune_rules=False)
        assert plain.kb["R28"].cf == eng.kb["R28"].cf
