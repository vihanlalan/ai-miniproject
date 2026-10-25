"""Forward-chaining inference engine with MYCIN certainty factors.

Recognise-act cycle:
    1. Match   - build the agenda (conflict set): every rule that has not yet
                 fired and whose conditions all hold in working memory.
    2. Resolve - pick one rule: highest priority, then highest specificity
                 (more premises), then knowledge-base order.
    3. Act     - fire it: contribution = rule CF x min(premise CFs), combined
                 into the conclusion fact with the parallel-combination formula.
Loop safety: refractoriness (each rule fires at most once per patient) and a
hard ``max_cycles`` guard, so the agenda always empties.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from synthdx.expert.facts import WorkingMemory, derive_facts
from synthdx.expert.knowledge_base import KnowledgeBase, Rule, load_knowledge_base
from synthdx.utils import get_logger

logger = get_logger(__name__)

GOAL = "heart_disease"


@dataclass
class FiringRecord:
    """One step of the inference trace."""

    step: int
    rule_id: str
    rule_name: str
    conclusion: str
    premise_cf: float
    rule_cf: float
    contribution: float
    cf_before: float
    cf_after: float
    matched: list[str]
    agenda_size: int


@dataclass
class NearMiss:
    """A rule that did not fire although most of its conditions held."""

    rule_id: str
    rule_name: str
    conclusion: str
    rule_cf: float
    satisfied: int
    total: int
    failed: list[str]


@dataclass
class Diagnosis:
    """Result of running the engine on one patient."""

    positive: bool
    cf: float
    risk_level: str
    label: str
    trace: list[FiringRecord]
    near_misses: list[NearMiss]
    working_memory: dict[str, dict[str, Any]]
    recommendations: list[str] = field(default_factory=list)
    threshold: float = 0.2


class InferenceEngine:
    """Rule-based diagnostic engine."""

    def __init__(
        self,
        kb: KnowledgeBase | None = None,
        premise_threshold: float = 0.2,
        decision_threshold: float = 0.2,
        risk_bands: Mapping[str, float] | None = None,
        max_cycles: int = 500,
        goal: str = GOAL,
    ) -> None:
        self.kb = kb if kb is not None else load_knowledge_base()
        self.premise_threshold = premise_threshold
        self.decision_threshold = decision_threshold
        self.risk_bands = dict(risk_bands or {"medium": 0.2, "high": 0.6})
        self.max_cycles = max_cycles
        self.goal = goal
        self._order = {r.id: i for i, r in enumerate(self.kb.rules)}
        # fact name -> rules with a premise on that fact (for incremental matching)
        self._dependents: dict[str, list[Rule]] = {}
        for r in self.kb.rules:
            for fact in {c.fact for c in r.conditions}:
                self._dependents.setdefault(fact, []).append(r)

    @classmethod
    def from_config(cls, cfg: dict, kb: KnowledgeBase | None = None) -> InferenceEngine:
        e = cfg["expert"]
        return cls(
            kb if kb is not None else load_knowledge_base(e.get("rules_path")),
            premise_threshold=e.get("premise_threshold", 0.2),
            decision_threshold=e.get("decision_threshold", 0.2),
            risk_bands=e.get("risk_bands"),
        )

    def clone(self, kb: KnowledgeBase | None = None,
              decision_threshold: float | None = None) -> InferenceEngine:
        """Copy with a different knowledge base and/or decision threshold."""
        return InferenceEngine(
            kb or self.kb, self.premise_threshold,
            self.decision_threshold if decision_threshold is None else decision_threshold,
            self.risk_bands, self.max_cycles, self.goal,
        )

    # ------------------------------------------------------------ inference
    def _match(self, rule: Rule, wm: WorkingMemory) -> tuple[bool, float, list[str]]:
        cfs, reasons = [], []
        for cond in rule.conditions:
            res = cond.evaluate(wm, self.premise_threshold)
            if not res.satisfied:
                return False, 0.0, []
            cfs.append(res.cf)
            reasons.append(res.reason)
        return True, min(cfs), reasons

    def _resolve(self, agenda: list[tuple[Rule, float, list[str]]]):
        return min(agenda, key=lambda item: (-item[0].priority, -item[0].specificity,
                                             self._order[item[0].id]))

    def infer(self, wm: WorkingMemory) -> list[FiringRecord]:
        """Run forward chaining to quiescence, mutating ``wm``; return the trace."""
        fired: set[str] = set()
        trace: list[FiringRecord] = []
        # Incremental matching (a light form of RETE): match every rule once, then
        # after each firing re-match only rules that test the fact that changed.
        matches = {r.id: self._match(r, wm) for r in self.kb.rules}
        for _cycle in range(self.max_cycles):
            agenda = [(r, matches[r.id][1], matches[r.id][2]) for r in self.kb.rules
                      if r.id not in fired and matches[r.id][0]]
            if not agenda:
                break
            rule, premise_cf, reasons = self._resolve(agenda)
            fired.add(rule.id)
            contribution = rule.cf * premise_cf
            before = wm.cf(rule.conclusion)
            after = wm.add_evidence(rule.conclusion, contribution, rule.id)
            for dependent in self._dependents.get(rule.conclusion, ()):
                matches[dependent.id] = self._match(dependent, wm)
            trace.append(FiringRecord(
                step=len(trace) + 1, rule_id=rule.id, rule_name=rule.name,
                conclusion=rule.conclusion, premise_cf=round(premise_cf, 4),
                rule_cf=rule.cf, contribution=round(contribution, 4),
                cf_before=round(before, 4), cf_after=round(after, 4),
                matched=reasons, agenda_size=len(agenda),
            ))
        else:
            logger.warning("Inference stopped after max_cycles=%d", self.max_cycles)
        return trace

    def near_misses(self, wm: WorkingMemory, fired: set[str], limit: int = 8) -> list[NearMiss]:
        """'Why not' analysis: unfired rules that were at most one condition short
        (or at least half satisfied)."""
        misses = []
        for rule in self.kb.rules:
            if rule.id in fired:
                continue
            results = [c.evaluate(wm, self.premise_threshold) for c in rule.conditions]
            ok = sum(r.satisfied for r in results)
            n = len(results)
            failed = [f"{c.describe()} -> {r.reason}"
                      for c, r in zip(rule.conditions, results) if not r.satisfied]
            if (n == 1 and _is_close(rule.conditions[0], wm, self.premise_threshold)) or \
               (n > 1 and ok >= 1 and ok >= n - 1):
                misses.append(NearMiss(rule.id, rule.name, rule.conclusion, rule.cf, ok, n, failed))
        misses.sort(key=lambda m: (-m.satisfied / m.total, m.rule_id))
        return misses[:limit]

    def risk_level(self, cf: float) -> str:
        if cf >= self.risk_bands["high"]:
            return "high"
        if cf >= self.risk_bands["medium"]:
            return "medium"
        return "low"

    def diagnose(self, record: Mapping[str, Any]) -> Diagnosis:
        """Full diagnosis with trace, 'why not' explanations and recommendations."""
        wm = derive_facts(record)
        trace = self.infer(wm)
        cf = wm.cf(self.goal)
        positive = cf >= self.decision_threshold
        recs = [f.name for f in wm.facts()
                if f.name.startswith("recommend_") and f.cf >= self.premise_threshold]
        return Diagnosis(
            positive=positive, cf=round(cf, 4), risk_level=self.risk_level(cf),
            label="Heart disease likely" if positive else "Heart disease unlikely",
            trace=trace, near_misses=self.near_misses(wm, {t.rule_id for t in trace}),
            working_memory=wm.as_dict(), recommendations=recs,
            threshold=self.decision_threshold,
        )

    # ------------------------------------------------------------ batch API
    def score_record(self, record: Mapping[str, Any]) -> tuple[float, set[str]]:
        """Goal CF and the set of fired rule ids for one record."""
        wm = derive_facts(record)
        trace = self.infer(wm)
        return wm.cf(self.goal), {t.rule_id for t in trace}

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Goal CF for every row (usable as a ROC score in [-1, 1])."""
        return np.array([self.score_record(r)[0] for r in df.to_dict("records")])

    def predict(self, df: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        t = self.decision_threshold if threshold is None else threshold
        return (self.score(df) >= t).astype(int)


def _is_close(cond, wm: WorkingMemory, threshold: float) -> bool:
    """Single-premise rules count as near misses when a numeric/CF test was
    missed by a small margin (within 10%, or CF within 0.1 of the threshold)."""
    fact = wm.get(cond.fact)
    if fact is None:
        return False
    if cond.op == "is":
        target = max(threshold, cond.min_cf or 0.0)
        return abs(abs(fact.cf) - target) < 0.1 and (fact.cf > 0) == bool(cond.value)
    if cond.op in {">", ">=", "<", "<="} and isinstance(fact.value, (int, float)):
        ref = float(cond.value)
        return abs(float(fact.value) - ref) <= 0.1 * max(abs(ref), 1.0)
    return False
