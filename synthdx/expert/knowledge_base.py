"""Rule representation, YAML parsing and validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from synthdx.config import resolve_path
from synthdx.expert.facts import WorkingMemory

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "knowledge" / "rules.yaml"
VALID_OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "is"}


class RuleValidationError(ValueError):
    """Raised when the knowledge base is malformed."""


@dataclass(frozen=True)
class ConditionResult:
    """Outcome of testing one condition against working memory."""

    satisfied: bool
    cf: float
    reason: str


@dataclass(frozen=True)
class Condition:
    """One premise ``fact op value`` of a rule."""

    fact: str
    op: str
    value: Any
    min_cf: float | None = None

    def describe(self) -> str:
        if self.op == "is":
            extra = f" (CF >= {self.min_cf})" if self.min_cf is not None else ""
            return f"{self.fact} is {str(self.value).lower()}{extra}"
        return f"{self.fact} {self.op} {self.value}"

    def evaluate(self, wm: WorkingMemory, threshold: float) -> ConditionResult:
        """Test this condition. Premise CF is the fact's CF (or |CF| for 'is false')."""
        fact = wm.get(self.fact)
        if fact is None:
            return ConditionResult(False, 0.0, f"'{self.fact}' was never established")
        if self.op == "is":
            need = max(threshold, self.min_cf or 0.0)
            if self.value:
                ok = fact.cf >= need
                return ConditionResult(ok, fact.cf if ok else 0.0,
                                       f"{self.fact} CF={fact.cf:+.2f} (needs >= {need:.2f})")
            ok = fact.cf <= -need
            return ConditionResult(ok, -fact.cf if ok else 0.0,
                                   f"{self.fact} CF={fact.cf:+.2f} (needs <= {-need:.2f})")
        if fact.cf < threshold:
            return ConditionResult(False, 0.0,
                                   f"{self.fact} CF={fact.cf:.2f} below threshold {threshold}")
        ok = _compare(fact.value, self.op, self.value)
        return ConditionResult(ok, fact.cf if ok else 0.0,
                               f"{self.fact} = {_fmt(fact.value)} ({'meets' if ok else 'fails'} "
                               f"{self.op} {self.value})")


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op == "in":
        return actual in expected
    try:
        a, e = float(actual), float(expected)
    except (TypeError, ValueError):
        return False
    return {">": a > e, ">=": a >= e, "<": a < e, "<=": a <= e}[op]


@dataclass(frozen=True)
class Rule:
    """An IF-THEN production rule with a certainty factor."""

    id: str
    name: str
    conditions: tuple[Condition, ...]
    conclusion: str
    cf: float
    priority: int = 0
    category: str = "general"
    rationale: str = ""
    conclusion_value: Any = True

    @property
    def specificity(self) -> int:
        """Number of premises: more specific rules win ties in conflict resolution."""
        return len(self.conditions)

    def describe(self) -> str:
        conds = " AND ".join(c.describe() for c in self.conditions)
        return f"IF {conds} THEN {self.conclusion} (CF {self.cf:+.2f})"


@dataclass
class KnowledgeBase:
    """An ordered collection of validated rules."""

    rules: list[Rule]
    disclaimer: str = ""
    version: str = ""
    _by_id: dict[str, Rule] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {r.id: r for r in self.rules}

    def __len__(self) -> int:
        return len(self.rules)

    def __getitem__(self, rule_id: str) -> Rule:
        return self._by_id[rule_id]

    def with_cfs(self, new_cfs: dict[str, float]) -> KnowledgeBase:
        """Return a copy with some rule CFs replaced (used by tuning)."""
        rules = [
            Rule(**{**r.__dict__, "cf": float(new_cfs[r.id])}) if r.id in new_cfs else r
            for r in self.rules
        ]
        return KnowledgeBase(rules=rules, disclaimer=self.disclaimer, version=self.version)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "id": r.id, "name": r.name, "category": r.category, "priority": r.priority,
            "conditions": " AND ".join(c.describe() for c in r.conditions),
            "conclusion": r.conclusion, "cf": r.cf, "rationale": r.rationale,
        } for r in self.rules])


def parse_rule(raw: dict[str, Any]) -> Rule:
    """Parse and validate one rule dictionary."""
    rid = raw.get("id", "<missing id>")
    for key in ("id", "name", "if", "then", "cf"):
        if key not in raw:
            raise RuleValidationError(f"rule {rid}: missing key '{key}'")
    cf = raw["cf"]
    if not isinstance(cf, (int, float)) or not -1 <= cf <= 1:
        raise RuleValidationError(f"rule {rid}: cf must be a number in [-1, 1], got {cf!r}")
    if not isinstance(raw["if"], list) or not raw["if"]:
        raise RuleValidationError(f"rule {rid}: 'if' must be a non-empty list")
    conditions = []
    for cond in raw["if"]:
        if not isinstance(cond, dict) or not {"fact", "op", "value"} <= set(cond):
            raise RuleValidationError(f"rule {rid}: condition needs fact/op/value: {cond}")
        if cond["op"] not in VALID_OPS:
            raise RuleValidationError(f"rule {rid}: unknown operator {cond['op']!r}")
        if cond["op"] == "in" and not isinstance(cond["value"], list):
            raise RuleValidationError(f"rule {rid}: 'in' needs a list value")
        if cond["op"] == "is" and not isinstance(cond["value"], bool):
            raise RuleValidationError(f"rule {rid}: 'is' needs true/false")
        conditions.append(Condition(cond["fact"], cond["op"], cond["value"], cond.get("min_cf")))
    then = raw["then"]
    if not isinstance(then, dict) or "fact" not in then:
        raise RuleValidationError(f"rule {rid}: 'then' must contain a 'fact'")
    return Rule(
        id=str(raw["id"]), name=str(raw["name"]), conditions=tuple(conditions),
        conclusion=str(then["fact"]), conclusion_value=then.get("value", True),
        cf=float(cf), priority=int(raw.get("priority", 0)),
        category=str(raw.get("category", "general")),
        rationale=" ".join(str(raw.get("rationale", "")).split()),
    )


def parse_knowledge_base(data: dict[str, Any]) -> KnowledgeBase:
    """Validate a loaded YAML document and build a :class:`KnowledgeBase`."""
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise RuleValidationError("knowledge base must contain a 'rules' list")
    rules = [parse_rule(copy.deepcopy(r)) for r in data["rules"]]
    ids = [r.id for r in rules]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise RuleValidationError(f"duplicate rule ids: {sorted(dupes)}")
    return KnowledgeBase(rules, str(data.get("disclaimer", "")), str(data.get("version", "")))


def load_knowledge_base(path: str | Path | None = None) -> KnowledgeBase:
    """Load the rules YAML (defaults to the bundled knowledge base)."""
    p = resolve_path(path) if path else DEFAULT_RULES_PATH
    with open(p, encoding="utf-8") as fh:
        return parse_knowledge_base(yaml.safe_load(fh))
