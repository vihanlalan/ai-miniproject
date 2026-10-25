"""MYCIN certainty-factor (CF) algebra.

CFs live in [-1, 1]: +1 = certainly true, -1 = certainly false, 0 = unknown.

* Conjunction of premises:   CF(p1 AND p2) = min(CF(p1), CF(p2))
* Rule application:          CF(conclusion) = CF(rule) * CF(premises)
* Parallel combination of two pieces of evidence for the same conclusion
  (Shortliffe & Buchanan 1975, revised in EMYCIN):

      both > 0 : a + b * (1 - a)
      both < 0 : a + b * (1 + a)
      mixed    : (a + b) / (1 - min(|a|, |b|))
"""

from __future__ import annotations

from collections.abc import Iterable


def _check(cf: float) -> float:
    if not -1.0 <= cf <= 1.0:
        raise ValueError(f"certainty factor {cf} outside [-1, 1]")
    return float(cf)


def conjunction(premise_cfs: Iterable[float]) -> float:
    """CF of a conjunction of premises: the minimum premise CF."""
    cfs = [_check(c) for c in premise_cfs]
    if not cfs:
        raise ValueError("a rule needs at least one premise")
    return min(cfs)


def rule_conclusion_cf(rule_cf: float, premise_cfs: Iterable[float]) -> float:
    """CF contributed by one rule firing: rule CF x min(premise CFs)."""
    return _check(rule_cf) * conjunction(premise_cfs)


def combine(a: float, b: float) -> float:
    """Combine two CFs supporting the same conclusion (parallel combination)."""
    a, b = _check(a), _check(b)
    if a >= 0 and b >= 0:
        result = a + b * (1 - a)
    elif a < 0 and b < 0:
        result = a + b * (1 + a)
    else:
        denom = 1 - min(abs(a), abs(b))
        # total contradiction (+1 vs -1) is undefined; treat as "unknown"
        result = 0.0 if denom == 0 else (a + b) / denom
    return max(-1.0, min(1.0, result))


def combine_all(cfs: Iterable[float]) -> float:
    """Fold :func:`combine` over many CFs (the operation is commutative)."""
    total = 0.0
    for cf in cfs:
        total = combine(total, cf)
    return total
