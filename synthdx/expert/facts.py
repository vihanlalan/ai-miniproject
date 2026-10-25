"""Working memory and fact derivation from a raw patient record.

Each fact has a value and a certainty factor (CF). Raw measurements become
crisp facts (CF 1.0). Derived facts discretise them into clinical categories
(age bands, BP categories, ...) and a few "fuzzy-ish" facts whose CF is a
linear membership degree, e.g. ``hypertension`` grows from 0 at 125 mm Hg to
1 at 160 mm Hg. Hypothesis facts concluded by rules accumulate CF evidence
with the MYCIN parallel-combination formula.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from synthdx.expert.certainty import combine

CHEST_PAIN = {1: "typical_angina", 2: "atypical_angina", 3: "non_anginal", 4: "asymptomatic"}
REST_ECG = {0: "normal", 1: "st_t_abnormality", 2: "lv_hypertrophy"}
ST_SLOPE = {1: "upsloping", 2: "flat", 3: "downsloping"}
THAL = {3: "normal", 6: "fixed_defect", 7: "reversible_defect"}

REQUIRED_FIELDS = (
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
)


@dataclass
class Fact:
    """A single entry in working memory."""

    name: str
    value: Any
    cf: float = 1.0
    source: str = "input"


class WorkingMemory:
    """Dictionary-like store of facts with CF-aware evidence accumulation."""

    def __init__(self) -> None:
        self._facts: dict[str, Fact] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._facts

    def __len__(self) -> int:
        return len(self._facts)

    def get(self, name: str) -> Fact | None:
        return self._facts.get(name)

    def assert_fact(self, name: str, value: Any, cf: float = 1.0, source: str = "input") -> None:
        """Add or overwrite a fact."""
        self._facts[name] = Fact(name, value, float(cf), source)

    def cf(self, name: str) -> float:
        """CF of a hypothesis fact (0.0 when unknown)."""
        fact = self._facts.get(name)
        return fact.cf if fact is not None else 0.0

    def add_evidence(self, name: str, cf: float, source: str) -> float:
        """Combine new CF evidence for a hypothesis; returns the updated CF."""
        new_cf = combine(self.cf(name), cf)
        prev = self._facts.get(name)
        src = source if prev is None else f"{prev.source},{source}"
        self._facts[name] = Fact(name, True, new_cf, src)
        return new_cf

    def facts(self) -> list[Fact]:
        return list(self._facts.values())

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {f.name: {"value": f.value, "cf": round(f.cf, 4), "source": f.source}
                for f in self._facts.values()}


def ramp_up(x: float, low: float, high: float) -> float:
    """Membership rising linearly from 0 at ``low`` to 1 at ``high``."""
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return (x - low) / (high - low)


def ramp_down(x: float, low: float, high: float) -> float:
    """Membership falling linearly from 1 at ``low`` to 0 at ``high``."""
    return 1.0 - ramp_up(x, low, high)


def age_band(age: float) -> str:
    return "young" if age < 45 else ("middle" if age < 60 else "senior")


def bp_category(bp: float) -> str:
    """ACC/AHA 2017 categories (systolic only; diastolic not available)."""
    if bp < 120:
        return "normal"
    if bp < 130:
        return "elevated"
    if bp < 140:
        return "stage1_hypertension"
    return "stage2_hypertension"


def chol_category(chol: float) -> str:
    """NCEP ATP III total-cholesterol categories."""
    return "desirable" if chol < 200 else ("borderline" if chol < 240 else "high")


def st_depression_level(oldpeak: float) -> str:
    return "none" if oldpeak < 0.5 else ("mild" if oldpeak < 2.0 else "significant")


def _fuzzy(wm: WorkingMemory, name: str, membership: float) -> None:
    """Assert a fuzzy boolean: True with CF=membership, or False with CF 1."""
    if membership > 0:
        wm.assert_fact(name, True, membership, "derived")
    else:
        wm.assert_fact(name, False, 1.0, "derived")


def derive_facts(record: Mapping[str, Any]) -> WorkingMemory:
    """Build the initial working memory for one patient record."""
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise KeyError(f"patient record missing fields: {missing}")
    r = {k: float(record[k]) for k in REQUIRED_FIELDS}
    wm = WorkingMemory()
    for key, value in r.items():
        wm.assert_fact(key, value, 1.0, "input")

    age = r["age"]
    hr_pct = r["thalach"] / max(220.0 - age, 1.0)
    derived: dict[str, Any] = {
        "age_band": age_band(age),
        "is_male": int(r["sex"]) == 1,
        "chest_pain": CHEST_PAIN.get(int(r["cp"]), "unknown"),
        "bp_category": bp_category(r["trestbps"]),
        "chol_category": chol_category(r["chol"]),
        "high_fasting_glucose": int(r["fbs"]) == 1,
        "rest_ecg": REST_ECG.get(int(r["restecg"]), "unknown"),
        "exercise_angina": int(r["exang"]) == 1,
        "st_depression": st_depression_level(r["oldpeak"]),
        "st_slope": ST_SLOPE.get(int(r["slope"]), "unknown"),
        "thal_result": THAL.get(int(r["thal"]), "unknown"),
        "hr_reserve_pct": round(hr_pct, 3),
    }
    for key, value in derived.items():
        wm.assert_fact(key, value, 1.0, "derived")

    _fuzzy(wm, "elderly", ramp_up(age, 50, 70))
    _fuzzy(wm, "hypertension", ramp_up(r["trestbps"], 125, 160))
    _fuzzy(wm, "hypercholesterolemia", ramp_up(r["chol"], 200, 280))
    _fuzzy(wm, "significant_st_depression", ramp_up(r["oldpeak"], 1.0, 3.0))
    _fuzzy(wm, "low_heart_rate_response", ramp_down(hr_pct, 0.65, 0.85))
    _fuzzy(wm, "good_heart_rate_response", ramp_up(hr_pct, 0.85, 1.0))
    return wm
