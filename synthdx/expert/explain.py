"""Explanation facility: human-readable 'how' and 'why not' explanations."""

from __future__ import annotations

import pandas as pd

from synthdx.expert.engine import Diagnosis, InferenceEngine

FACT_LABELS = {
    "ischemia_evidence": "evidence of ischaemia",
    "perfusion_defect": "perfusion defect",
    "vessel_disease": "coronary vessel disease",
    "exercise_intolerance": "poor exercise capacity",
    "risk_factor_burden": "cardiovascular risk-factor burden",
    "structural_abnormality": "resting ECG abnormality",
    "heart_disease": "HEART DISEASE",
    "recommend_cardiology_referral": "recommendation: cardiology referral",
    "recommend_risk_factor_management": "recommendation: risk-factor management",
}


def trace_to_dataframe(diagnosis: Diagnosis) -> pd.DataFrame:
    """Tabular view of the firing trace."""
    return pd.DataFrame([{
        "step": t.step, "rule": t.rule_id, "name": t.rule_name, "concludes": t.conclusion,
        "premise CF": t.premise_cf, "rule CF": t.rule_cf, "contribution": t.contribution,
        "CF before": t.cf_before, "CF after": t.cf_after, "agenda size": t.agenda_size,
    } for t in diagnosis.trace])


def explain_how(diagnosis: Diagnosis, engine: InferenceEngine | None = None) -> list[str]:
    """Step-by-step narrative of the rules that fired."""
    lines = []
    for t in diagnosis.trace:
        direction = "supports" if t.contribution >= 0 else "argues against"
        label = FACT_LABELS.get(t.conclusion, t.conclusion)
        desc = engine.kb[t.rule_id].describe() if engine else t.rule_name
        lines.append(
            f"Step {t.step}: {t.rule_id} '{t.rule_name}' fired - {direction} {label}. "
            f"Contribution = rule CF {t.rule_cf:+.2f} x premise CF {t.premise_cf:.2f} = "
            f"{t.contribution:+.3f}; CF({t.conclusion}) {t.cf_before:+.3f} -> {t.cf_after:+.3f}. "
            f"[{desc}]"
        )
    return lines


def explain_why_not(diagnosis: Diagnosis) -> list[str]:
    """Explain rules that almost fired."""
    return [
        f"{m.rule_id} '{m.rule_name}' (would give {m.conclusion} CF {m.rule_cf:+.2f}) "
        f"did not fire: {m.satisfied}/{m.total} conditions held; failed: "
        + "; ".join(m.failed)
        for m in diagnosis.near_misses
    ]


def format_report(diagnosis: Diagnosis, engine: InferenceEngine | None = None) -> str:
    """Complete plain-text explanation suitable for display or logging."""
    out = [
        f"DIAGNOSIS: {diagnosis.label}",
        f"Certainty factor: {diagnosis.cf:+.3f} (decision threshold {diagnosis.threshold:+.2f})",
        f"Risk level: {diagnosis.risk_level.upper()}",
        "",
        "HOW (rules fired in order):",
        *(f"  {line}" for line in explain_how(diagnosis, engine)),
        "",
        "WHY NOT (rules that almost fired):",
        *(f"  {line}" for line in (explain_why_not(diagnosis) or ["none"])),
    ]
    if diagnosis.recommendations:
        out += ["", "RECOMMENDATIONS (demo logic): " + ", ".join(diagnosis.recommendations)]
    out += ["", "Academic demonstration only - not medical advice."]
    return "\n".join(out)
