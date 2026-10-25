"""Shared chart palette (validated categorical order; colour follows the entity).

Every arm / model / generator keeps the same colour in the report figures
and the Streamlit app.
"""

from __future__ import annotations

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
               "#6250d6", "#e34948"]

INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b52f2f"]  # blue - gray - red

ARM_ORDER = ["A_real", "B_ctgan", "B_cgan", "C_smote", "D_ctgan", "D_cgan"]
ARM_COLORS = dict(zip(ARM_ORDER, CATEGORICAL))

MODEL_ORDER = ["logistic_regression", "random_forest", "gradient_boosting",
               "expert_fixed", "expert_calibrated", "expert_tuned"]
MODEL_COLORS = dict(zip(MODEL_ORDER, CATEGORICAL))

DATA_COLORS = {"real": CATEGORICAL[0], "synthetic": CATEGORICAL[1]}
CLASS_COLORS = {0: CATEGORICAL[0], 1: CATEGORICAL[1]}


def ordered(items, order: list[str]) -> list[str]:
    """Sort ``items`` by a canonical order (unknown items go last, alphabetically)."""
    rank = {k: i for i, k in enumerate(order)}
    return sorted(items, key=lambda x: (rank.get(x, len(order)), x))
