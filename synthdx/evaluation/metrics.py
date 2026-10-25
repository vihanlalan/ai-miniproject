"""Classification metrics (minority = positive = disease class)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_NAMES: list[str] = [
    "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
    "f1", "roc_auc", "pr_auc",
]


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> dict[str, float]:
    """All headline metrics plus the confusion-matrix cells.

    ``recall`` is minority-class (disease) recall, i.e. sensitivity.
    ROC-AUC / PR-AUC are NaN if ``y_score`` is missing or only one class is present.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)
        if len(np.unique(y_true)) > 1 else float("nan"),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    return {k: float(v) for k, v in out.items()}


def roc_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False- and true-positive rates for an ROC curve."""
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), y_score)
    return fpr, tpr
