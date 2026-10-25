"""Evaluation: classification, fidelity, privacy, fairness and statistics."""

from synthdx.evaluation.fairness import (
    add_sensitive_columns,
    demographic_parity_difference,
    equal_opportunity_difference,
    group_metrics,
)
from synthdx.evaluation.fidelity import (
    column_fidelity,
    correlation_difference,
    correlation_matrices,
    detectability,
    fidelity_summary,
)
from synthdx.evaluation.metrics import METRIC_NAMES, classification_metrics, roc_points
from synthdx.evaluation.privacy import (
    dcr_stats,
    exact_duplicates,
    membership_inference_auc,
    privacy_summary,
)
from synthdx.evaluation.stats import mean_std, paired_test

__all__ = [
    "METRIC_NAMES", "add_sensitive_columns", "classification_metrics", "column_fidelity",
    "correlation_difference", "correlation_matrices", "dcr_stats",
    "demographic_parity_difference", "detectability", "equal_opportunity_difference",
    "exact_duplicates", "fidelity_summary", "group_metrics", "mean_std",
    "membership_inference_auc", "paired_test", "privacy_summary", "roc_points",
]
