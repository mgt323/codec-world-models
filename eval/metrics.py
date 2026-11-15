"""Metric reductions shared by the evaluator and the replication aggregator.

Quartiles use ``numpy.percentile(..., method="linear")`` (contract §5.1).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Present under every eval split. ``null`` means the task does not apply.
METRIC_KEYS: tuple[str, ...] = (
    "next_state_mse",
    "regime_accuracy",
    "intervention_accuracy",
    "calibration_ece",
    "calibration_brier",
    "counterfactual_divergence_mse",
    "partial_obs_next_state_mse",
    "ood_regime_accuracy",
    "transfer_aligned_next_state_mse",
    "next_state_parse_rate",
)

# ood_noise / ood_confound / ood_regime_switch are not unlocked (PROGRAM_SPEC).
OOD_REGIME_NOTE = "ood_regime axis is not unlocked; ood_regime_accuracy is null"


def empty_metrics() -> dict[str, float | None]:
    return {key: None for key in METRIC_KEYS}


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Multiclass ECE: confidence is the max class probability."""
    if len(labels) == 0:
        raise ValueError("labels must be non-empty")
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(labels))
    ece = 0.0
    for index in range(n_bins):
        lo, hi = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / total) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def brier_score(probabilities: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Mean squared error of the probability vector against a one-hot label."""
    if len(labels) == 0:
        raise ValueError("labels must be non-empty")
    target = np.eye(n_classes, dtype=np.float64)[labels]
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def median_iqr(values: list[float]) -> dict[str, Any]:
    """``median``, ``q1``, ``q3`` via NumPy's linear percentile (contract §5.1)."""
    if not values:
        raise ValueError("values must be non-empty")
    array = np.asarray(values, dtype=np.float64)
    q1, median, q3 = np.percentile(array, [25, 50, 75], method="linear")
    return {
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "values": [float(value) for value in values],
    }
