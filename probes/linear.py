"""Frozen ridge probes scored on a held-out split.

Pure numpy helpers shared by ``scripts/run_probes.py`` and its tests. Every
codec is probed with the identical procedure: the same group-wise train/test
split, the same feature standardization (statistics from the train rows only)
and the same fixed ridge penalty. No randomness is involved.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Locked L2 penalty on standardized features (identical for every codec).
PROBE_RIDGE_ALPHA = 1.0
# Fraction of fixture episodes (not states) assigned to the probe train split.
PROBE_TRAIN_FRACTION = 0.5

_STD_FLOOR = 1e-8


def split_by_group(
    groups: Sequence[int],
    *,
    train_fraction: float = PROBE_TRAIN_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/test row masks that never split a group.

    ``groups`` holds one group id (episode index) per row. Distinct groups are
    ordered by first appearance; the leading ``train_fraction`` of them are
    train, the rest test. Prefixes of one episode are strongly correlated, so
    splitting by row would leak the test targets into the train fit.
    """
    group_array = np.asarray(groups, dtype=np.int64)
    ordered: list[int] = list(dict.fromkeys(group_array.tolist()))
    if len(ordered) < 2:
        raise ValueError("probe split needs at least 2 distinct groups")
    n_train = min(len(ordered) - 1, max(1, int(len(ordered) * train_fraction)))
    train_groups = set(ordered[:n_train])
    train_mask = np.asarray([g in train_groups for g in group_array.tolist()], dtype=bool)
    return train_mask, ~train_mask


def fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = PROBE_RIDGE_ALPHA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a ridge map on standardized features with an unpenalized intercept.

    Returns ``(mean, scale, weight, intercept)`` for :func:`predict_ridge`.
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("features must be (N, D) and target (N,)")
    if x.shape[0] < 2:
        raise ValueError("ridge fit needs at least 2 rows")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < _STD_FLOOR, 1.0, scale)
    z = (x - mean) / scale
    intercept = float(y.mean())
    gram = z.T @ z + alpha * np.eye(z.shape[1])
    weight = np.linalg.solve(gram, z.T @ (y - intercept))
    return mean, scale, weight, intercept


def predict_ridge(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    weight: np.ndarray,
    intercept: float,
) -> np.ndarray:
    z = (np.asarray(features, dtype=np.float64) - mean) / scale
    return z @ weight + intercept


def r2_score(target: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination; 0.0 when the target is constant."""
    y = np.asarray(target, dtype=np.float64)
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0.0:
        return 0.0
    return float(1.0 - residual / total)


def heldout_ridge_r2(
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    test_target: np.ndarray,
    *,
    alpha: float = PROBE_RIDGE_ALPHA,
) -> float:
    """Fit ridge on the train rows and return :math:`R^2` on the test rows."""
    if np.asarray(test_target).shape[0] < 2:
        raise ValueError("held-out R^2 needs at least 2 test rows")
    mean, scale, weight, intercept = fit_ridge(train_features, train_target, alpha=alpha)
    predicted = predict_ridge(test_features, mean, scale, weight, intercept)
    return r2_score(test_target, predicted)
