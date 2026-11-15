"""Linear CKA, SVCCA, and orthogonal Procrustes on frozen activations.

Activations are centered. SVCCA keeps the leading singular components, then
reports the mean canonical correlation. Procrustes similarity is the Frobenius
inner product after an orthogonal map and unit-Frobenius normalization.
"""

from __future__ import annotations

import numpy as np


def _center(features: np.ndarray) -> np.ndarray:
    return features - features.mean(axis=0, keepdims=True)


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear CKA in ``[0, 1]`` for identical centered matrices."""
    x_centered = _center(np.asarray(left, dtype=np.float64))
    y_centered = _center(np.asarray(right, dtype=np.float64))
    cross = np.linalg.norm(x_centered.T @ y_centered, ord="fro") ** 2
    left_norm = np.linalg.norm(x_centered.T @ x_centered, ord="fro")
    right_norm = np.linalg.norm(y_centered.T @ y_centered, ord="fro")
    return float(cross / (left_norm * right_norm + 1e-12))


def _svd_reduce(features: np.ndarray, components: int) -> np.ndarray:
    centered = _center(np.asarray(features, dtype=np.float64))
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    keep = min(components, vt.shape[0], centered.shape[0])
    return centered @ vt[:keep].T


def svcca_mean_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    components: int = 20,
) -> float:
    reduced_left = _svd_reduce(left, components)
    reduced_right = _svd_reduce(right, components)
    q_left, _r_left = np.linalg.qr(reduced_left)
    q_right, _r_right = np.linalg.qr(reduced_right)
    _u, singular, _vt = np.linalg.svd(q_left.T @ q_right, full_matrices=False)
    return float(np.mean(singular))


def orthogonal_procrustes(left: np.ndarray, right: np.ndarray) -> float:
    """Similarity of ``left`` after the orthogonal map that best matches ``right``."""
    x_centered = _center(np.asarray(left, dtype=np.float64))
    y_centered = _center(np.asarray(right, dtype=np.float64))
    x_centered = x_centered / (np.linalg.norm(x_centered) + 1e-12)
    y_centered = y_centered / (np.linalg.norm(y_centered) + 1e-12)
    u_mat, _s, vt = np.linalg.svd(x_centered.T @ y_centered, full_matrices=False)
    mapped = x_centered @ (u_mat @ vt)
    return float(np.sum(mapped * y_centered))
