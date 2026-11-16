"""Representation similarity is 1 for a matrix compared with itself."""

from __future__ import annotations

import numpy as np
import pytest

from probes.rsa import linear_cka, orthogonal_procrustes, svcca_mean_correlation


def test_similarity_is_one_for_identical_activations() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(32, 8))
    assert linear_cka(features, features) == pytest.approx(1.0, abs=1e-6)
    assert svcca_mean_correlation(features, features, components=4) == pytest.approx(
        1.0, abs=1e-6
    )
    assert orthogonal_procrustes(features, features) == pytest.approx(1.0, abs=1e-6)
