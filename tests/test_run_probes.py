"""Held-out ridge probes and prefix fixture for scripts/run_probes.py."""

from __future__ import annotations

import numpy as np
import pytest

from probes.linear import (
    PROBE_RIDGE_ALPHA,
    fit_ridge,
    heldout_ridge_r2,
    predict_ridge,
    split_by_group,
)
from scripts.run_probes import _flatten_states, _states_from_raw
from world.simulate import DEFAULT_DIFFICULTY_V0, simulate


def _episodes(n: int) -> list[tuple[int, list]]:
    return [(100 + i, simulate(100 + i, 30, difficulty=DEFAULT_DIFFICULTY_V0)) for i in range(n)]


def test_flatten_states_builds_prefixes_with_original_t_and_next_b() -> None:
    episodes = _episodes(2)
    samples = _flatten_states(episodes, n_states=60)
    assert len(samples) == 60
    for sample in samples[:30]:
        states = episodes[0][1]
        assert len(sample.prefix) == sample.t + 1
        assert [s.t for s in sample.prefix] == list(range(sample.t + 1))
        assert sample.prefix[-1] is states[sample.t]
        assert sample.hidden_c == states[sample.t].latent.hidden_c
        if sample.t < 29:
            assert sample.next_b == states[sample.t + 1].b
        else:
            assert sample.next_b is None
    assert {s.episode_index for s in samples} == {0, 1}


def test_flatten_states_respects_n_states_cap() -> None:
    assert len(_flatten_states(_episodes(2), n_states=45)) == 45


def test_states_from_raw_reads_ceil_n_states_over_30_episodes(tmp_path) -> None:
    # No val.h5 -> deterministic simulator fallback with ceil(n / 30) episodes.
    assert len(_states_from_raw(tmp_path, 61)) == 3
    assert len(_states_from_raw(tmp_path, 60)) == 2
    first = _states_from_raw(tmp_path, 30)[0]
    again = _states_from_raw(tmp_path, 30)[0]
    assert first[0] == again[0] and first[1] == again[1]


def test_split_by_group_never_splits_a_group() -> None:
    groups = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5
    train, test = split_by_group(groups)
    assert train.sum() == 10 and test.sum() == 10
    assert set(np.asarray(groups)[train]).isdisjoint(set(np.asarray(groups)[test]))
    with pytest.raises(ValueError):
        split_by_group([0, 0, 0])


def test_heldout_ridge_is_not_in_sample() -> None:
    rng = np.random.default_rng(0)
    # Pure-noise features/targets: in-sample lstsq would reach R^2 ~ 1 here.
    features = rng.normal(size=(60, 50))
    target = rng.normal(size=60)
    r2 = heldout_ridge_r2(features[:30], target[:30], features[30:], target[30:])
    assert r2 < 0.2


def test_heldout_ridge_recovers_linear_signal() -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(400, 6))
    target = features @ np.arange(1, 7, dtype=np.float64) + 3.0
    r2 = heldout_ridge_r2(features[:200], target[:200], features[200:], target[200:])
    assert r2 > 0.99


def test_ridge_penalty_shrinks_weights() -> None:
    rng = np.random.default_rng(2)
    features = rng.normal(size=(100, 5))
    target = features @ np.ones(5)
    _, _, weak, _ = fit_ridge(features, target, alpha=0.0)
    _, _, strong, _ = fit_ridge(features, target, alpha=1e4)
    assert np.linalg.norm(strong) < np.linalg.norm(weak)
    assert PROBE_RIDGE_ALPHA > 0
    mean, scale, weight, intercept = fit_ridge(features, target)
    assert predict_ridge(features, mean, scale, weight, intercept).shape == (100,)


def test_training_provenance_inherits_manifest_commit(tmp_path) -> None:
    import json

    from scripts.run_probes import _training_provenance

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "git_commit": "train123",
                "git_dirty": True,
                "budget_profile": "full_v0",
                "n_replications": 5,
            }
        ),
        encoding="utf-8",
    )
    prov = _training_provenance(tmp_path)
    assert prov["git_commit"] == "train123"
    assert prov["git_dirty"] is True
    assert prov["git_commit_source"] == "manifest"
    assert prov["n_replications"] == 5

    fallback = _training_provenance(tmp_path / "missing")
    assert fallback["git_commit_source"] == "current_head_no_manifest"
