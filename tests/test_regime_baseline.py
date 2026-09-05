"""Tests for eval.regime_baseline (oracle / majority / O-only heuristic)."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.regime_baseline import (
    CORR_THRESHOLD,
    difficulty_sweep,
    heuristic_confusion_matrix,
    heuristic_regime_accuracy,
    majority_regime_accuracy,
    oracle_regime_accuracy,
    predict_regime_heuristic,
    predict_regime_oracle,
)
from world.schema import CausalRegime, Difficulty
from world.simulate import simulate

_DIFF = Difficulty(noise_scale=0.15, partial_obs_rate=0.0, confounding_strength=0.0)
_ROOT = Path(__file__).resolve().parents[1]


def _collect_episodes(
    *,
    n_per_regime: int = 40,
    n_steps: int = 30,
    difficulty: Difficulty = _DIFF,
    start_seed: int = 0,
) -> list[list]:
    buckets: dict[CausalRegime, list] = {r: [] for r in CausalRegime}
    seed = start_seed
    while any(len(v) < n_per_regime for v in buckets.values()):
        ep = simulate(seed, n_steps, difficulty=difficulty, interventions=None)
        r = ep[0].latent.regime
        if len(buckets[r]) < n_per_regime:
            buckets[r].append(ep)
        seed += 1
    episodes: list[list] = []
    for r in CausalRegime:
        episodes.extend(buckets[r])
    return episodes


def test_oracle_regime_accuracy_is_exactly_one() -> None:
    episodes = _collect_episodes(n_per_regime=20)
    assert oracle_regime_accuracy(episodes) == 1.0
    for ep in episodes:
        assert predict_regime_oracle(ep[0]) is ep[0].latent.regime


def test_majority_regime_accuracy_near_quarter_on_balanced_batch() -> None:
    episodes = _collect_episodes(n_per_regime=50)  # 200 episodes, balanced
    acc = majority_regime_accuracy(episodes)
    assert 0.20 <= acc <= 0.30, f"majority accuracy {acc} outside [0.20, 0.30]"


def test_heuristic_common_cause_recall_near_zero() -> None:
    """Ceiling check: Step 2b always predicts SPURIOUS when corr≈0."""
    episodes = _collect_episodes(n_per_regime=40)
    cm = heuristic_confusion_matrix(episodes)
    cc_row = cm[CausalRegime.COMMON_CAUSE.value]
    n_cc = sum(cc_row.values())
    assert n_cc > 0
    recall = cc_row[CausalRegime.COMMON_CAUSE.value] / n_cc
    assert recall == 0.0, (
        f"common_cause recall={recall}; expected 0 under documented Step 2b fallback"
    )
    # Most common_cause mass should land on SPURIOUS.
    assert cc_row[CausalRegime.SPURIOUS.value] / n_cc >= 0.8


def test_heuristic_separates_directed_regimes() -> None:
    episodes = _collect_episodes(n_per_regime=40)
    cm = heuristic_confusion_matrix(episodes)
    for regime in (CausalRegime.A_CAUSES_B, CausalRegime.B_CAUSES_A):
        row = cm[regime.value]
        n = sum(row.values())
        recall = row[regime.value] / n
        assert recall >= 0.7, f"{regime.value} recall={recall} expected >= 0.7"


def test_heuristic_overall_accuracy_in_solvable_band() -> None:
    """Step 2b(a) maps both confused classes → SPURIOUS ⇒ ~3/4 correct (~0.75)."""
    episodes = _collect_episodes(n_per_regime=40)
    acc = heuristic_regime_accuracy(episodes)
    assert 0.55 <= acc <= 0.85, f"heuristic accuracy {acc} outside expected band"


def test_corr_threshold_matches_documented_default() -> None:
    assert CORR_THRESHOLD == 0.3


def test_train_package_absent_or_no_regime_baseline_import() -> None:
    """If train/ exists later, it must not import eval.regime_baseline."""
    train_dir = _ROOT / "train"
    if not train_dir.is_dir():
        pytest.skip("train/ package not present yet")
    offenders: list[str] = []
    for path in train_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "regime_baseline" in text or "eval.regime_baseline" in text:
            offenders.append(str(path.relative_to(_ROOT)))
    assert not offenders, f"train/ must not import regime_baseline: {offenders}"


def test_obs_codecs_do_not_import_regime_baseline() -> None:
    offenders: list[str] = []
    for path in (_ROOT / "obs_codecs").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "regime_baseline" in text or "eval.regime_baseline" in text:
            offenders.append(str(path.relative_to(_ROOT)))
    assert not offenders


def test_difficulty_sweep_smoke() -> None:
    rows = difficulty_sweep(
        noise_scale_values=[0.15],
        confounding_strength_values=[0.0],
        n_episodes_per_cell=40,
        n_steps=20,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["oracle_regime_accuracy"] == 1.0
    assert "heuristic_confusion_matrix" in row
    cm = row["heuristic_confusion_matrix"]
    assert set(cm.keys()) == {r.value for r in CausalRegime}
