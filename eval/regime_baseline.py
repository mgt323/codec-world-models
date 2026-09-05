"""Eval-only regime baselines for World v0 difficulty calibration.

Calibration/harness only — not a model metric. common_cause vs
spurious is expected to be near-chance-separable from a single
episode by World v0 design (see identifiability note below); this
is a known ceiling, not a bug in this module or in simulate.py.

Identifiability note (test4.py / test5.py):
``hidden_c`` is drawn once per episode and held fixed, so within-episode
corr(A, B) and var(A)/var(B) cannot separate COMMON_CAUSE from SPURIOUS.
Across-episode corr of episode means does separate them, but that signal
is not available to a model (or this O-only heuristic) from a single
episode's O_{1:T}.

GUARDRAILS:
- Must NOT be imported from ``train/`` or ``obs_codecs/``.
- Never pass ``predict_regime_oracle`` output into any codec or training loss.
- Oracle/cheat helpers keep the ``oracle_`` / ``cheat_`` naming on purpose.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Mapping

import numpy as np

from world.schema import CausalRegime, Difficulty, State
from world.simulate import simulate

# Tuned from explore gaps: directed regimes ~0.7 corr; common/spurious ~0.01.
CORR_THRESHOLD = 0.3
RATIO_LOW_THRESHOLD = 0.75
RATIO_HIGH_THRESHOLD = 1.33

_REGIME_ORDER: tuple[CausalRegime, ...] = tuple(CausalRegime)

Episode = list[State]
ConfusionMatrix = dict[str, dict[str, int]]


def predict_regime_oracle(state: State) -> CausalRegime:
    """Cheat: return ``state.latent.regime`` directly (eval/harness only)."""
    return state.latent.regime


def cheat_predict_regime(state: State) -> CausalRegime:
    """Alias of ``predict_regime_oracle`` (explicit cheat_ prefix)."""
    return predict_regime_oracle(state)


def oracle_regime_accuracy(episodes: Sequence[Episode]) -> float:
    """Episode-level oracle accuracy; must be exactly 1.0 if harness is intact."""
    if not episodes:
        raise ValueError("episodes must be non-empty")
    correct = 0
    for episode in episodes:
        if not episode:
            raise ValueError("each episode must contain at least one State")
        gt = episode[0].latent.regime
        pred = predict_regime_oracle(episode[0])
        if pred is gt:
            correct += 1
    return correct / len(episodes)


def majority_regime_accuracy(episodes: Sequence[Episode]) -> float:
    """Predict the batch-majority regime label for every episode (ignores O and H)."""
    if not episodes:
        raise ValueError("episodes must be non-empty")
    labels = [ep[0].latent.regime for ep in episodes]
    majority = Counter(labels).most_common(1)[0][0]
    return sum(1 for lab in labels if lab is majority) / len(labels)


def predict_regime_heuristic(episode: Episode) -> CausalRegime:
    """Non-learned O-only regime guess from a single episode trajectory.

    Partial-obs policy for this heuristic: use only timesteps where
    ``a_observed`` and ``b_observed`` are both True (jointly observed).
    Episodes with fewer than 2 jointly observed timesteps raise ValueError.
    """
    if not episode:
        raise ValueError("episode must be non-empty")

    pairs = [
        (s.a, s.b)
        for s in episode
        if s.a_observed and s.b_observed
    ]
    if len(pairs) < 2:
        raise ValueError(
            "predict_regime_heuristic requires >= 2 jointly observed timesteps "
            f"(got {len(pairs)}); partial-obs-only episodes are out of scope"
        )

    a = np.asarray([p[0] for p in pairs], dtype=np.float64)
    b = np.asarray([p[1] for p in pairs], dtype=np.float64)
    corr = _pearson(a, b)

    if abs(corr) >= CORR_THRESHOLD:
        var_a = float(np.var(a, ddof=1)) if len(a) > 1 else 0.0
        var_b = float(np.var(b, ddof=1)) if len(b) > 1 else 0.0
        if var_b > 0.0:
            ratio = var_a / var_b
            if ratio < RATIO_LOW_THRESHOLD:
                return CausalRegime.A_CAUSES_B
            if ratio > RATIO_HIGH_THRESHOLD:
                return CausalRegime.B_CAUSES_A
        # Ambiguous: corr present but ratio near 1 → fall through to Step 2b.

    # Step 2b — World v0 identifiability ceiling (test4.py / test5.py):
    # within-episode corr≈0 and ratio≈1 for BOTH common_cause and spurious.
    # Always predict SPURIOUS so common_cause recall ≈ 0 in the confusion
    # matrix (visible ceiling). Do not "fix" without revisiting the design.
    return CausalRegime.SPURIOUS


def heuristic_regime_accuracy(episodes: Sequence[Episode]) -> float:
    """4-class accuracy of ``predict_regime_heuristic`` (episode-level)."""
    if not episodes:
        raise ValueError("episodes must be non-empty")
    correct = 0
    for episode in episodes:
        gt = episode[0].latent.regime
        if predict_regime_heuristic(episode) is gt:
            correct += 1
    return correct / len(episodes)


def heuristic_confusion_matrix(episodes: Sequence[Episode]) -> ConfusionMatrix:
    """Full 4×4 counts: true_regime -> predicted_regime -> count."""
    matrix: ConfusionMatrix = {
        true.value: {pred.value: 0 for pred in _REGIME_ORDER}
        for true in _REGIME_ORDER
    }
    for episode in episodes:
        if not episode:
            raise ValueError("each episode must contain at least one State")
        true = episode[0].latent.regime
        pred = predict_regime_heuristic(episode)
        matrix[true.value][pred.value] += 1
    return matrix


def difficulty_sweep(
    noise_scale_values: Iterable[float],
    confounding_strength_values: Iterable[float],
    *,
    n_episodes_per_cell: int = 100,
    n_steps: int = 30,
    start_seed: int = 0,
) -> list[dict[str, object]]:
    """Sweep difficulty cells; return rows with accuracies + confusion matrices.

    Does not reduce to a single accuracy — each cell includes the full
    heuristic confusion matrix (required for interpreting the World v0 ceiling).
    """
    rows: list[dict[str, object]] = []
    seed = start_seed
    for noise_scale in noise_scale_values:
        for confounding_strength in confounding_strength_values:
            difficulty = Difficulty(
                noise_scale=float(noise_scale),
                confounding_strength=float(confounding_strength),
                partial_obs_rate=0.0,
                regime_switch_rate=0.0,
            )
            episodes, seed = _collect_balanced_episodes(
                difficulty=difficulty,
                n_per_regime=n_episodes_per_cell // 4
                if n_episodes_per_cell >= 4
                else n_episodes_per_cell,
                n_steps=n_steps,
                start_seed=seed,
                target_total=n_episodes_per_cell,
            )
            row = {
                "noise_scale": float(noise_scale),
                "confounding_strength": float(confounding_strength),
                "n_episodes": len(episodes),
                "oracle_regime_accuracy": oracle_regime_accuracy(episodes),
                "majority_regime_accuracy": majority_regime_accuracy(episodes),
                "heuristic_regime_accuracy": heuristic_regime_accuracy(episodes),
                "heuristic_confusion_matrix": heuristic_confusion_matrix(episodes),
            }
            rows.append(row)
    return rows


def _collect_balanced_episodes(
    *,
    difficulty: Difficulty,
    n_per_regime: int,
    n_steps: int,
    start_seed: int,
    target_total: int,
) -> tuple[list[Episode], int]:
    """Collect roughly balanced regimes then trim/pad to ``target_total``."""
    buckets: dict[CausalRegime, list[Episode]] = {r: [] for r in _REGIME_ORDER}
    seed = start_seed
    # Cap search so pathological seeds cannot hang.
    max_seeds = max(10_000, target_total * 20)
    tried = 0
    while (
        any(len(v) < n_per_regime for v in buckets.values())
        and tried < max_seeds
    ):
        ep = simulate(seed, n_steps, difficulty=difficulty, interventions=None)
        regime = ep[0].latent.regime
        if len(buckets[regime]) < n_per_regime:
            buckets[regime].append(ep)
        seed += 1
        tried += 1

    episodes: list[Episode] = []
    for regime in _REGIME_ORDER:
        episodes.extend(buckets[regime])

    # If user asked for 100 with 4 regimes, prefer 25 each (exact balance).
    if len(episodes) > target_total:
        episodes = episodes[:target_total]
    while len(episodes) < target_total and tried < max_seeds:
        ep = simulate(seed, n_steps, difficulty=difficulty, interventions=None)
        episodes.append(ep)
        seed += 1
        tried += 1
    return episodes, seed


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def format_sweep_row(row: Mapping[str, object]) -> str:
    """Human-readable one-cell report (accuracy + confusion matrix)."""
    cm = row["heuristic_confusion_matrix"]
    assert isinstance(cm, dict)
    lines = [
        f"noise_scale={row['noise_scale']} "
        f"confounding_strength={row['confounding_strength']} "
        f"n={row['n_episodes']}",
        f"  oracle={row['oracle_regime_accuracy']:.3f} "
        f"majority={row['majority_regime_accuracy']:.3f} "
        f"heuristic={row['heuristic_regime_accuracy']:.3f}",
        "  confusion (true\\\\pred): "
        + " ".join(r.value for r in _REGIME_ORDER),
    ]
    for true in _REGIME_ORDER:
        counts = cm[true.value]
        assert isinstance(counts, dict)
        cell = " ".join(f"{counts[pred.value]:4d}" for pred in _REGIME_ORDER)
        lines.append(f"    {true.value:<14} {cell}")
    return "\n".join(lines)
