"""Statistical regime checks: mean Pearson corr(A, B) under World v0 dynamics.

Replaces eyeballing single trajectories with episode-aggregated correlation
structure at ``Difficulty(noise_scale=0.15, partial_obs_rate=0.0)``.

Notes on COMMON_CAUSE:
``hidden_c`` is fixed within an episode, so *within-episode* corr(A, B) is
near 0 (only independent noises). Shared-cause structure appears in
*between-episode* correlation of episode means (both track ``hidden_c``)
and in pooled (A, B) across episodes. Directed regimes are checked via
within-episode correlations (theory ≈ 1/√2 ≈ 0.707 for ``f(x)=x``).
"""

from __future__ import annotations

import numpy as np
import pytest

from world.schema import CausalRegime, Difficulty
from world.simulate import simulate

_DIFF = Difficulty(noise_scale=0.15, partial_obs_rate=0.0)
_N_EPISODES = 100
_N_STEPS = 50
# Theory for A→B / B→A with f(x)=x and equal noise: corr = 1/sqrt(2) ≈ 0.707.
_DIRECTED_MEAN_MIN = 0.5
# SPURIOUS: mean within-episode r near 0 (allow small sampling noise).
_SPURIOUS_ABS_MEAN_MAX = 0.1
# COMMON_CAUSE: between-episode mean(A) vs mean(B) should be near 1.
_COMMON_BETWEEN_MIN = 0.9


def _collect_episodes(
    regime: CausalRegime,
    *,
    n_episodes: int = _N_EPISODES,
    n_steps: int = _N_STEPS,
    difficulty: Difficulty = _DIFF,
    start_seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``n_episodes`` of (A series, B series) for ``regime``."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    seed = start_seed
    while len(out) < n_episodes:
        states = simulate(seed, n_steps, difficulty=difficulty, interventions=None)
        if states[0].latent.regime is regime:
            a = np.asarray([s.a for s in states], dtype=np.float64)
            b = np.asarray([s.b for s in states], dtype=np.float64)
            out.append((a, b))
        seed += 1
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _within_episode_corrs(
    episodes: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    return np.asarray([_pearson(a, b) for a, b in episodes], dtype=np.float64)


def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return mean, se


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile CI for the mean via bootstrap (explicit local RNG)."""
    rng = rng or np.random.Generator(np.random.PCG64(0))
    values = values[np.isfinite(values)]
    n = len(values)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = float(np.mean(sample))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


@pytest.mark.parametrize(
    "regime",
    [CausalRegime.A_CAUSES_B, CausalRegime.B_CAUSES_A],
)
def test_directed_regimes_have_strong_within_episode_correlation(
    regime: CausalRegime,
) -> None:
    episodes = _collect_episodes(regime)
    corrs = _within_episode_corrs(episodes)
    mean, se = _mean_and_se(corrs)
    lo, hi = _bootstrap_mean_ci(corrs)

    assert mean > _DIRECTED_MEAN_MIN, (
        f"{regime.value}: mean within-episode corr={mean:.3f} (se={se:.3f}, "
        f"95% bootstrap CI [{lo:.3f}, {hi:.3f}]) expected > {_DIRECTED_MEAN_MIN}"
    )
    assert lo > 0.3, (
        f"{regime.value}: lower 95% CI on mean corr is {lo:.3f}, expected > 0.3"
    )


def test_spurious_within_episode_correlation_near_zero() -> None:
    episodes = _collect_episodes(CausalRegime.SPURIOUS)
    corrs = _within_episode_corrs(episodes)
    mean, se = _mean_and_se(corrs)
    lo, hi = _bootstrap_mean_ci(corrs)

    assert abs(mean) < _SPURIOUS_ABS_MEAN_MAX, (
        f"SPURIOUS mean within-episode corr={mean:.3f} (se={se:.3f}), "
        f"expected |mean| < {_SPURIOUS_ABS_MEAN_MAX}"
    )
    # CI should cover 0 (not significantly different from 0 at ~95%).
    assert lo <= 0.0 <= hi, (
        f"SPURIOUS 95% bootstrap CI for mean corr [{lo:.3f}, {hi:.3f}] "
        f"should include 0"
    )


def test_common_cause_positive_via_shared_hidden_c() -> None:
    """Between-episode means track shared hidden_c → strong positive corr."""
    episodes = _collect_episodes(CausalRegime.COMMON_CAUSE)
    mean_a = np.asarray([a.mean() for a, _ in episodes], dtype=np.float64)
    mean_b = np.asarray([b.mean() for _, b in episodes], dtype=np.float64)
    between = _pearson(mean_a, mean_b)

    # Sanity: within-episode mean stays near 0 (C fixed inside episode).
    within = _within_episode_corrs(episodes)
    within_mean, within_se = _mean_and_se(within)
    assert abs(within_mean) < 0.1, (
        f"COMMON_CAUSE unexpected within-episode mean corr={within_mean:.3f} "
        f"(se={within_se:.3f}); hidden_c should be fixed per episode"
    )

    assert between > _COMMON_BETWEEN_MIN, (
        f"COMMON_CAUSE between-episode corr(mean A, mean B)={between:.3f}, "
        f"expected > {_COMMON_BETWEEN_MIN} (shared hidden_c)"
    )

    pooled_a = np.concatenate([a for a, _ in episodes])
    pooled_b = np.concatenate([b for _, b in episodes])
    pooled = _pearson(pooled_a, pooled_b)
    assert pooled > 0.5, (
        f"COMMON_CAUSE pooled corr={pooled:.3f}, expected > 0.5"
    )
