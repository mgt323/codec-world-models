"""Analytical next-``b`` reference predictors for the probe fixture (Tier B2).

These are *recording-only* diagnostics. They give the locked ``next_b_r2``
probe score context (what a trivial or an oracle predictor of
``state_{t+1}.b`` reaches on the same held-out rows) and never feed a codec,
a model, or a training loss.

* ``last_observed_b`` / ``prefix_mean_observed_b`` read only the sensor-masked
  values that every codec also encodes (``b`` where ``b_observed``).
* ``oracle_next_b_mean`` reads the simulator's latent ``(regime, hidden_c)``.
  It is an evaluation-side ceiling (I7), not a model input (I2, I8).

World v0 facts used by the oracle (see ``world/simulate.py``): every step
draws fresh zero-mean Gaussian noise, so given ``(regime, hidden_c)`` the
expectation of ``b_{t+1}`` is ``hidden_c`` under ``COMMON_CAUSE`` and ``0``
under the other three regimes (``confounding_strength == 0`` for the locked
v0 difficulty).
"""

from __future__ import annotations

from collections.abc import Sequence

from world.schema import CausalRegime, State


def last_observed_b(prefix: Sequence[State], *, fallback: float) -> float:
    """Most recent observed ``b`` in ``prefix`` (``fallback`` if none)."""
    for state in reversed(prefix):
        if state.b_observed:
            return float(state.b)
    return float(fallback)


def prefix_mean_observed_b(prefix: Sequence[State], *, fallback: float) -> float:
    """Mean of observed ``b`` over ``prefix`` (``fallback`` if none observed)."""
    observed = [float(state.b) for state in prefix if state.b_observed]
    if not observed:
        return float(fallback)
    return float(sum(observed) / len(observed))


def oracle_next_b_mean(regime: CausalRegime, hidden_c: float) -> float:
    """``E[b_{t+1} | regime, hidden_c]`` for World v0 (noise-free graph mean)."""
    if regime is CausalRegime.COMMON_CAUSE:
        return float(hidden_c)
    return 0.0
