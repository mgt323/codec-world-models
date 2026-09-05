"""World v0 simulator: deterministic 1D causal toy episodes.

Implements EXPERIMENT_PLAN.md §2 / §2.1 and PROGRAM_SPEC Simulator +
Interventions (option B). Same ``(seed, n_steps, difficulty, interventions)``
⇒ identical ``list[State]`` (INVARIANT I3).

NoiseBin cutoffs (locked for v0; map ``difficulty.noise_scale``):
- ``noise_scale < 0.1`` → LOW
- ``0.1 <= noise_scale < 0.3`` → MEDIUM
- ``noise_scale >= 0.3`` → HIGH

``n_samples`` is fixed at 3 per episode for v0 (deterministic; not derived
from regime or ``hidden_c``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from world.schema import (
    CausalRegime,
    Difficulty,
    Intervention,
    InterventionTarget,
    LatentState,
    NoiseBin,
    SensorSource,
    State,
)

_N_SAMPLES_V0 = 3
_REGIMES: tuple[CausalRegime, ...] = tuple(CausalRegime)


def simulate(
    seed: int,
    n_steps: int,
    difficulty: Difficulty = Difficulty(),
    interventions: Sequence[Intervention] | None = None,
) -> list[State]:
    """Simulate one episode. Does not call Codec or Model."""
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    if difficulty.regime_switch_rate != 0.0:
        raise NotImplementedError(
            "World v0 requires difficulty.regime_switch_rate == 0 "
            f"(fixed regime per episode); got {difficulty.regime_switch_rate}"
        )

    schedule = _validate_interventions(interventions, n_steps=n_steps)
    rng = np.random.Generator(np.random.PCG64(seed))

    regime = _REGIMES[int(rng.integers(0, len(_REGIMES)))]
    hidden_c = float(rng.uniform(0.0, 1.0))
    noise_bin = _noise_bin_from_scale(difficulty.noise_scale)
    n_samples = _N_SAMPLES_V0
    scale = float(difficulty.noise_scale)
    confound = float(difficulty.confounding_strength)

    latent = LatentState(
        hidden_c=hidden_c,
        regime=regime,
        oracle_regime_posterior=None,
    )

    states: list[State] = []
    for t in range(n_steps):
        # Fixed draw order per step (intervention-independent) for I3 / causal GT.
        noise_a = float(rng.normal(0.0, scale))
        noise_b = float(rng.normal(0.0, scale))
        shared = float(rng.normal(0.0, scale))

        a, b = _generate_observables(
            regime=regime,
            hidden_c=hidden_c,
            noise_a=noise_a,
            noise_b=noise_b,
            shared=shared,
            confounding_strength=confound,
        )

        active = schedule.get(t)
        if active is not None:
            a, b = _apply_intervention(
                regime=regime,
                a=a,
                b=b,
                intervention=active,
                noise_a=noise_a,
                noise_b=noise_b,
                shared=shared,
                hidden_c=hidden_c,
                confounding_strength=confound,
            )

        a_observed, b_observed = _sample_partial_obs_masks(
            rng, partial_obs_rate=float(difficulty.partial_obs_rate)
        )
        source = _source_for_masks(a_observed, b_observed)

        states.append(
            State(
                t=t,
                a=a,
                b=b,
                a_observed=a_observed,
                b_observed=b_observed,
                noise_bin=noise_bin,
                n_samples=n_samples,
                source=source,
                latent=latent,
                active_intervention=active,
            )
        )
    return states


def _noise_bin_from_scale(noise_scale: float) -> NoiseBin:
    if noise_scale < 0.1:
        return NoiseBin.LOW
    if noise_scale < 0.3:
        return NoiseBin.MEDIUM
    return NoiseBin.HIGH


def _validate_interventions(
    interventions: Sequence[Intervention] | None,
    *,
    n_steps: int,
) -> dict[int, Intervention]:
    if not interventions:
        return {}
    schedule: dict[int, Intervention] = {}
    for intervention in interventions:
        if intervention.timestep is None:
            raise ValueError(
                "Scheduled interventions must set timestep (got timestep=None)"
            )
        t = int(intervention.timestep)
        if t < 0 or t >= n_steps:
            raise ValueError(
                f"intervention.timestep={t} out of range for n_steps={n_steps}"
            )
        if t in schedule:
            raise ValueError(
                f"At most one intervention per timestep; duplicate at t={t}"
            )
        schedule[t] = intervention
    return schedule


def _causal_f(x: float) -> float:
    return float(x)


def _generate_observables(
    *,
    regime: CausalRegime,
    hidden_c: float,
    noise_a: float,
    noise_b: float,
    shared: float,
    confounding_strength: float,
) -> tuple[float, float]:
    if regime is CausalRegime.COMMON_CAUSE:
        return hidden_c + noise_a, hidden_c + noise_b
    if regime is CausalRegime.A_CAUSES_B:
        a = noise_a
        return a, _causal_f(a) + noise_b
    if regime is CausalRegime.B_CAUSES_A:
        b = noise_b
        return _causal_f(b) + noise_a, b
    if regime is CausalRegime.SPURIOUS:
        # Weak shared component; kept small so it does not mimic COMMON_CAUSE.
        c = 0.25 * confounding_strength
        return noise_a + c * shared, noise_b + c * shared
    raise ValueError(f"Unknown regime: {regime!r}")


def _apply_intervention(
    *,
    regime: CausalRegime,
    a: float,
    b: float,
    intervention: Intervention,
    noise_a: float,
    noise_b: float,
    shared: float,
    hidden_c: float,
    confounding_strength: float,
) -> tuple[float, float]:
    """Apply do(target=value) with regime-correct causal responses."""
    target = intervention.target
    value = float(intervention.value)

    if regime is CausalRegime.COMMON_CAUSE:
        if target is InterventionTarget.A:
            return value, b
        return a, value

    if regime is CausalRegime.A_CAUSES_B:
        if target is InterventionTarget.A:
            return value, _causal_f(value) + noise_b
        return a, value

    if regime is CausalRegime.B_CAUSES_A:
        if target is InterventionTarget.B:
            return _causal_f(value) + noise_a, value
        return value, b

    if regime is CausalRegime.SPURIOUS:
        if target is InterventionTarget.A:
            return value, b
        return a, value

    raise ValueError(f"Unknown regime: {regime!r}")


def _sample_partial_obs_masks(
    rng: np.random.Generator,
    *,
    partial_obs_rate: float,
) -> tuple[bool, bool]:
    """Episode policy: at least one of A/B observed (never SENSOR_NONE)."""
    if partial_obs_rate < 0.0 or partial_obs_rate > 1.0:
        raise ValueError(
            f"partial_obs_rate must be in [0, 1], got {partial_obs_rate}"
        )
    if partial_obs_rate == 0.0 or float(rng.random()) >= partial_obs_rate:
        return True, True
    # Drop exactly one sensor.
    if float(rng.random()) < 0.5:
        return True, False
    return False, True


def _source_for_masks(a_observed: bool, b_observed: bool) -> SensorSource:
    if a_observed and b_observed:
        return SensorSource.SENSOR_BOTH
    if a_observed:
        return SensorSource.SENSOR_A
    if b_observed:
        return SensorSource.SENSOR_B
    # Unreachable under simulate partial-obs policy; kept for completeness.
    return SensorSource.SENSOR_NONE
