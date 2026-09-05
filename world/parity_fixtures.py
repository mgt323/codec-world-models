"""State fixtures for parity / diagnostics sampling (simulator-side).

Lives in ``world/`` so ``obs_codecs`` never imports ``LatentState`` / ``State``
(PROGRAM_SPEC §4.1: obs_codecs ↛ LatentState).
"""

from __future__ import annotations

from world.schema import (
    CausalRegime,
    Intervention,
    InterventionTarget,
    LatentState,
    NoiseBin,
    SensorSource,
    State,
)


def sample_parity_states() -> list[State]:
    """Deterministic batch covering regimes, all noise bins, and partial obs.

    Coverage:
    - all 4 ``CausalRegime`` values
    - ``NoiseBin.LOW``, ``MEDIUM``, and ``HIGH``
    - both, a_only, b_only, and neither (``SENSOR_NONE``)
    """
    states: list[State] = []
    t = 0
    for regime in CausalRegime:
        for noise in (NoiseBin.LOW, NoiseBin.MEDIUM, NoiseBin.HIGH):
            for a_obs, b_obs in (
                (True, True),
                (False, True),
                (True, False),
                (False, False),
            ):
                source = _source_for_masks(a_obs, b_obs)
                states.append(
                    State(
                        t=t,
                        a=0.41,
                        b=0.38,
                        a_observed=a_obs,
                        b_observed=b_obs,
                        noise_bin=noise,
                        n_samples=3,
                        source=source,
                        latent=LatentState(hidden_c=0.5, regime=regime),
                        active_intervention=None,
                    )
                )
                t += 1
    return states


def sample_diverse_parity_states(target_n: int = 500) -> list[State]:
    """Build a diverse ~``target_n`` State batch for large parity audits.

    Guaranteed coverage (within the first covering block, then padded):
    - all 4 ``CausalRegime`` values
    - ``NoiseBin.LOW``, ``MEDIUM``, and ``HIGH``
    - partial-obs: a_only, b_only, both, neither
    - ``n_samples=1`` edge cases
    - ``active_intervention`` set and ``None``
    """
    value_grid = (
        (0.0, 0.0),
        (0.05, 0.95),
        (0.1, 0.2),
        (0.3, 0.3),
        (0.41, 0.38),
        (0.5, 0.5),
        (0.66, 0.25),
        (0.75, 0.8),
        (0.99, 0.01),
        (1.0, 1.0),
        (-0.1, 1.2),
        (0.22, 0.78),
    )
    hidden_cs = (0.0, 0.25, 0.5, 0.75, 1.0)
    n_samples_choices = (1, 3, 5)
    partial_masks = (
        (True, True),  # both
        (True, False),  # a_only
        (False, True),  # b_only
        (False, False),  # neither
    )
    intervention_plans: tuple[Intervention | None, ...] = (
        None,
        Intervention(target=InterventionTarget.A, value=0.9, timestep=0),
        Intervention(target=InterventionTarget.B, value=0.1, timestep=0),
    )
    noises = (NoiseBin.LOW, NoiseBin.MEDIUM, NoiseBin.HIGH)
    regimes = tuple(CausalRegime)

    # Round-robin over required axes so early truncation still covers all factors.
    combos: list[tuple[object, ...]] = []
    for regime in regimes:
        for noise in noises:
            for a_obs, b_obs in partial_masks:
                for n_samples in n_samples_choices:
                    for intervention in intervention_plans:
                        combos.append(
                            (regime, noise, a_obs, b_obs, n_samples, intervention)
                        )

    states: list[State] = []
    t = 0
    vi = 0
    hi = 0
    while len(states) < target_n:
        for regime, noise, a_obs, b_obs, n_samples, intervention in combos:
            if len(states) >= target_n:
                break
            a_val, b_val = value_grid[vi % len(value_grid)]
            hidden_c = hidden_cs[hi % len(hidden_cs)]
            vi += 1
            hi += 1
            states.append(
                State(
                    t=t,
                    a=float(a_val),
                    b=float(b_val),
                    a_observed=bool(a_obs),
                    b_observed=bool(b_obs),
                    noise_bin=noise,  # type: ignore[arg-type]
                    n_samples=int(n_samples),
                    source=_source_for_masks(bool(a_obs), bool(b_obs)),
                    latent=LatentState(
                        hidden_c=float(hidden_c),
                        regime=regime,  # type: ignore[arg-type]
                    ),
                    active_intervention=intervention,  # type: ignore[arg-type]
                )
            )
            t += 1
    return states


def _source_for_masks(a_observed: bool, b_observed: bool) -> SensorSource:
    if a_observed and b_observed:
        return SensorSource.SENSOR_BOTH
    if a_observed:
        return SensorSource.SENSOR_A
    if b_observed:
        return SensorSource.SENSOR_B
    return SensorSource.SENSOR_NONE
