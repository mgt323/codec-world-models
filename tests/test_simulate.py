"""Tests for world.simulate (I3, do-calculus, partial-obs policy)."""

from __future__ import annotations

import inspect
import pickle

import pytest

from world.schema import (
    CausalRegime,
    Difficulty,
    Intervention,
    InterventionTarget,
    LatentState,
    NoiseBin,
    SensorSource,
    State,
    observe,
)
from world.simulate import simulate


def _find_seed_for_regime(
    regime: CausalRegime,
    *,
    n_steps: int = 10,
    difficulty: Difficulty | None = None,
    max_tries: int = 10_000,
) -> int:
    diff = difficulty if difficulty is not None else Difficulty()
    for seed in range(max_tries):
        states = simulate(seed, n_steps, difficulty=diff, interventions=None)
        if states[0].latent.regime is regime:
            return seed
    raise RuntimeError(f"No seed found for regime={regime} within {max_tries} tries")


def test_simulate_determinism_same_inputs() -> None:
    diff = Difficulty(noise_scale=0.1, partial_obs_rate=0.3)
    interventions = (
        Intervention(target=InterventionTarget.A, value=0.7, timestep=3),
    )
    a = simulate(42, 12, difficulty=diff, interventions=interventions)
    b = simulate(42, 12, difficulty=diff, interventions=interventions)
    assert a == b
    assert pickle.dumps(a) == pickle.dumps(b)


@pytest.mark.parametrize("regime", list(CausalRegime))
def test_do_a_calculus_asymmetry_per_regime(regime: CausalRegime) -> None:
    """Same seed; do(A) at t=5 — prefix identical; causal responses match regime."""
    n_steps = 10
    t_do = 5
    x = 0.77
    diff = Difficulty(noise_scale=0.15, partial_obs_rate=0.0)
    seed = _find_seed_for_regime(regime, n_steps=n_steps, difficulty=diff)

    observational = simulate(seed, n_steps, difficulty=diff, interventions=None)
    interventional = simulate(
        seed,
        n_steps,
        difficulty=diff,
        interventions=(
            Intervention(target=InterventionTarget.A, value=x, timestep=t_do),
        ),
    )

    assert observational[0].latent.regime is regime
    for t in range(t_do):
        assert observational[t] == interventional[t]

    obs_t = observational[t_do]
    int_t = interventional[t_do]
    assert int_t.a == x
    assert int_t.active_intervention is not None
    assert int_t.active_intervention.target is InterventionTarget.A
    assert observational[t_do].active_intervention is None

    if regime is CausalRegime.COMMON_CAUSE:
        assert int_t.b == obs_t.b
    elif regime is CausalRegime.A_CAUSES_B:
        # B recomputed from intervened A: B = f(A) + noise_b = x + (obs_b - f(obs_a))
        expected_b = x + (obs_t.b - obs_t.a)
        assert int_t.b == pytest.approx(expected_b)
        assert int_t.b != obs_t.b or obs_t.a == x
    elif regime is CausalRegime.B_CAUSES_A:
        # A is downstream of B; do(A) does not change B.
        assert int_t.b == obs_t.b
    elif regime is CausalRegime.SPURIOUS:
        assert int_t.b == obs_t.b


@pytest.mark.parametrize("regime", list(CausalRegime))
def test_do_b_calculus_asymmetry_per_regime(regime: CausalRegime) -> None:
    n_steps = 10
    t_do = 5
    x = 0.33
    diff = Difficulty(noise_scale=0.15, partial_obs_rate=0.0)
    seed = _find_seed_for_regime(regime, n_steps=n_steps, difficulty=diff)

    observational = simulate(seed, n_steps, difficulty=diff, interventions=None)
    interventional = simulate(
        seed,
        n_steps,
        difficulty=diff,
        interventions=(
            Intervention(target=InterventionTarget.B, value=x, timestep=t_do),
        ),
    )

    for t in range(t_do):
        assert observational[t] == interventional[t]

    obs_t = observational[t_do]
    int_t = interventional[t_do]
    assert int_t.b == x

    if regime is CausalRegime.COMMON_CAUSE:
        assert int_t.a == obs_t.a
    elif regime is CausalRegime.B_CAUSES_A:
        expected_a = x + (obs_t.a - obs_t.b)
        assert int_t.a == pytest.approx(expected_a)
    elif regime is CausalRegime.A_CAUSES_B:
        assert int_t.a == obs_t.a
    elif regime is CausalRegime.SPURIOUS:
        assert int_t.a == obs_t.a


def test_partial_obs_never_both_unobserved() -> None:
    diff = Difficulty(noise_scale=0.1, partial_obs_rate=0.9)
    for seed in range(50):
        states = simulate(seed, 40, difficulty=diff, interventions=None)
        for state in states:
            assert state.a_observed or state.b_observed
            assert state.source is not SensorSource.SENSOR_NONE


def test_regime_switch_rate_nonzero_raises() -> None:
    with pytest.raises(NotImplementedError, match="regime_switch_rate"):
        simulate(
            0,
            5,
            difficulty=Difficulty(regime_switch_rate=0.1),
            interventions=None,
        )


def test_duplicate_intervention_timestep_raises() -> None:
    with pytest.raises(ValueError, match="At most one intervention"):
        simulate(
            0,
            8,
            interventions=(
                Intervention(target=InterventionTarget.A, value=0.1, timestep=2),
                Intervention(target=InterventionTarget.B, value=0.2, timestep=2),
            ),
        )


def test_simulate_module_does_not_import_obs_codecs() -> None:
    import world.simulate as sim

    source = inspect.getsource(sim)
    assert "obs_codecs" not in source
    assert "encode_" not in source


def test_observe_strips_active_intervention() -> None:
    states = simulate(
        7,
        6,
        interventions=(
            Intervention(target=InterventionTarget.B, value=0.5, timestep=2),
        ),
    )
    assert states[2].active_intervention is not None
    obs = observe(states[2])
    assert not hasattr(obs, "active_intervention")


def test_observe_byte_identical_across_regimes_and_interventions_from_simulate_template() -> None:
    """Hold O-fields fixed from a simulate State; vary H and do() — Observation bytes match.

    Extends ``test_observe_strips_latent_and_intervention`` / information-contract
    gates to values taken from a real ``simulate`` timestep (not hand-picked floats).
    States are rebuilt manually so all 4 regimes × intervened/non-intervened
    share identical A/B/noise_bin/n_samples/source/masks.
    """
    template = simulate(
        11,
        8,
        difficulty=Difficulty(noise_scale=0.12, partial_obs_rate=0.0),
        interventions=(
            Intervention(target=InterventionTarget.A, value=0.55, timestep=4),
        ),
    )[4]
    assert template.active_intervention is not None

    interventions = (
        None,
        Intervention(target=InterventionTarget.A, value=0.91, timestep=4),
        Intervention(target=InterventionTarget.B, value=0.09, timestep=4),
    )
    payloads: list[bytes] = []
    observations = []
    for regime in CausalRegime:
        for active in interventions:
            state = State(
                t=template.t,
                a=template.a,
                b=template.b,
                a_observed=template.a_observed,
                b_observed=template.b_observed,
                noise_bin=template.noise_bin,
                n_samples=template.n_samples,
                source=template.source,
                latent=LatentState(
                    hidden_c=0.0 if regime is CausalRegime.SPURIOUS else 0.75,
                    regime=regime,
                    oracle_regime_posterior=None,
                ),
                active_intervention=active,
            )
            # Observable channels held constant by construction.
            assert state.a == template.a
            assert state.b == template.b
            assert state.noise_bin is template.noise_bin
            assert state.n_samples == template.n_samples
            assert state.source is template.source
            obs = observe(state)
            observations.append(obs)
            payloads.append(pickle.dumps(obs, protocol=pickle.HIGHEST_PROTOCOL))

    assert len(payloads) == 4 * len(interventions)
    assert len(set(payloads)) == 1, (
        "observe() must be byte-identical across all regimes and "
        "active_intervention settings when A/B/noise_bin/n_samples/source match"
    )
    assert all(o == observations[0] for o in observations)
    assert observations[0].noise_bin is NoiseBin.MEDIUM  # noise_scale=0.12
    assert not hasattr(observations[0], "active_intervention")


def test_oracle_regime_posterior_is_none() -> None:
    states = simulate(1, 3)
    assert all(s.latent.oracle_regime_posterior is None for s in states)


def test_hidden_c_fixed_within_episode() -> None:
    states = simulate(99, 15)
    assert len({s.latent.hidden_c for s in states}) == 1
    assert len({s.latent.regime for s in states}) == 1
