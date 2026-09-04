"""Unit tests for world.schema (PROGRAM_SPEC type layering)."""

from __future__ import annotations

import pytest

from world.schema import (
    CausalRegime,
    FactRecord,
    Intervention,
    InterventionTarget,
    LatentState,
    NoiseBin,
    Observation,
    SensorSource,
    State,
    facts_from_observation,
    observation_field_names,
    observe,
    quantize_value,
    schema_checksum,
)


def _make_state(
    *,
    hidden_c: float = 0.5,
    regime: CausalRegime = CausalRegime.COMMON_CAUSE,
    oracle: dict[str, float] | None = None,
    a: float = 0.41,
    b: float = 0.38,
    a_observed: bool = True,
    b_observed: bool = True,
) -> State:
    return State(
        t=0,
        a=a,
        b=b,
        a_observed=a_observed,
        b_observed=b_observed,
        noise_bin=NoiseBin.HIGH,
        n_samples=3,
        source=SensorSource.SENSOR_BOTH,
        latent=LatentState(
            hidden_c=hidden_c,
            regime=regime,
            oracle_regime_posterior=oracle,
        ),
        active_intervention=Intervention(
            target=InterventionTarget.A,
            value=0.9,
            timestep=0,
        ),
    )


def test_observe_strips_latent_and_intervention() -> None:
    state = _make_state(
        hidden_c=0.99,
        regime=CausalRegime.A_CAUSES_B,
        oracle={"a_causes_b": 0.9, "common_cause": 0.1},
    )
    obs = observe(state)

    assert isinstance(obs, Observation)
    assert not hasattr(obs, "latent")
    assert not hasattr(obs, "regime")
    assert not hasattr(obs, "hidden_c")
    assert not hasattr(obs, "oracle_regime_posterior")
    assert not hasattr(obs, "active_intervention")
    assert "regime" not in observation_field_names()
    assert obs.a_observed is True
    assert obs.a_value == pytest.approx(0.41)
    assert obs.b_value == pytest.approx(0.38)
    assert obs.noise_bin is NoiseBin.HIGH
    assert obs.n_samples == 3
    assert obs.source is SensorSource.SENSOR_BOTH


def test_same_observation_when_only_latent_differs() -> None:
    """I1 / leakage gate: encodings will see identical O if only H changes."""
    s1 = _make_state(hidden_c=0.1, regime=CausalRegime.COMMON_CAUSE)
    s2 = _make_state(
        hidden_c=0.9,
        regime=CausalRegime.B_CAUSES_A,
        oracle={"b_causes_a": 1.0},
    )
    assert observe(s1) == observe(s2)


def test_partial_observation_masks_values() -> None:
    state = _make_state(a_observed=True, b_observed=False, a=0.2, b=0.8)
    obs = observe(state)
    assert obs.a_value == pytest.approx(0.2)
    assert obs.b_value is None
    assert obs.b_observed is False


def test_facts_from_observation_shared_quantization() -> None:
    obs = observe(_make_state(a=0.41, b=0.38))
    facts = facts_from_observation(obs)
    assert isinstance(facts, FactRecord)
    assert facts.a_obs is True
    assert facts.a_val_q == quantize_value(0.41)
    assert facts.b_val_q == quantize_value(0.38)
    assert facts.noise_bin is NoiseBin.HIGH
    assert facts.n_samples == 3
    assert facts.source is SensorSource.SENSOR_BOTH


def test_facts_none_when_unobserved() -> None:
    obs = observe(_make_state(a_observed=False, b_observed=True))
    facts = facts_from_observation(obs)
    assert facts.a_obs is False
    assert facts.a_val_q is None
    assert facts.b_val_q == quantize_value(0.38)


def test_quantize_value_edges() -> None:
    assert quantize_value(0.0) == 0
    assert quantize_value(0.41) == 4
    assert quantize_value(0.99) == 9
    assert quantize_value(1.0) == 9
    assert quantize_value(-0.5) == 0


def test_observation_and_state_are_distinct_types() -> None:
    state = _make_state()
    obs = observe(state)
    assert type(obs) is not type(state)
    assert Observation.__name__ == "Observation"
    assert State.__name__ == "State"


def test_frozen_types_reject_mutation() -> None:
    obs = observe(_make_state())
    with pytest.raises(Exception):
        obs.n_samples = 99  # type: ignore[misc]


def test_schema_checksum_stable() -> None:
    assert schema_checksum() == schema_checksum()
    assert len(schema_checksum()) == 64
