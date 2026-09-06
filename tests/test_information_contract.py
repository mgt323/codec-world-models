"""Information-contract tests: Observation must not carry latent H_t."""

from __future__ import annotations

import pickle
from dataclasses import fields, is_dataclass
from typing import get_args, get_origin, get_type_hints

from world.schema import (
    CausalRegime,
    LatentState,
    NoiseBin,
    Observation,
    SensorSource,
    State,
    facts_from_observation,
    observe,
)

# Field names that would constitute a latent / regime side channel on Observation.
_FORBIDDEN_OBS_FIELD_NAMES = frozenset(
    {
        "hidden_c",
        "C_hidden",
        "c_hidden",
        "regime",
        "causal_regime",
        "oracle_regime_posterior",
        "latent",
        "active_intervention",
        "intervention",
    }
)


def _type_mentions(annotation: object, target: type) -> bool:
    """True if annotation is target or a Union/Optional wrapping target."""
    if annotation is target:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    return any(_type_mentions(arg, target) for arg in get_args(annotation))


def _base_state(*, regime: CausalRegime, hidden_c: float = 0.5) -> State:
    """Fixed observables; only latent may differ across callers."""
    return State(
        t=0,
        a=0.41,
        b=0.38,
        a_observed=True,
        b_observed=True,
        noise_bin=NoiseBin.HIGH,
        n_samples=3,
        source=SensorSource.SENSOR_BOTH,
        latent=LatentState(hidden_c=hidden_c, regime=regime),
        active_intervention=None,
    )


def test_observation_fields_cannot_hold_regime_or_hidden_c() -> None:
    """Via dataclass field introspection: no CausalRegime / hidden_c channel."""
    assert is_dataclass(Observation)
    hints = get_type_hints(Observation)

    for field in fields(Observation):
        assert field.name not in _FORBIDDEN_OBS_FIELD_NAMES, (
            f"Observation field {field.name!r} must not carry latent/regime"
        )
        annotation = hints[field.name]
        assert not _type_mentions(annotation, CausalRegime), (
            f"Observation.{field.name} type {annotation!r} can hold CausalRegime"
        )

    assert "hidden_c" not in hints
    assert not any(_type_mentions(t, CausalRegime) for t in hints.values())
    # LatentState.hidden_c is float; Observation must not expose that channel by name.
    assert "hidden_c" not in {f.name for f in fields(Observation)}
    assert "hidden_c" in LatentState.__dataclass_fields__

def test_observe_byte_identical_across_all_regimes() -> None:
    """Same a/b/noise/source/n_samples; all 4 regimes -> byte-identical Observation."""
    regimes = list(CausalRegime)
    assert len(regimes) == 4

    payloads = [pickle.dumps(observe(_base_state(regime=r)), protocol=pickle.HIGHEST_PROTOCOL) for r in regimes]

    assert len(set(payloads)) == 1, (
        "observe() must not leak LatentState.regime into Observation bytes"
    )
    # Sanity: objects also compare equal.
    observations = [observe(_base_state(regime=r)) for r in regimes]
    assert all(o == observations[0] for o in observations)


def test_facts_from_observation_deterministic_and_quantization_stable() -> None:
    """Same Observation -> same FactRecord across repeated calls."""
    obs = observe(_base_state(regime=CausalRegime.COMMON_CAUSE))
    first = facts_from_observation(obs)
    for _ in range(16):
        again = facts_from_observation(obs)
        assert again == first
        assert pickle.dumps(again, protocol=pickle.HIGHEST_PROTOCOL) == pickle.dumps(
            first, protocol=pickle.HIGHEST_PROTOCOL
        )


def test_encode_identical_when_only_latent_differs() -> None:
    """PROGRAM_SPEC §4.3: same O, different H ⇒ identical encodings (A–D)."""
    from obs_codecs.encode_a import encode_A
    from obs_codecs.encode_b import encode_B
    from obs_codecs.encode_c import encode_C
    from obs_codecs.encode_d import encode_D

    states = [
        _base_state(regime=r, hidden_c=h)
        for r in CausalRegime
        for h in (0.0, 0.5, 1.0)
    ]
    observations = [observe(s) for s in states]
    assert len(set(observations)) == 1

    texts_a = {encode_A(observe(s)) for s in states}
    texts_b = {encode_B(observe(s)) for s in states}
    texts_c = {encode_C(observe(s)) for s in states}
    texts_d = {encode_D(observe(s)) for s in states}
    assert len(texts_a) == 1
    assert len(texts_b) == 1
    assert len(texts_c) == 1
    assert len(texts_d) == 1


def test_encode_identical_when_only_intervention_differs() -> None:
    """Option B: active_intervention on State must not change codec encodings."""
    from obs_codecs.encode_a import encode_A
    from obs_codecs.encode_b import encode_B
    from obs_codecs.encode_c import encode_C
    from obs_codecs.encode_d import encode_D
    from world.schema import Intervention, InterventionTarget

    base = _base_state(regime=CausalRegime.A_CAUSES_B)
    with_do = State(
        t=base.t,
        a=base.a,
        b=base.b,
        a_observed=base.a_observed,
        b_observed=base.b_observed,
        noise_bin=base.noise_bin,
        n_samples=base.n_samples,
        source=base.source,
        latent=base.latent,
        active_intervention=Intervention(
            target=InterventionTarget.A, value=0.9, timestep=0
        ),
    )
    assert observe(base) == observe(with_do)
    assert encode_A(observe(base)) == encode_A(observe(with_do))
    assert encode_B(observe(base)) == encode_B(observe(with_do))
    assert encode_C(observe(base)) == encode_C(observe(with_do))
    assert encode_D(observe(base)) == encode_D(observe(with_do))
