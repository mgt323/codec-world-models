"""Round-trip / info-equality / leakage tests for Codec D."""

from __future__ import annotations

import inspect

import pytest

from obs_codecs.encode_d import decode_D_facts, encode_D, observation_from_D, parse_D
from obs_codecs.parity_audit import run_parity_audit
from world.parity_fixtures import sample_diverse_parity_states
from world.schema import (
    CausalRegime,
    Intervention,
    InterventionTarget,
    LatentState,
    NoiseBin,
    Observation,
    SensorSource,
    State,
    facts_from_observation,
    observe,
)


def _obs(
    *,
    a_observed: bool = True,
    a_value: float | None = 0.41,
    b_observed: bool = True,
    b_value: float | None = 0.38,
    noise_bin: NoiseBin = NoiseBin.HIGH,
    n_samples: int = 3,
    source: SensorSource = SensorSource.SENSOR_BOTH,
) -> Observation:
    return Observation(
        a_observed=a_observed,
        a_value=a_value if a_observed else None,
        b_observed=b_observed,
        b_value=b_value if b_observed else None,
        noise_bin=noise_bin,
        n_samples=n_samples,
        source=source,
    )


def _state(
    *,
    regime: CausalRegime = CausalRegime.A_CAUSES_B,
    active_intervention: Intervention | None = None,
) -> State:
    return State(
        t=0,
        a=0.41,
        b=0.38,
        a_observed=True,
        b_observed=True,
        noise_bin=NoiseBin.HIGH,
        n_samples=3,
        source=SensorSource.SENSOR_BOTH,
        latent=LatentState(hidden_c=0.5, regime=regime),
        active_intervention=active_intervention,
    )


@pytest.mark.parametrize(
    "obs",
    [
        _obs(),
        _obs(a_observed=False, a_value=None, source=SensorSource.SENSOR_B),
        _obs(b_observed=False, b_value=None, source=SensorSource.SENSOR_A),
        _obs(
            a_observed=False,
            b_observed=False,
            noise_bin=NoiseBin.LOW,
            n_samples=1,
            source=SensorSource.SENSOR_NONE,
        ),
        _obs(a_value=0.3, b_value=0.99, noise_bin=NoiseBin.MEDIUM),
    ],
)
def test_codec_d_roundtrip_facts_match(obs: Observation) -> None:
    text = encode_D(obs)
    assert "→" not in text
    assert "Entity" not in text
    assert "alts" not in text
    assert "Relation" not in text
    assert parse_D(text) == facts_from_observation(obs)
    assert facts_from_observation(observation_from_D(text)) == facts_from_observation(obs)
    assert decode_D_facts(text) == parse_D(text)


def test_encode_d_structured_baseline_format() -> None:
    """Smoke: flat key=value line; None for missing; no decorative fields."""
    text = encode_D(_obs(b_observed=False, b_value=None, source=SensorSource.SENSOR_A))
    assert text == (
        "A.x=0.41; A.obs=1; B.x=None; B.obs=0; "
        "noise=high; n=3; source=sensor_a"
    )
    assert "source-like" not in text
    assert "none" not in text  # prose token from A; D uses None
    assert "?" not in text


def test_codec_d_roundtrip_on_diverse_parity_sample() -> None:
    observations = [observe(s) for s in sample_diverse_parity_states(target_n=500)]
    report = run_parity_audit(observations, encode=encode_D, parse=parse_D)
    assert report.ok
    assert report.n_passed == 500
    for obs in observations:
        assert observation_from_D(encode_D(obs)) == obs


def test_d_intervention_blind() -> None:
    base = _state()
    do_a = _state(
        active_intervention=Intervention(
            target=InterventionTarget.A, value=0.9, timestep=0
        )
    )
    assert observe(base) == observe(do_a)
    assert encode_D(observe(base)) == encode_D(observe(do_a))


def test_encode_d_imports_observation_only() -> None:
    import obs_codecs.encode_d as mod

    enc_sig = inspect.signature(encode_D)
    assert list(enc_sig.parameters) == ["obs"]
    hints = inspect.get_annotations(encode_D, eval_str=True)
    assert hints["obs"] is Observation
    assert not hasattr(mod, "State")
    assert not hasattr(mod, "LatentState")
    assert not hasattr(mod, "CausalRegime")
    assert not hasattr(mod, "Intervention")


def test_parse_d_rejects_wrong_field_count() -> None:
    good = encode_D(_obs())
    truncated = "; ".join(good.split("; ")[:5])
    with pytest.raises(ValueError, match="expects 7 fields"):
        parse_D(truncated)


def test_parse_d_rejects_wrong_key_names() -> None:
    text = (
        "A.val=0.41; A.obs=1; B.x=0.38; B.obs=1; "
        "noise=high; n=3; source=sensor_both"
    )
    with pytest.raises(ValueError, match="expected key 'A.x'"):
        parse_D(text)


def test_parse_d_rejects_wrong_separator() -> None:
    text = (
        "A.x=0.41, A.obs=1, B.x=0.38, B.obs=1, "
        "noise=high, n=3, source=sensor_both"
    )
    with pytest.raises(ValueError, match="expects 7 fields"):
        parse_D(text)


def test_codec_a_b_c_d_recover_same_facts() -> None:
    from obs_codecs.encode_a import encode_A, parse_A
    from obs_codecs.encode_b import encode_B, parse_B
    from obs_codecs.encode_c import encode_C, parse_C

    obs = _obs()
    expected = facts_from_observation(obs)
    assert parse_A(encode_A(obs)) == expected
    assert parse_B(encode_B(obs)) == expected
    assert parse_C(encode_C(obs)) == expected
    assert parse_D(encode_D(obs)) == expected
