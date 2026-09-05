"""Round-trip / info-equality tests for Codec B."""

from __future__ import annotations

import inspect

import pytest

from obs_codecs.encode_b import encode_B, observation_from_B, parse_B
from world.schema import (
    NoiseBin,
    Observation,
    SensorSource,
    facts_from_observation,
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
def test_codec_b_roundtrip_facts_match(obs: Observation) -> None:
    """encode_B -> parse_B equals F(O); observation_from_B recovers Observation."""
    text = encode_B(obs)
    assert "→" in text
    assert "Entity" not in text

    assert parse_B(text) == facts_from_observation(obs)
    assert facts_from_observation(observation_from_B(text)) == facts_from_observation(obs)


def test_encode_b_process_event_chain_format() -> None:
    """Smoke: encode_B returns relational event chain with → scaffolding."""
    text = encode_B(_obs())
    assert text.startswith("(observe A=")
    assert "(co-vary B=" in text
    assert "(meta n=3 source=sensor_both noise=high)" in text
    assert "↑" not in text
    assert "link" not in text
    assert "strength" not in text
    assert text.count(" → ") == 2


def test_encode_b_never_emits_do_or_intervention_tokens() -> None:
    """PROGRAM_SPEC option B: codecs must not lexicalize hidden do()."""
    text = encode_B(_obs())
    lowered = text.lower()
    assert "do(" not in lowered
    assert "interven" not in lowered
    assert "is_intervened" not in lowered


def test_encode_b_imports_observation_only() -> None:
    """I1: encode_B consumes Observation only — no State / Intervention types."""
    import obs_codecs.encode_b as mod

    enc_sig = inspect.signature(encode_B)
    assert list(enc_sig.parameters) == ["obs"]
    hints = inspect.get_annotations(encode_B, eval_str=True)
    assert hints["obs"] is Observation

    assert not hasattr(mod, "State")
    assert not hasattr(mod, "LatentState")
    assert not hasattr(mod, "Intervention")


def test_codec_a_and_b_recover_same_facts() -> None:
    """Cross-codec info-equality: parse∘encode F matches for A and B on same O."""
    from obs_codecs.encode_a import encode_A, parse_A

    obs = _obs()
    assert parse_A(encode_A(obs)) == parse_B(encode_B(obs)) == facts_from_observation(obs)
