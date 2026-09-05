"""Round-trip / info-equality tests for Codec A."""

from __future__ import annotations

import pytest

from obs_codecs.encode_a import encode_A, observation_from_A, parse_A
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
def test_codec_a_roundtrip_facts_match(obs: Observation) -> None:
    """encode_A -> parse_A equals F(O); observation_from_A recovers Observation."""
    text = encode_A(obs)
    assert "→" not in text
    assert "hits" not in text.lower()

    assert parse_A(text) == facts_from_observation(obs)
    assert facts_from_observation(observation_from_A(text)) == facts_from_observation(obs)


def test_encode_a_essentialist_format() -> None:
    """Smoke: encode_A returns essentialist caption lines."""
    text = encode_A(_obs())
    assert text.startswith("Entity A:")
    assert "Context:" in text
    assert "Relation: A and B are similar." in text
