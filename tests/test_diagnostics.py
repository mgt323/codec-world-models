"""Tests for codec surface diagnostics."""

from __future__ import annotations

from obs_codecs.diagnostics import compute_codec_diagnostics, tokenize
from obs_codecs.encode_a import encode_A
from world.schema import NoiseBin, Observation, SensorSource


def _obs() -> Observation:
    return Observation(
        a_observed=True,
        a_value=0.41,
        b_observed=True,
        b_value=0.38,
        noise_bin=NoiseBin.HIGH,
        n_samples=3,
        source=SensorSource.SENSOR_BOTH,
    )


def test_tokenize_splits_ident_number_and_punct() -> None:
    text = "Entity A: value=0.41, status=present."
    tokens = tokenize(text)
    assert "Entity" in tokens
    assert "A" in tokens
    assert ":" in tokens
    assert "value" in tokens
    assert "=" in tokens
    assert "0.41" in tokens
    assert "," in tokens
    assert "present" in tokens
    assert "." in tokens


def test_compute_diagnostics_codec_a() -> None:
    sample = [_obs()]
    report = compute_codec_diagnostics(sample, encode=encode_A)
    assert report.n_observations == 1
    assert report.vocab_size > 0
    assert report.avg_tokens_per_observation > 0
    assert report.avg_string_length_chars > 0
    assert report.total_tokens == int(report.avg_tokens_per_observation)
    assert sum(report.token_histogram.values()) == report.total_tokens
    assert report.tokenization_scheme.startswith("regex_v0")


def test_diagnostics_module_is_codec_agnostic() -> None:
    import inspect

    import obs_codecs.diagnostics as diag

    source = inspect.getsource(diag)
    assert "encode_A" not in source
