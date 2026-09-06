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


def test_tokenize_keeps_codec_c_markers() -> None:
    """v1 scheme: braces and the missing-value marker are tokens, not dropped."""
    tokens = tokenize("obs: A=?, B=0.38\nalts: {common_cause, spurious}")
    assert "?" in tokens
    assert "{" in tokens
    assert "}" in tokens
    assert "common_cause" in tokens


def test_compute_diagnostics_codec_a() -> None:
    sample = [_obs()]
    report = compute_codec_diagnostics(sample, encode=encode_A)
    assert report.n_observations == 1
    assert report.vocab_size > 0
    assert report.avg_tokens_per_observation > 0
    assert report.avg_string_length_chars > 0
    assert report.total_tokens == int(report.avg_tokens_per_observation)
    assert sum(report.token_histogram.values()) == report.total_tokens
    assert report.tokenization_scheme.startswith("regex_v3")


def test_tokenize_keeps_codec_d_semicolon() -> None:
    """v3 scheme: Codec D field separators are tokens, not dropped."""
    tokens = tokenize("A.x=None; A.obs=0; noise=high")
    assert ";" in tokens
    assert "None" in tokens
    assert "A" in tokens  # 'A.x' splits as A . x under this scheme


def test_tokenize_keeps_bag_pipe() -> None:
    """v2 scheme: bag_b pipe is a token, not dropped."""
    tokens = tokenize("(observe A=0.41) | (miss B)")
    assert "|" in tokens
    assert "observe" in tokens


def test_regex_v3_covers_all_shipped_codec_surfaces() -> None:
    """No non-whitespace residue after tokenize — guards silent v4 pressure."""
    import re

    from obs_codecs.diagnostics import _TOKEN_RE
    from obs_codecs.encode_a import encode_A
    from obs_codecs.encode_b import encode_B
    from obs_codecs.encode_c import encode_C
    from obs_codecs.encode_d import encode_D
    from obs_codecs.transforms_e import bag_b, reverse_b, shuffle_b
    from world.parity_fixtures import sample_diverse_parity_states
    from world.schema import observe

    observations = [observe(s) for s in sample_diverse_parity_states(target_n=200)]

    def encode_paths(obs):
        b = encode_B(obs)
        return (
            encode_A(obs),
            b,
            encode_C(obs),
            encode_D(obs),
            shuffle_b(b, 0),
            reverse_b(b),
            bag_b(b, 0),
            # Optional C-shuffle / A-bag are reorder-only; same glyph set.
            "\n".join(reversed(encode_C(obs).splitlines())),
        )

    uncovered: set[str] = set()
    for obs in observations:
        for text in encode_paths(obs):
            leftover = re.sub(r"\s+", "", _TOKEN_RE.sub("", text))
            uncovered.update(leftover)
    assert not uncovered, f"regex_v3 leaves uncovered glyphs: {sorted(uncovered)!r}"


def test_diagnostics_module_is_codec_agnostic() -> None:
    import inspect

    import obs_codecs.diagnostics as diag

    source = inspect.getsource(diag)
    assert "encode_A" not in source
