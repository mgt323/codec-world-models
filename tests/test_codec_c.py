"""Round-trip / info-equality / leakage tests for Codec C."""

from __future__ import annotations

import inspect
from dataclasses import asdict, fields

import pytest

from obs_codecs.encode_c import decode_C_facts, encode_C, observation_from_C, parse_C
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
    observe,
)

_ALTS_LINE = "alts: {common_cause, a_causes_b, b_causes_a, spurious}"


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
    regime: CausalRegime,
    hidden_c: float = 0.5,
    active_intervention: Intervention | None = None,
) -> State:
    """Fixed observables; only latent / intervention may differ across callers."""
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
def test_codec_c_roundtrip_facts_match(obs: Observation) -> None:
    """encode_C -> parse_C equals F(O); observation_from_C recovers Observation."""
    text = encode_C(obs)
    assert "Entity" not in text
    assert "→" not in text

    assert parse_C(text) == facts_from_observation(obs)
    assert facts_from_observation(observation_from_C(text)) == facts_from_observation(obs)


def test_encode_c_evidential_record_format() -> None:
    """Smoke: 5 lines in locked order; missing values use ? not none."""
    text = encode_C(_obs(b_observed=False, b_value=None, source=SensorSource.SENSOR_A))
    lines = text.splitlines()
    assert lines == [
        "obs: A=0.41, B=?",
        "source: sensor_a",
        "n: 3",
        "noise: high",
        _ALTS_LINE,
    ]
    assert "none" not in text


def test_encode_c_has_no_oracle_or_regime_fields() -> None:
    """I8: no provided posterior, no true cause, no do() lexicalization."""
    text = encode_C(_obs()).lower()
    assert "p=" not in text
    assert "posterior" not in text
    assert "regime" not in text
    assert "true_cause" not in text
    assert "do(" not in text
    assert "interven" not in text


def test_alts_is_byte_identical_regardless_of_state() -> None:
    """Same O, any H: encodings are byte-identical (design-review safety table)."""
    states = [
        _state(regime=regime, hidden_c=hidden_c)
        for regime in CausalRegime
        for hidden_c in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert len({s.latent.regime for s in states}) == 4
    assert len({s.latent.hidden_c for s in states}) == 5

    encodings = {encode_C(observe(s)) for s in states}
    assert len(encodings) == 1, "encode_C output varies with LatentState"

    only = encodings.pop()
    assert only.splitlines()[-1] == _ALTS_LINE


def test_alts_line_is_hardcoded_constant_across_observations() -> None:
    """alts never varies with any Observation field, and is not interpolated."""
    varied = [
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
        _obs(a_value=0.0, b_value=1.0, noise_bin=NoiseBin.MEDIUM, n_samples=5),
    ]
    assert {encode_C(o).splitlines()[-1] for o in varied} == {_ALTS_LINE}

    source = inspect.getsource(inspect.getmodule(encode_C))
    assert f'_ALTS_LINE = "{_ALTS_LINE}"' in source


def test_alts_tokens_match_causal_regime_declaration_order() -> None:
    """Guard against drift between the constant and CausalRegime values."""
    expected = ", ".join(regime.value for regime in CausalRegime)
    assert _ALTS_LINE == f"alts: {{{expected}}}"
    assert "spurious" in _ALTS_LINE
    # The 4th hypothesis must not reuse the measurement-noise word.
    assert _ALTS_LINE.count("noise") == 0


def test_alts_excluded_from_fact_record() -> None:
    """parse_C/decode_C_facts return the fixed 7-field F(O) with nothing alts-derived."""
    expected_fields = (
        "a_obs",
        "a_val_q",
        "b_obs",
        "b_val_q",
        "noise_bin",
        "n_samples",
        "source",
    )
    obs = _obs()
    record = parse_C(encode_C(obs))

    assert type(record) is FactRecord
    assert tuple(f.name for f in fields(record)) == expected_fields
    assert tuple(asdict(record)) == expected_fields
    assert record == facts_from_observation(obs)
    assert decode_C_facts(encode_C(obs)) == record

    for name in ("alts", "alternatives", "hypotheses", "regime", "oracle_p", "p"):
        assert not hasattr(record, name)
    assert "spurious" not in str(asdict(record))
    with pytest.raises((AttributeError, TypeError)):
        record.alts = "leak"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "bad_alts",
    [
        "alts: {a_causes_b, common_cause, b_causes_a, spurious}",  # reordered
        "alts: {common_cause, a_causes_b, b_causes_a}",  # missing entry
        "alts: {common_cause, a_causes_b, b_causes_a, spurious, noise}",  # extra entry
        "alts: {Common_Cause, a_causes_b, b_causes_a, spurious}",  # casing
        "alts: {common_cause, a_causes_b, b_causes_a, noise}",  # plan's ambiguous label
        "alts: common_cause, a_causes_b, b_causes_a, spurious",  # no braces
        "alts: {}",  # emptied
        "",  # dropped line
    ],
)
def test_alts_malformed_rejected(bad_alts: str) -> None:
    """Any deviation from the exact literal must raise, never parse silently."""
    good = encode_C(_obs())
    mangled = "\n".join(good.splitlines()[:4] + [bad_alts])

    with pytest.raises(ValueError):
        parse_C(mangled)
    with pytest.raises(ValueError):
        observation_from_C(mangled)


def test_c_intervention_blind() -> None:
    """Option B: active_intervention on State must not change encode_C output."""
    base = _state(regime=CausalRegime.A_CAUSES_B)
    do_a = _state(
        regime=CausalRegime.A_CAUSES_B,
        active_intervention=Intervention(target=InterventionTarget.A, value=0.9, timestep=0),
    )
    do_b = _state(
        regime=CausalRegime.A_CAUSES_B,
        active_intervention=Intervention(target=InterventionTarget.B, value=0.1, timestep=0),
    )

    assert observe(base) == observe(do_a) == observe(do_b)
    texts = {encode_C(observe(s)) for s in (base, do_a, do_b)}
    assert len(texts) == 1


def test_encode_c_imports_observation_only() -> None:
    """I1 / §4.1: encode_C consumes Observation only — no State / latent types."""
    import obs_codecs.encode_c as mod

    enc_sig = inspect.signature(encode_C)
    assert list(enc_sig.parameters) == ["obs"]
    hints = inspect.get_annotations(encode_C, eval_str=True)
    assert hints["obs"] is Observation

    assert not hasattr(mod, "State")
    assert not hasattr(mod, "LatentState")
    assert not hasattr(mod, "Intervention")
    assert not hasattr(mod, "CausalRegime")


def test_codec_a_b_c_recover_same_facts() -> None:
    """Cross-codec info-equality: parse∘encode F matches for A, B, and C on same O."""
    from obs_codecs.encode_a import encode_A, parse_A
    from obs_codecs.encode_b import encode_B, parse_B

    obs = _obs()
    expected = facts_from_observation(obs)
    assert parse_A(encode_A(obs)) == expected
    assert parse_B(encode_B(obs)) == expected
    assert parse_C(encode_C(obs)) == expected
