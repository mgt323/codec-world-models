"""Tests for reusable information-parity audit."""

from __future__ import annotations

import inspect

from obs_codecs.encode_a import decode_A_facts, encode_A
from obs_codecs.parity_audit import run_parity_audit, sample_parity_states
from world.schema import (
    CausalRegime,
    FactRecord,
    NoiseBin,
    SensorSource,
    facts_from_observation,
)


def test_sample_parity_states_coverage() -> None:
    states = sample_parity_states()
    regimes = {s.latent.regime for s in states}
    noises = {s.noise_bin for s in states}
    assert regimes == set(CausalRegime)
    assert NoiseBin.LOW in noises
    assert NoiseBin.HIGH in noises
    assert any(not s.a_observed for s in states)
    assert any(not s.b_observed for s in states)
    assert any(s.a_observed and s.b_observed for s in states)


def test_parity_audit_codec_a_full_pass() -> None:
    report = run_parity_audit(
        sample_parity_states(),
        encode=encode_A,
        decode_facts=decode_A_facts,
    )
    assert report.ok
    assert report.pass_rate == 1.0
    assert report.n_total == len(sample_parity_states())
    assert report.mismatches == ()


def test_parity_audit_reports_mismatch_detail() -> None:
    def bad_decode(_text: str) -> FactRecord:
        return FactRecord(
            a_obs=False,
            a_val_q=None,
            b_obs=False,
            b_val_q=None,
            noise_bin=NoiseBin.LOW,
            n_samples=0,
            source=SensorSource.SENSOR_NONE,
        )

    states = sample_parity_states()[:1]
    report = run_parity_audit(states, encode=encode_A, decode_facts=bad_decode)
    assert not report.ok
    assert report.pass_rate == 0.0
    assert len(report.mismatches) == 1
    m = report.mismatches[0]
    assert m.error == "fact_mismatch"
    assert m.encoded.startswith("Entity A:")
    assert m.expected == facts_from_observation(m.observation)
    assert m.recovered != m.expected
    detail = m.to_detail_dict()
    assert "state" in detail and "observation" in detail
    assert "expected" in detail and "recovered" in detail
    assert "parity_audit:" in report.summary()


def test_parity_audit_module_is_codec_agnostic() -> None:
    """Runner must not hardcode Codec A (encode/decode injected by caller)."""
    import obs_codecs.parity_audit as audit_mod

    source = inspect.getsource(audit_mod)
    assert "encode_A" not in source
    assert "decode_A" not in source
    assert "parse_A" not in source
