"""Tests for reusable information-parity audit."""

from __future__ import annotations

import inspect

from obs_codecs.encode_a import encode_A, parse_A
from obs_codecs.encode_b import encode_B, parse_B
from obs_codecs.parity_audit import run_parity_audit
from world.parity_fixtures import sample_parity_states
from world.schema import (
    CausalRegime,
    FactRecord,
    NoiseBin,
    SensorSource,
    facts_from_observation,
    observe,
)


def test_sample_parity_states_coverage() -> None:
    states = sample_parity_states()
    regimes = {s.latent.regime for s in states}
    noises = {s.noise_bin for s in states}
    assert regimes == set(CausalRegime)
    assert NoiseBin.LOW in noises
    assert NoiseBin.MEDIUM in noises
    assert NoiseBin.HIGH in noises
    assert any(not s.a_observed for s in states)
    assert any(not s.b_observed for s in states)
    assert any(s.a_observed and s.b_observed for s in states)
    assert any((not s.a_observed) and (not s.b_observed) for s in states)
    assert any(s.source is SensorSource.SENSOR_NONE for s in states)


def test_parity_audit_codec_a_full_pass() -> None:
    observations = [observe(s) for s in sample_parity_states()]
    report = run_parity_audit(observations, encode=encode_A, parse=parse_A)
    assert report.ok
    assert report.pass_rate == 1.0
    assert report.n_total == len(observations)
    assert report.mismatches == ()


def test_parity_audit_codec_b_full_pass() -> None:
    """Same shared runner as A; Codec B encode/parse injected only."""
    observations = [observe(s) for s in sample_parity_states()]
    report = run_parity_audit(observations, encode=encode_B, parse=parse_B)
    assert report.ok
    assert report.pass_rate == 1.0
    assert report.n_total == len(observations)
    assert report.mismatches == ()


def test_parity_audit_reports_mismatch_detail() -> None:
    def bad_parse(_text: str) -> FactRecord:
        return FactRecord(
            a_obs=False,
            a_val_q=None,
            b_obs=False,
            b_val_q=None,
            noise_bin=NoiseBin.LOW,
            n_samples=0,
            source=SensorSource.SENSOR_NONE,
        )

    observations = [observe(s) for s in sample_parity_states()[:1]]
    report = run_parity_audit(observations, encode=encode_A, parse=bad_parse)
    assert not report.ok
    assert report.pass_rate == 0.0
    assert len(report.mismatches) == 1
    m = report.mismatches[0]
    assert m.error == "fact_mismatch"
    assert m.encoded.startswith("Entity A:")
    assert m.expected == facts_from_observation(m.observation)
    assert m.recovered != m.expected
    detail = m.to_detail_dict()
    assert "observation" in detail
    assert "state" not in detail
    assert "expected" in detail and "recovered" in detail
    assert "parity_audit:" in report.summary()


def test_parity_audit_module_is_codec_agnostic_and_state_free() -> None:
    """Runner must not hardcode codecs or bind State / latent types."""
    import obs_codecs.parity_audit as audit_mod

    source = inspect.getsource(audit_mod)
    assert "encode_A" not in source
    assert "decode_A" not in source
    assert "parse_A" not in source
    assert "from world.schema import" in source
    # Only Observation / FactRecord from world.schema — no State/latent imports.
    import_block = source.split("from world.schema import", 1)[1].split(")", 1)[0]
    assert "Observation" in import_block
    assert "FactRecord" in import_block
    assert "State" not in import_block
    assert "LatentState" not in import_block
    assert "Intervention" not in import_block
    assert not hasattr(audit_mod, "State")
    assert not hasattr(audit_mod, "LatentState")
