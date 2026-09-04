"""Observation codecs (encode/parse), parity audit, and surface diagnostics."""

from .diagnostics import (
    CodecDiagnostics,
    compute_codec_diagnostics,
    tokenize,
    write_diagnostics_json,
)
from .encode_a import decode_A_facts, encode_A, parse_A
from .parity_audit import (
    ParityAuditReport,
    ParityMismatch,
    run_parity_audit,
    sample_diverse_parity_states,
    sample_parity_states,
)

__all__ = [
    "CodecDiagnostics",
    "ParityAuditReport",
    "ParityMismatch",
    "compute_codec_diagnostics",
    "decode_A_facts",
    "encode_A",
    "parse_A",
    "run_parity_audit",
    "sample_diverse_parity_states",
    "sample_parity_states",
    "tokenize",
    "write_diagnostics_json",
]
