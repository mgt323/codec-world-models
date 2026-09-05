"""Observation codecs (encode/parse), parity audit, and surface diagnostics."""

from .diagnostics import (
    CodecDiagnostics,
    compute_codec_diagnostics,
    tokenize,
    write_diagnostics_json,
)
from .encode_a import decode_A_facts, encode_A, observation_from_A, parse_A
from .encode_b import decode_B_facts, encode_B, observation_from_B, parse_B
from .parity_audit import (
    ParityAuditReport,
    ParityMismatch,
    ParseFactsFn,
    run_parity_audit,
)

__all__ = [
    "CodecDiagnostics",
    "ParityAuditReport",
    "ParityMismatch",
    "ParseFactsFn",
    "compute_codec_diagnostics",
    "decode_A_facts",
    "decode_B_facts",
    "encode_A",
    "encode_B",
    "observation_from_A",
    "observation_from_B",
    "parse_A",
    "parse_B",
    "run_parity_audit",
    "tokenize",
    "write_diagnostics_json",
]
