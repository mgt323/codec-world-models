"""Observation codecs (encode/parse), parity audit, and surface diagnostics."""

from .diagnostics import (
    CodecDiagnostics,
    compute_codec_diagnostics,
    tokenize,
    write_diagnostics_json,
)
from .encode_a import decode_A_facts, encode_A, observation_from_A, parse_A
from .encode_b import (
    decode_B_facts,
    decode_B_facts_unordered,
    encode_B,
    observation_from_B,
    observation_from_B_unordered,
    parse_B,
)
from .encode_c import decode_C_facts, encode_C, observation_from_C, parse_C
from .encode_d import decode_D_facts, encode_D, observation_from_D, parse_D
from .parity_audit import (
    ParityAuditReport,
    ParityMismatch,
    ParseFactsFn,
    run_parity_audit,
)
from .transforms_e import bag_b, derive_transform_seed, reverse_b, shuffle_b

__all__ = [
    "CodecDiagnostics",
    "ParityAuditReport",
    "ParityMismatch",
    "ParseFactsFn",
    "bag_b",
    "compute_codec_diagnostics",
    "decode_A_facts",
    "decode_B_facts",
    "decode_B_facts_unordered",
    "decode_C_facts",
    "decode_D_facts",
    "derive_transform_seed",
    "encode_A",
    "encode_B",
    "encode_C",
    "encode_D",
    "observation_from_A",
    "observation_from_B",
    "observation_from_B_unordered",
    "observation_from_C",
    "observation_from_D",
    "parse_A",
    "parse_B",
    "parse_C",
    "parse_D",
    "reverse_b",
    "run_parity_audit",
    "shuffle_b",
    "tokenize",
    "write_diagnostics_json",
]
