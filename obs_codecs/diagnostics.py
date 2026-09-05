"""Codec surface diagnostics: vocab, length, and token histogram.

Codec-agnostic: callers inject ``encode: Observation -> str``.
Tokenization is a shared whitespace/punctuation scheme (documented below)
so A–E diagnostics stay comparable until a trained tokenizer exists.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from world.schema import Observation

EncodeFn = Callable[[Observation], str]

# Shared provisional tokenizer (until model tokenizer lands):
# - identifiers / enums: letters, digits, underscore, hyphen
# - numbers: optional sign + digits + optional fraction
# - punctuation kept as singleton tokens: : = , . ( ) ~ → ↑
# Whitespace (including newlines) is a separator only.
_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_-]*|-?\d+(?:\.\d+)?|[:=,.()~→↑]"
)

TOKENIZATION_SCHEME = "regex_v0: ident|number|:=,.()~→↑ ; whitespace-separated"


def tokenize(text: str) -> list[str]:
    """Tokenize an encoded codec string with the shared provisional scheme."""
    return _TOKEN_RE.findall(text)


@dataclass(frozen=True, slots=True)
class CodecDiagnostics:
    """Surface statistics for one encode function over a sample of Observations."""

    n_observations: int
    vocab_size: int
    avg_tokens_per_observation: float
    avg_string_length_chars: float
    total_tokens: int
    tokenization_scheme: str
    token_histogram: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_codec_diagnostics(
    observations: Sequence[Observation],
    *,
    encode: EncodeFn,
) -> CodecDiagnostics:
    """Encode each observation and aggregate vocab / length / histogram stats."""
    if not observations:
        return CodecDiagnostics(
            n_observations=0,
            vocab_size=0,
            avg_tokens_per_observation=0.0,
            avg_string_length_chars=0.0,
            total_tokens=0,
            tokenization_scheme=TOKENIZATION_SCHEME,
            token_histogram={},
        )

    counts: Counter[str] = Counter()
    token_lens: list[int] = []
    char_lens: list[int] = []

    for obs in observations:
        text = encode(obs)
        tokens = tokenize(text)
        counts.update(tokens)
        token_lens.append(len(tokens))
        char_lens.append(len(text))

    n = len(observations)
    total_tokens = int(sum(token_lens))
    return CodecDiagnostics(
        n_observations=n,
        vocab_size=len(counts),
        avg_tokens_per_observation=total_tokens / n,
        avg_string_length_chars=sum(char_lens) / n,
        total_tokens=total_tokens,
        tokenization_scheme=TOKENIZATION_SCHEME,
        token_histogram=dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    )


def write_diagnostics_json(report: CodecDiagnostics, path: str | Path) -> Path:
    """Write diagnostics report as JSON (UTF-8)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
