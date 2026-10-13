"""Codec surface diagnostics: vocab, length, and token histogram.

Codec-agnostic: callers inject ``encode: Observation -> str``.
Tokenization is a shared whitespace/punctuation scheme (documented below)
so A–E diagnostics stay comparable until a trained model tokenizer exists.

**Locked for compute-matching:** ``regex_v4`` is the final *provisional*
scheme used for avg-token diagnostics and matched-compute / token-update
budgets until a trained tokenizer replaces it. Do not bump to v5 after
budgets are set from v4 counts. Optional reorder-only controls (C-shuffle,
A-bag) must not introduce new punctuation; any new glyph requires an
explicit scheme bump **and** recomputation of all token-based budgets.

**v4 numeric decision:** matched float/int spans are split into character
tokens (digits, ``-``, ``.``). Encoded codec *strings* are unchanged —
tokenizer-only. Rejected: (1) rounding ``_format_value`` — bin-flip vs
``quantize_value`` / FactRecord; (2) bin-index surface — drops continuous
values needed for eval task #1 (MSE next-state).
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
# - numbers (match): optional sign + digits + optional fraction, then
#   *split* into per-character tokens (digits, "-", ".") — regex_v4
# - punctuation kept as singleton tokens: : = , . ; ( ) { } | ~ → ↑ ?
# Whitespace (including newlines) is a separator only.
#
# History: v1 { } ? (Codec C); v2 | (E bag_b); v3 ; (Codec D);
# v4 digit-level numeric pieces (bounded vocab; strings unchanged).
# Pattern glyph set is still the v3 surface coverage; reserved ↑ ~ kept for
# plan-mentioned glyphs that are not currently emitted by encode_*.
# LOCKED — see module docstring (no silent v5 after compute budgets).
_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_-]*|-?\d+(?:\.\d+)?|[:=,.;(){}|~→↑?]"
)
_NUMBER_SPAN_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

TOKENIZATION_SCHEME_VERSION = "regex_v4"
TOKENIZATION_SCHEME = (
    "regex_v4: ident|number->chars|:=,.;(){}|~→↑? ; whitespace-separated"
)


def tokenize(text: str) -> list[str]:
    """Tokenize an encoded codec string with the shared provisional scheme.

    Non-numeric matches are kept as in regex_v3. Each matched number span is
    expanded to individual character tokens (e.g. ``0.41`` → ``0`` ``.`` ``4``
    ``1``).
    """
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        if _NUMBER_SPAN_RE.fullmatch(tok) is not None:
            out.extend(tok)
        else:
            out.append(tok)
    return out


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
