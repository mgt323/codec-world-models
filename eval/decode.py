"""Rebuild a codec string from predicted regex_v4 tokens.

Whitespace is not a token. Reconstruction keeps the true string's whitespace
and newlines (the skeleton) and writes the predicted tokens into the token
slots, so ``observation_from_*`` still sees the codec's line breaks.
"""

from __future__ import annotations

import re

from obs_codecs.diagnostics import tokenize

_NUMERIC_CHAR = frozenset("0123456789.")
_PUNCT = frozenset(":=,.;(){}|~→↑?")
_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_-]*|-?\d+(?:\.\d+)?|[:=,.;(){}|~→↑?]"
)
_NUMBER_SPAN_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def detokenize(tokens: list[str]) -> str:
    """Invert ``tokenize`` for codec strings produced by encode_A..D and E."""
    if not tokens:
        return ""
    parts: list[str] = []
    for tok in tokens:
        if not parts:
            parts.append(tok)
            continue
        prev = parts[-1]
        prev_last = prev[-1]
        is_num_char = len(tok) == 1 and tok in _NUMERIC_CHAR
        if tok == "-" and prev_last not in _NUMERIC_CHAR:
            if prev_last not in _PUNCT and not prev.endswith((" ", "\n")):
                parts.append(" ")
            parts.append("-")
            continue
        if is_num_char and (prev_last in _NUMERIC_CHAR or prev_last == "-"):
            parts[-1] = prev + tok
            continue
        if len(tok) == 1 and tok in _PUNCT:
            parts.append(tok)
            continue
        if prev_last not in _PUNCT and not prev.endswith((" ", "\n")):
            parts.append(" ")
        parts.append(tok)
    return "".join(parts)


def reconstruct_with_skeleton(true_text: str, predicted_tokens: list[str]) -> str | None:
    """Place ``predicted_tokens`` into the whitespace skeleton of ``true_text``.

    Returns ``None`` when the predicted token count does not match ``tokenize``.
    """
    if len(predicted_tokens) != len(tokenize(true_text)):
        return None
    pieces: list[str] = []
    cursor = 0
    pred_index = 0
    for match in _TOKEN_RE.finditer(true_text):
        pieces.append(true_text[cursor : match.start()])
        span = match.group()
        if _NUMBER_SPAN_RE.fullmatch(span) is not None:
            width = len(span)
            chars = predicted_tokens[pred_index : pred_index + width]
            if len(chars) != width:
                return None
            pieces.append("".join(chars))
            pred_index += width
        else:
            pieces.append(predicted_tokens[pred_index])
            pred_index += 1
        cursor = match.end()
    if pred_index != len(predicted_tokens):
        return None
    pieces.append(true_text[cursor:])
    return "".join(pieces)
