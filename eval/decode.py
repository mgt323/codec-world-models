"""Rebuild a codec string from predicted regex_v4 tokens.

Whitespace is not a token. Reconstruction keeps the true string's whitespace
and newlines (the skeleton) and writes the predicted tokens into the token
slots, so ``observation_from_*`` still sees the codec's line breaks.
"""

from __future__ import annotations

import re

from obs_codecs.diagnostics import tokenize

_NUMERIC_CHAR = frozenset("0123456789.")
_NUMERIC_CHAR_OR_SIGN = frozenset("0123456789.-")
_ENTITY_VALUE_PREFIXES = ("value=", "A=", "B=", "A.x=", "B.x=")
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


def reconstruct_field_to_delimiter(
    true_text: str,
    predicted_tokens: list[str],
) -> str | None:
    """Optional field-to-delimiter decoder for variable-length numeric spans.

    Unlike ``reconstruct_with_skeleton``, handles numeric-field length
    mismatches without fabricating non-numeric structure:
    - If a non-numeric token (e.g., an early delimiter) appears inside an
      entity numeric span, truncates the numeric field before that token.
    - If a numeric character overflows into the boundary token immediately
      following an entity numeric span, restores the true boundary token.
    """
    if len(predicted_tokens) != len(tokenize(true_text)):
        return None
    matches = list(_TOKEN_RE.finditer(true_text))
    pieces: list[str] = []
    cursor = 0
    pred_index = 0
    prev_was_entity_num = False
    for match in matches:
        pieces.append(true_text[cursor : match.start()])
        span = match.group()
        if _NUMBER_SPAN_RE.fullmatch(span) is not None:
            width = len(span)
            chars = predicted_tokens[pred_index : pred_index + width]
            if len(chars) != width:
                return None
            prefix = true_text[: match.start()]
            is_entity_num = prefix.endswith(_ENTITY_VALUE_PREFIXES)
            if is_entity_num:
                kept: list[str] = []
                for tok in chars:
                    if tok in _NUMERIC_CHAR_OR_SIGN:
                        kept.append(tok)
                    else:
                        break
                pieces.append("".join(kept) if kept else "".join(chars))
            else:
                pieces.append("".join(chars))
            pred_index += width
            prev_was_entity_num = is_entity_num
        else:
            pred_tok = predicted_tokens[pred_index]
            if prev_was_entity_num and pred_tok in _NUMERIC_CHAR_OR_SIGN:
                pieces.append(span)
            else:
                pieces.append(pred_tok)
            pred_index += 1
            prev_was_entity_num = False
        cursor = match.end()
    if pred_index != len(predicted_tokens):
        return None
    pieces.append(true_text[cursor:])
    return "".join(pieces)
