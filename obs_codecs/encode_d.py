"""Codec D — canonical structured baseline (Observation -> string).

``encode_D: Observation -> str``
``parse_D: str -> FactRecord``  (PROGRAM_SPEC parse_k; official info-equality entrypoint)
``observation_from_D: str -> Observation``  (round-trip helper)

Format (single flat line; fixed key order; semicolon-separated):

    A.x=<val|None>; A.obs=<0|1>; B.x=<val|None>; B.obs=<0|1>; noise=<bin>; n=<n>; source=<src>

D is the lean structured control for “structure beats prose” (EXPERIMENT_PLAN
§3). It carries only FactRecord-relevant fields — no entity labels, Relation
line, event arrows, evidential ``alts``, or other zero-bit decoration.

Unobserved continuous values use the literal token ``None`` (Python-style,
maximally neutral). This is a deliberate Codec D choice: the plan example
shows only observed floats and does not lock a null glyph the way Codec C
locks ``?``. ``None`` is preferred over A's prose ``none`` and C's ``?`` so
D stays free of narrative / evidential surface cues.

Intervention visibility (PROGRAM_SPEC option B — hidden ``do()``):
``active_intervention`` is stripped by ``observe()`` and is never encoded here.
"""

from __future__ import annotations

import re

from world.schema import (
    FactRecord,
    NoiseBin,
    Observation,
    SensorSource,
    facts_from_observation,
)

_MISSING = "None"
_FIELD_SEP = "; "
_EXPECTED_KEYS = (
    "A.x",
    "A.obs",
    "B.x",
    "B.obs",
    "noise",
    "n",
    "source",
)

_PAIR_RE = re.compile(r"^(?P<key>[A-Za-z.]+)=(?P<value>.*)$")
_OBS_FLAG_RE = re.compile(r"^[01]$")
_N_RE = re.compile(r"^\d+$")
_NOISE_RE = re.compile(r"^(?:low|medium|high)$")


def encode_D(obs: Observation) -> str:
    """Encode one Observation as a flat structured key=value record."""
    parts = [
        f"A.x={_value_token(obs.a_observed, obs.a_value)}",
        f"A.obs={1 if obs.a_observed else 0}",
        f"B.x={_value_token(obs.b_observed, obs.b_value)}",
        f"B.obs={1 if obs.b_observed else 0}",
        f"noise={obs.noise_bin.value}",
        f"n={obs.n_samples}",
        f"source={obs.source.value}",
    ]
    return _FIELD_SEP.join(parts)


def parse_D(text: str) -> FactRecord:
    """Official Codec D parser: string -> FactRecord (PROGRAM_SPEC parse_k)."""
    return facts_from_observation(observation_from_D(text))


# Back-compat alias matching the A/B/C parity adapters.
decode_D_facts = parse_D


def observation_from_D(text: str) -> Observation:
    """Parse a Codec D structured line back to Observation."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Codec D expects a non-empty record")
    if "\n" in stripped:
        raise ValueError("Codec D expects a single line")

    parts = stripped.split(_FIELD_SEP)
    if len(parts) != len(_EXPECTED_KEYS):
        raise ValueError(
            f"Codec D expects {len(_EXPECTED_KEYS)} fields separated by "
            f"{_FIELD_SEP!r}, got {len(parts)}"
        )

    fields: dict[str, str] = {}
    for expected_key, part in zip(_EXPECTED_KEYS, parts, strict=True):
        match = _PAIR_RE.match(part)
        if match is None:
            raise ValueError(f"Invalid Codec D field: {part!r}")
        key = match.group("key")
        if key != expected_key:
            raise ValueError(
                f"Codec D expected key {expected_key!r} at this position, "
                f"got {key!r}"
            )
        fields[key] = match.group("value")

    a_obs = _parse_obs_flag("A.obs", fields["A.obs"])
    b_obs = _parse_obs_flag("B.obs", fields["B.obs"])
    a_val = _parse_value("A.x", fields["A.x"], observed=a_obs)
    b_val = _parse_value("B.x", fields["B.x"], observed=b_obs)

    if not _NOISE_RE.match(fields["noise"]):
        raise ValueError(f"Invalid Codec D noise: {fields['noise']!r}")
    if not _N_RE.match(fields["n"]):
        raise ValueError(f"Invalid Codec D n: {fields['n']!r}")

    return Observation(
        a_observed=a_obs,
        a_value=a_val,
        b_observed=b_obs,
        b_value=b_val,
        noise_bin=NoiseBin(fields["noise"]),
        n_samples=int(fields["n"]),
        source=SensorSource(fields["source"]),
    )


def _value_token(observed: bool, value: float | None) -> str:
    if not observed:
        return _MISSING
    if value is None:
        raise ValueError("Observed entity marked observed but value is None")
    return _format_value(value)


def _format_value(value: float) -> str:
    """Stable decimal rendering sufficient for shared quantization bins."""
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _parse_obs_flag(key: str, raw: str) -> bool:
    if _OBS_FLAG_RE.match(raw) is None:
        raise ValueError(f"Invalid Codec D {key}: {raw!r}")
    return raw == "1"


def _parse_value(key: str, raw: str, *, observed: bool) -> float | None:
    if not observed:
        if raw != _MISSING:
            raise ValueError(
                f"Codec D {key} must be {_MISSING!r} when unobserved, got {raw!r}"
            )
        return None
    if raw == _MISSING:
        raise ValueError(f"Codec D {key} is {_MISSING!r} but entity is observed")
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid Codec D {key} value: {raw!r}") from exc
