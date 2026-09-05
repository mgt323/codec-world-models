"""Codec A — substance / essentialist captions (Observation -> string).

``encode_A: Observation -> str``
``parse_A: str -> FactRecord``  (PROGRAM_SPEC parse_k; official info-equality entrypoint)
``observation_from_A: str -> Observation``  (round-trip helper; Relation line discarded for F)

Python package name is ``obs_codecs`` (not ``codecs``) to avoid shadowing
the stdlib ``codecs`` module. Matches EXPERIMENT_PLAN / PROGRAM_SPEC layout.
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

_ENTITY_RE = re.compile(
    r"^Entity (?P<name>[AB]): value=(?P<value>[^,]+), "
    r"status=(?P<status>present|absent), label=(?P<label>[^.]+)\.$"
)
_CONTEXT_RE = re.compile(
    r"^Context: noise=(?P<noise>[^,]+), n_samples=(?P<n>\d+), "
    r"source=(?P<source>[^.]+)\.$"
)
_RELATION_LINE = "Relation: A and B are similar."


def encode_A(obs: Observation) -> str:
    """Encode one Observation as an essentialist caption string."""
    lines = [
        _entity_line("A", obs.a_observed, obs.a_value, label="source-like"),
        _entity_line("B", obs.b_observed, obs.b_value, label="target-like"),
        (
            f"Context: noise={obs.noise_bin.value}, "
            f"n_samples={obs.n_samples}, source={obs.source.value}."
        ),
        _RELATION_LINE,
    ]
    return "\n".join(lines)


def parse_A(text: str) -> FactRecord:
    """Official Codec A parser: string -> FactRecord (PROGRAM_SPEC parse_k)."""
    return facts_from_observation(observation_from_A(text))


# Back-compat alias used by older scripts/tests.
decode_A_facts = parse_A


def observation_from_A(text: str) -> Observation:
    """Parse a Codec A caption back to Observation (Relation line validated, not in F)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) != 4:
        raise ValueError(f"Codec A expects 4 lines, got {len(lines)}")

    a = _parse_entity(lines[0], expected="A")
    b = _parse_entity(lines[1], expected="B")
    ctx = _CONTEXT_RE.match(lines[2])
    if ctx is None:
        raise ValueError(f"Invalid Codec A context line: {lines[2]!r}")
    if lines[3] != _RELATION_LINE:
        raise ValueError(f"Invalid Codec A relation line: {lines[3]!r}")

    a_obs, a_val = bool(a["observed"]), a["value"]
    b_obs, b_val = bool(b["observed"]), b["value"]
    if a_val is not None and not isinstance(a_val, float):
        raise TypeError("Entity A value must be float or None")
    if b_val is not None and not isinstance(b_val, float):
        raise TypeError("Entity B value must be float or None")

    return Observation(
        a_observed=a_obs,
        a_value=a_val,
        b_observed=b_obs,
        b_value=b_val,
        noise_bin=NoiseBin(ctx.group("noise")),
        n_samples=int(ctx.group("n")),
        source=SensorSource(ctx.group("source")),
    )


def _entity_line(
    name: str,
    observed: bool,
    value: float | None,
    *,
    label: str,
) -> str:
    if observed:
        if value is None:
            raise ValueError(f"Entity {name} marked observed but value is None")
        value_str = _format_value(value)
        status = "present"
    else:
        value_str = "none"
        status = "absent"
    return f"Entity {name}: value={value_str}, status={status}, label={label}."


def _format_value(value: float) -> str:
    """Stable decimal rendering sufficient for shared quantization bins."""
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _parse_entity(line: str, *, expected: str) -> dict[str, object]:
    match = _ENTITY_RE.match(line)
    if match is None:
        raise ValueError(f"Invalid Codec A entity line: {line!r}")
    if match.group("name") != expected:
        raise ValueError(f"Expected Entity {expected}, got {match.group('name')}")

    status = match.group("status")
    raw_value = match.group("value")
    if status == "absent":
        return {"observed": False, "value": None}
    if raw_value == "none":
        raise ValueError(f"Entity {expected} present but value=none")
    return {"observed": True, "value": float(raw_value)}
