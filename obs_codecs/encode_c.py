"""Codec C — evidential observational record (Observation -> string).

``encode_C: Observation -> str``
``parse_C: str -> FactRecord``  (PROGRAM_SPEC parse_k; official info-equality entrypoint)
``observation_from_C: str -> Observation``  (round-trip helper; alts line discarded for F)

Format (per timestep; episode concat is ambient time, not this module):

    obs: A=<val|?>, B=<val|?>
    source: <sensor_source>
    n: <n_samples>
    noise: <low|medium|high>
    alts: {common_cause, a_causes_b, b_causes_a, spurious}

Unobserved entities use ``?`` — a deliberate C-specific rendering taken from
the plan's own C example, not Codec A's ``none``.

``alts`` is a fixed, value-independent constant: the same four hypothesis
labels in the same order in every encoding, regardless of Observation content.
It carries zero bits about the state — no oracle ``p``, no true regime (I8) —
and is excluded from ``FactRecord`` / info-equality exactly like Codec A's
Relation line. The parser accepts only the exact literal so later drift in the
constant cannot pass silently.

The four labels are spelled to match ``CausalRegime`` values (declaration
order), which deliberately shares vocabulary with the regime-classification
eval labels; the strings are hardcoded here because ``obs_codecs`` must not
import latent/eval enums (PROGRAM_SPEC §4.1). The fourth label is ``spurious``
rather than the plan example's ``noise``: that word already names the
measurement-noise field in the same block, so reusing it would bind two
unrelated concepts to one surface token.

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

_MISSING = "?"
_ALTS_LINE = "alts: {common_cause, a_causes_b, b_causes_a, spurious}"

_OBS_RE = re.compile(r"^obs: A=(?P<a>[^,]+), B=(?P<b>.+)$")
_SOURCE_RE = re.compile(r"^source: (?P<source>\S+)$")
_N_RE = re.compile(r"^n: (?P<n>\d+)$")
_NOISE_RE = re.compile(r"^noise: (?P<noise>low|medium|high)$")


def encode_C(obs: Observation) -> str:
    """Encode one Observation as an evidential observational record."""
    lines = [
        (
            f"obs: A={_value_field('A', obs.a_observed, obs.a_value)}, "
            f"B={_value_field('B', obs.b_observed, obs.b_value)}"
        ),
        f"source: {obs.source.value}",
        f"n: {obs.n_samples}",
        f"noise: {obs.noise_bin.value}",
        _ALTS_LINE,
    ]
    return "\n".join(lines)


def parse_C(text: str) -> FactRecord:
    """Official Codec C parser: string -> FactRecord (PROGRAM_SPEC parse_k)."""
    return facts_from_observation(observation_from_C(text))


# Back-compat alias matching the A/B parity adapters.
decode_C_facts = parse_C


def observation_from_C(text: str) -> Observation:
    """Parse a Codec C record back to Observation (alts line validated, not in F)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) != 5:
        raise ValueError(f"Codec C expects 5 lines, got {len(lines)}")

    obs_line = _OBS_RE.match(lines[0])
    if obs_line is None:
        raise ValueError(f"Invalid Codec C obs line: {lines[0]!r}")
    source_line = _SOURCE_RE.match(lines[1])
    if source_line is None:
        raise ValueError(f"Invalid Codec C source line: {lines[1]!r}")
    n_line = _N_RE.match(lines[2])
    if n_line is None:
        raise ValueError(f"Invalid Codec C n line: {lines[2]!r}")
    noise_line = _NOISE_RE.match(lines[3])
    if noise_line is None:
        raise ValueError(f"Invalid Codec C noise line: {lines[3]!r}")
    if lines[4] != _ALTS_LINE:
        raise ValueError(f"Invalid Codec C alts line: {lines[4]!r}")

    a_observed, a_value = _parse_value_field("A", obs_line.group("a"))
    b_observed, b_value = _parse_value_field("B", obs_line.group("b"))

    return Observation(
        a_observed=a_observed,
        a_value=a_value,
        b_observed=b_observed,
        b_value=b_value,
        noise_bin=NoiseBin(noise_line.group("noise")),
        n_samples=int(n_line.group("n")),
        source=SensorSource(source_line.group("source")),
    )


def _value_field(name: str, observed: bool, value: float | None) -> str:
    if not observed:
        return _MISSING
    if value is None:
        raise ValueError(f"Entity {name} marked observed but value is None")
    return _format_value(value)


def _format_value(value: float) -> str:
    """Stable decimal rendering sufficient for shared quantization bins."""
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _parse_value_field(name: str, raw: str) -> tuple[bool, float | None]:
    if raw == _MISSING:
        return False, None
    try:
        return True, float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid Codec C value for entity {name}: {raw!r}") from exc
