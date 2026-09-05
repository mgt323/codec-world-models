"""Codec B — process / relational event chain (Observation -> string).

``encode_B: Observation -> str``
``parse_B: str -> FactRecord``  (PROGRAM_SPEC parse_k; official info-equality entrypoint)
``observation_from_B: str -> Observation``  (round-trip helper)

Format (per timestep; episode concat is ambient time, not this module):

    (observe A=<val>) → (co-vary B=<val>) → (meta n=<n> source=<src> noise=<bin>)

Unobserved entities use ``(miss A)`` / ``(miss B)``.

No directional ``↑`` markers at single-Observation encode time (no prior
timestep available). Noise metadata lives only under ``meta`` as ``noise=``,
never as relational ``link`` / ``strength``.

Intervention visibility (PROGRAM_SPEC option B — hidden ``do()``):
``active_intervention`` is stripped by ``observe()`` and is never encoded here.
No ``do()`` / ``is_intervened`` tokens appear in Codec B strings.
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

_ARROW = " → "
_EVENT_A_OBS = re.compile(
    r"^\(observe A=(?P<value>[^)]+)\)$"
)
_EVENT_B_OBS = re.compile(
    r"^\(co-vary B=(?P<value>[^)]+)\)$"
)
_EVENT_MISS = re.compile(r"^\(miss (?P<name>[AB])\)$")
_EVENT_META = re.compile(
    r"^\(meta n=(?P<n>\d+) source=(?P<source>\S+) "
    r"noise=(?P<noise>low|medium|high)\)$"
)


def encode_B(obs: Observation) -> str:
    """Encode one Observation as a process / relational event chain."""
    events = [
        _entity_a_event(obs.a_observed, obs.a_value),
        _entity_b_event(obs.b_observed, obs.b_value),
        (
            f"(meta n={obs.n_samples} source={obs.source.value} "
            f"noise={obs.noise_bin.value})"
        ),
    ]
    return _ARROW.join(events)


def parse_B(text: str) -> FactRecord:
    """Official Codec B parser: string -> FactRecord (PROGRAM_SPEC parse_k)."""
    return facts_from_observation(observation_from_B(text))


# Back-compat alias used by older scripts/tests.
decode_B_facts = parse_B


def observation_from_B(text: str) -> Observation:
    """Parse a Codec B event chain back to Observation."""
    events = [e.strip() for e in text.strip().split(_ARROW) if e.strip()]
    if len(events) != 3:
        raise ValueError(f"Codec B expects 3 events, got {len(events)}")

    a_obs, a_val = _parse_a_event(events[0])
    b_obs, b_val = _parse_b_event(events[1])

    meta = _EVENT_META.match(events[2])
    if meta is None:
        raise ValueError(f"Invalid Codec B meta event: {events[2]!r}")

    return Observation(
        a_observed=a_obs,
        a_value=a_val,
        b_observed=b_obs,
        b_value=b_val,
        noise_bin=NoiseBin(meta.group("noise")),
        n_samples=int(meta.group("n")),
        source=SensorSource(meta.group("source")),
    )


def _entity_a_event(observed: bool, value: float | None) -> str:
    if not observed:
        return "(miss A)"
    if value is None:
        raise ValueError("Entity A marked observed but value is None")
    return f"(observe A={_format_value(value)})"


def _entity_b_event(observed: bool, value: float | None) -> str:
    if not observed:
        return "(miss B)"
    if value is None:
        raise ValueError("Entity B marked observed but value is None")
    return f"(co-vary B={_format_value(value)})"


def _format_value(value: float) -> str:
    """Stable decimal rendering sufficient for shared quantization bins."""
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _parse_a_event(event: str) -> tuple[bool, float | None]:
    miss = _EVENT_MISS.match(event)
    if miss is not None:
        if miss.group("name") != "A":
            raise ValueError(f"Expected miss A, got {event!r}")
        return False, None
    match = _EVENT_A_OBS.match(event)
    if match is None:
        raise ValueError(f"Invalid Codec B A-event: {event!r}")
    return True, float(match.group("value"))


def _parse_b_event(event: str) -> tuple[bool, float | None]:
    miss = _EVENT_MISS.match(event)
    if miss is not None:
        if miss.group("name") != "B":
            raise ValueError(f"Expected miss B, got {event!r}")
        return False, None
    match = _EVENT_B_OBS.match(event)
    if match is None:
        raise ValueError(f"Invalid Codec B B-event: {event!r}")
    return True, float(match.group("value"))
