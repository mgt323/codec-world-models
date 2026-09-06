"""Codec B — process / relational event chain (Observation -> string).

``encode_B: Observation -> str``
``parse_B: str -> FactRecord``  (PROGRAM_SPEC parse_k; official info-equality entrypoint)
``observation_from_B: str -> Observation``  (position-based round-trip helper)
``observation_from_B_unordered: str -> Observation``  (role-matched; order-agnostic)

Format (per timestep; episode concat is ambient time, not this module):

    (observe A=<val>) → (co-vary B=<val>) → (meta n=<n> source=<src> noise=<bin>)

Unobserved entities use ``(miss A)`` / ``(miss B)``.

No directional ``↑`` markers at single-Observation encode time (no prior
timestep available). Noise metadata lives only under ``meta`` as ``noise=``,
never as relational ``link`` / ``strength``.

``observation_from_B`` keeps fixed event indices (A, B, meta) for existing
callers. ``observation_from_B_unordered`` classifies each event by regex role
so E-transforms (shuffle / reverse / bag) can recover the same Observation
without assuming encode order. It accepts either the process arrow
(``ARROW_SEP``) or the bag separator (``BAG_SEP``).

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
ARROW_SEP = _ARROW
# Neutral join used by transforms_e.bag_b — no directional cue; events never
# contain this substring, so bag strings remain unambiguously re-split.
BAG_SEP = " | "

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


def observation_from_B_unordered(text: str) -> Observation:
    """Parse Codec B events by role match, ignoring event order.

    Splits on ``ARROW_SEP`` when present, otherwise on ``BAG_SEP`` (bag_b
    output). Each token is classified against all known event patterns;
    slots are filled by role, not by index. Requires exactly one A-event,
    one B-event, and one meta event; rejects unrecognized tokens.
    """
    events = split_b_events(text)
    if not events:
        raise ValueError("Codec B unordered parser found no events")

    a_slots: list[tuple[bool, float | None]] = []
    b_slots: list[tuple[bool, float | None]] = []
    meta_slots: list[re.Match[str]] = []

    for event in events:
        classified = _classify_b_event(event)
        role = classified[0]
        if role == "A":
            a_slots.append(classified[1])
        elif role == "B":
            b_slots.append(classified[1])
        elif role == "meta":
            meta_slots.append(classified[1])
        else:
            raise ValueError(f"Unrecognized Codec B event token: {event!r}")

    if len(a_slots) != 1:
        raise ValueError(
            f"Codec B unordered parser expects exactly one A-event, "
            f"found {len(a_slots)}"
        )
    if len(b_slots) != 1:
        raise ValueError(
            f"Codec B unordered parser expects exactly one B-event, "
            f"found {len(b_slots)}"
        )
    if len(meta_slots) != 1:
        raise ValueError(
            f"Codec B unordered parser expects exactly one meta event, "
            f"found {len(meta_slots)}"
        )

    a_obs, a_val = a_slots[0]
    b_obs, b_val = b_slots[0]
    meta = meta_slots[0]
    return Observation(
        a_observed=a_obs,
        a_value=a_val,
        b_observed=b_obs,
        b_value=b_val,
        noise_bin=NoiseBin(meta.group("noise")),
        n_samples=int(meta.group("n")),
        source=SensorSource(meta.group("source")),
    )


def split_b_events(text: str) -> list[str]:
    """Split a Codec B (or B-E) string into event tokens.

    Prefers ``ARROW_SEP`` when present (natural / shuffled / reversed B);
    otherwise splits on ``BAG_SEP`` (bag_b). A single event with neither
    separator is returned as a one-element list.
    """
    stripped = text.strip()
    if not stripped:
        return []
    if ARROW_SEP in stripped:
        return [e.strip() for e in stripped.split(ARROW_SEP) if e.strip()]
    if BAG_SEP in stripped:
        return [e.strip() for e in stripped.split(BAG_SEP) if e.strip()]
    return [stripped]


def decode_B_facts_unordered(text: str) -> FactRecord:
    """Unordered Codec B adapter: string -> FactRecord (role-matched parse)."""
    return facts_from_observation(observation_from_B_unordered(text))


def _classify_b_event(
    event: str,
) -> (
    tuple[str, tuple[bool, float | None]]
    | tuple[str, re.Match[str]]
    | tuple[str, None]
):
    """Return (role, payload) for a single event token, or ('unknown', None)."""
    miss = _EVENT_MISS.match(event)
    if miss is not None:
        name = miss.group("name")
        if name == "A":
            return ("A", (False, None))
        if name == "B":
            return ("B", (False, None))
        return ("unknown", None)

    a_match = _EVENT_A_OBS.match(event)
    if a_match is not None:
        return ("A", (True, float(a_match.group("value"))))

    b_match = _EVENT_B_OBS.match(event)
    if b_match is not None:
        return ("B", (True, float(b_match.group("value"))))

    meta = _EVENT_META.match(event)
    if meta is not None:
        return ("meta", meta)

    return ("unknown", None)


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
