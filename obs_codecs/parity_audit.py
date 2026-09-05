"""Reusable information-parity audit for observation codecs.

Codec-agnostic: callers inject ``encode`` and ``parse`` (string -> FactRecord).
Accepts ``Observation`` only — never simulator full-state or hidden-state types
(PROGRAM_SPEC §4.1: obs_codecs must not depend on those).

State sampling for audits lives in ``world.parity_fixtures``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from world.schema import FactRecord, Observation, facts_from_observation

EncodeFn = Callable[[Observation], str]
ParseFactsFn = Callable[[str], FactRecord]


@dataclass(frozen=True, slots=True)
class ParityMismatch:
    """One failed parity case with full detail for debugging."""

    index: int
    observation: Observation
    encoded: str
    expected: FactRecord
    recovered: FactRecord | None
    error: str | None

    def to_detail_dict(self) -> dict[str, object]:
        """JSON-serializable detail blob (enums as values)."""
        return {
            "index": self.index,
            "observation": _observation_as_dict(self.observation),
            "encoded": self.encoded,
            "expected": _fact_as_dict(self.expected),
            "recovered": None if self.recovered is None else _fact_as_dict(self.recovered),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ParityAuditReport:
    """Aggregate parity result for a batch of observations."""

    n_total: int
    n_passed: int
    mismatches: tuple[ParityMismatch, ...]

    @property
    def pass_rate(self) -> float:
        if self.n_total == 0:
            return 1.0
        return self.n_passed / self.n_total

    @property
    def ok(self) -> bool:
        return self.n_passed == self.n_total and len(self.mismatches) == 0

    def summary(self) -> str:
        rate_pct = 100.0 * self.pass_rate
        lines = [
            f"parity_audit: {self.n_passed}/{self.n_total} passed "
            f"({rate_pct:.1f}%), mismatches={len(self.mismatches)}"
        ]
        for m in self.mismatches:
            lines.append(
                f"  [{m.index}] expected={m.expected!r} recovered={m.recovered!r} "
                f"error={m.error!r} encoded={m.encoded!r}"
            )
        return "\n".join(lines)


def run_parity_audit(
    observations: Sequence[Observation],
    *,
    encode: EncodeFn,
    parse: ParseFactsFn,
) -> ParityAuditReport:
    """Audit encode→parse fact recovery against ``facts_from_observation``.

    For each observation:
      1. ``text = encode(obs)``
      2. ``recovered = parse(text)``  # FactRecord (PROGRAM_SPEC parse_k)
      3. require ``recovered == facts_from_observation(obs)``
    """
    mismatches: list[ParityMismatch] = []
    n_passed = 0

    for index, obs in enumerate(observations):
        expected = facts_from_observation(obs)
        encoded = ""
        recovered: FactRecord | None = None
        error: str | None = None
        try:
            encoded = encode(obs)
            recovered = parse(encoded)
            if recovered != expected:
                error = "fact_mismatch"
        except Exception as exc:  # noqa: BLE001 — audit must record any codec failure
            error = f"{type(exc).__name__}: {exc}"

        if error is None:
            n_passed += 1
        else:
            mismatches.append(
                ParityMismatch(
                    index=index,
                    observation=obs,
                    encoded=encoded,
                    expected=expected,
                    recovered=recovered,
                    error=error,
                )
            )

    return ParityAuditReport(
        n_total=len(observations),
        n_passed=n_passed,
        mismatches=tuple(mismatches),
    )


def _fact_as_dict(facts: FactRecord) -> dict[str, object]:
    raw = asdict(facts)
    raw["noise_bin"] = facts.noise_bin.value
    raw["source"] = facts.source.value
    return raw


def _observation_as_dict(obs: Observation) -> dict[str, object]:
    raw = asdict(obs)
    raw["noise_bin"] = obs.noise_bin.value
    raw["source"] = obs.source.value
    return raw
