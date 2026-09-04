"""Reusable information-parity audit for observation codecs.

The runner is codec-agnostic: callers inject ``encode`` and ``decode_facts``.
Codec-specific fact decoders live next to each codec module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from world.schema import (
    CausalRegime,
    FactRecord,
    Intervention,
    InterventionTarget,
    LatentState,
    NoiseBin,
    Observation,
    SensorSource,
    State,
    facts_from_observation,
    observe,
)

EncodeFn = Callable[[Observation], str]
DecodeFactsFn = Callable[[str], FactRecord]


@dataclass(frozen=True, slots=True)
class ParityMismatch:
    """One failed parity case with full detail for debugging."""

    index: int
    state: State
    observation: Observation
    encoded: str
    expected: FactRecord
    recovered: FactRecord | None
    error: str | None

    def to_detail_dict(self) -> dict[str, object]:
        """JSON-serializable detail blob (enums as values)."""
        return {
            "index": self.index,
            "state": _state_as_dict(self.state),
            "observation": _observation_as_dict(self.observation),
            "encoded": self.encoded,
            "expected": _fact_as_dict(self.expected),
            "recovered": None if self.recovered is None else _fact_as_dict(self.recovered),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ParityAuditReport:
    """Aggregate parity result for a batch of states."""

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
    states: Sequence[State],
    *,
    encode: EncodeFn,
    decode_facts: DecodeFactsFn,
) -> ParityAuditReport:
    """Audit encode→decode fact recovery against ``facts_from_observation``.

    For each state:
      1. ``obs = observe(state)``
      2. ``text = encode(obs)``
      3. ``recovered = decode_facts(text)``
      4. require ``recovered == facts_from_observation(obs)``
    """
    mismatches: list[ParityMismatch] = []
    n_passed = 0

    for index, state in enumerate(states):
        obs = observe(state)
        expected = facts_from_observation(obs)
        encoded = ""
        recovered: FactRecord | None = None
        error: str | None = None
        try:
            encoded = encode(obs)
            recovered = decode_facts(encoded)
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
                    state=state,
                    observation=obs,
                    encoded=encoded,
                    expected=expected,
                    recovered=recovered,
                    error=error,
                )
            )

    return ParityAuditReport(
        n_total=len(states),
        n_passed=n_passed,
        mismatches=tuple(mismatches),
    )


def sample_parity_states() -> list[State]:
    """Deterministic batch covering regimes, noise extremes, and partial obs.

    Coverage:
    - all 4 ``CausalRegime`` values
    - ``NoiseBin.LOW`` and ``NoiseBin.HIGH``
    - ``a_observed=False`` and ``b_observed=False`` cases
    """
    states: list[State] = []
    t = 0
    for regime in CausalRegime:
        for noise in (NoiseBin.LOW, NoiseBin.HIGH):
            for a_obs, b_obs in ((True, True), (False, True), (True, False)):
                source = _source_for_masks(a_obs, b_obs)
                states.append(
                    State(
                        t=t,
                        a=0.41,
                        b=0.38,
                        a_observed=a_obs,
                        b_observed=b_obs,
                        noise_bin=noise,
                        n_samples=3,
                        source=source,
                        latent=LatentState(hidden_c=0.5, regime=regime),
                        active_intervention=None,
                    )
                )
                t += 1
    return states


def sample_diverse_parity_states(target_n: int = 500) -> list[State]:
    """Build a diverse ~``target_n`` State batch for large parity audits.

    Guaranteed coverage (within the first covering block, then padded):
    - all 4 ``CausalRegime`` values
    - ``NoiseBin.LOW`` and ``NoiseBin.HIGH``
    - partial-obs: a_only, b_only, both
    - ``n_samples=1`` edge cases
    - ``active_intervention`` set and ``None``
    """
    value_grid = (
        (0.0, 0.0),
        (0.05, 0.95),
        (0.1, 0.2),
        (0.3, 0.3),
        (0.41, 0.38),
        (0.5, 0.5),
        (0.66, 0.25),
        (0.75, 0.8),
        (0.99, 0.01),
        (1.0, 1.0),
        (-0.1, 1.2),
        (0.22, 0.78),
    )
    hidden_cs = (0.0, 0.25, 0.5, 0.75, 1.0)
    n_samples_choices = (1, 3, 5)
    partial_masks = (
        (True, True),   # both
        (True, False),  # a_only
        (False, True),  # b_only
    )
    intervention_plans: tuple[Intervention | None, ...] = (
        None,
        Intervention(target=InterventionTarget.A, value=0.9, timestep=0),
        Intervention(target=InterventionTarget.B, value=0.1, timestep=0),
    )
    noises = (NoiseBin.LOW, NoiseBin.HIGH)
    regimes = tuple(CausalRegime)

    # Round-robin over required axes so early truncation still covers all factors.
    combos: list[tuple[object, ...]] = []
    for regime in regimes:
        for noise in noises:
            for a_obs, b_obs in partial_masks:
                for n_samples in n_samples_choices:
                    for intervention in intervention_plans:
                        combos.append((regime, noise, a_obs, b_obs, n_samples, intervention))

    states: list[State] = []
    t = 0
    vi = 0
    hi = 0
    while len(states) < target_n:
        for regime, noise, a_obs, b_obs, n_samples, intervention in combos:
            if len(states) >= target_n:
                break
            a_val, b_val = value_grid[vi % len(value_grid)]
            hidden_c = hidden_cs[hi % len(hidden_cs)]
            vi += 1
            hi += 1
            states.append(
                State(
                    t=t,
                    a=float(a_val),
                    b=float(b_val),
                    a_observed=bool(a_obs),
                    b_observed=bool(b_obs),
                    noise_bin=noise,  # type: ignore[arg-type]
                    n_samples=int(n_samples),
                    source=_source_for_masks(bool(a_obs), bool(b_obs)),
                    latent=LatentState(
                        hidden_c=float(hidden_c),
                        regime=regime,  # type: ignore[arg-type]
                    ),
                    active_intervention=intervention,  # type: ignore[arg-type]
                )
            )
            t += 1
    return states


def _source_for_masks(a_observed: bool, b_observed: bool) -> SensorSource:
    if a_observed and b_observed:
        return SensorSource.SENSOR_BOTH
    if a_observed:
        return SensorSource.SENSOR_A
    if b_observed:
        return SensorSource.SENSOR_B
    return SensorSource.SENSOR_NONE


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


def _state_as_dict(state: State) -> dict[str, object]:
    return {
        "t": state.t,
        "a": state.a,
        "b": state.b,
        "a_observed": state.a_observed,
        "b_observed": state.b_observed,
        "noise_bin": state.noise_bin.value,
        "n_samples": state.n_samples,
        "source": state.source.value,
        "latent": {
            "hidden_c": state.latent.hidden_c,
            "regime": state.latent.regime.value,
            "oracle_regime_posterior": (
                None
                if state.latent.oracle_regime_posterior is None
                else dict(state.latent.oracle_regime_posterior)
            ),
        },
        "active_intervention": (
            None
            if state.active_intervention is None
            else {
                "target": state.active_intervention.target.value,
                "value": state.active_intervention.value,
                "timestep": state.active_intervention.timestep,
            }
        ),
    }
