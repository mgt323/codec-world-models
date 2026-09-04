"""World type schema: State, LatentState, Observation, and Observer.

Enforces PROGRAM_SPEC type layering: codecs consume Observation only;
LatentState is stripped by observe() and must never enter encode_*.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping

# --- Shared quantization (info-equality F uses the same bins for all codecs) ---

VALUE_BIN_EDGES: tuple[float, ...] = tuple(i / 10.0 for i in range(11))
"""Edges for quantizing continuous observable values into shared bins [0, 1]."""

SCHEMA_VERSION: str = "world.schema.v0"


class CausalRegime(str, Enum):
    """True generative regime (latent / eval only — never part of Observation)."""

    COMMON_CAUSE = "common_cause"
    A_CAUSES_B = "a_causes_b"
    B_CAUSES_A = "b_causes_a"
    SPURIOUS = "spurious"


class NoiseBin(str, Enum):
    """Discrete noise level exposed in Observation (shared across codecs)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SensorSource(str, Enum):
    """Which sensors contributed to this observation."""

    SENSOR_A = "sensor_a"
    SENSOR_B = "sensor_b"
    SENSOR_BOTH = "sensor_both"
    SENSOR_NONE = "sensor_none"


class InterventionTarget(str, Enum):
    A = "A"
    B = "B"


@dataclass(frozen=True, slots=True)
class Difficulty:
    """Difficulty knobs for World v0 (§2.1). Simulator consumes these later."""

    noise_scale: float = 0.1
    regime_switch_rate: float = 0.0
    partial_obs_rate: float = 0.0
    confounding_strength: float = 0.0


@dataclass(frozen=True, slots=True)
class Intervention:
    """do(target=value) applied at a timestep (simulator / eval)."""

    target: InterventionTarget
    value: float
    timestep: int | None = None


@dataclass(frozen=True, slots=True)
class LatentState:
    """Hidden state H_t — simulator, evaluator, and probe targets only.

    Must never be passed to codecs or stage-1 training.
    """

    hidden_c: float
    regime: CausalRegime
    oracle_regime_posterior: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class State:
    """Full simulator state at one timestep (observables + latents)."""

    t: int
    a: float
    b: float
    a_observed: bool
    b_observed: bool
    noise_bin: NoiseBin
    n_samples: int
    source: SensorSource
    latent: LatentState
    active_intervention: Intervention | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    """O_t = Obs(S_t). Sole legal codec input (I1).

    Contains no regime label, hidden_c, or oracle posteriors (I2, I8).
    """

    a_observed: bool
    a_value: float | None
    b_observed: bool
    b_value: float | None
    noise_bin: NoiseBin
    n_samples: int
    source: SensorSource


@dataclass(frozen=True, slots=True)
class FactRecord:
    """Canonical recoverable facts F(O) for info-equality (§1.1)."""

    a_obs: bool
    a_val_q: int | None
    b_obs: bool
    b_val_q: int | None
    noise_bin: NoiseBin
    n_samples: int
    source: SensorSource


def quantize_value(value: float, edges: tuple[float, ...] = VALUE_BIN_EDGES) -> int:
    """Map a continuous value to a shared bin index using right-open edges.

    Values below the first edge map to 0; values at/above the last edge map
    to len(edges) - 2 (last bin).
    """
    if len(edges) < 2:
        raise ValueError("edges must contain at least two values")
    for i in range(len(edges) - 1):
        if value < edges[i + 1]:
            return i
    return len(edges) - 2


def observe(state: State) -> Observation:
    """Pure projection State -> Observation. Strips LatentState (PROGRAM_SPEC)."""
    return Observation(
        a_observed=state.a_observed,
        a_value=state.a if state.a_observed else None,
        b_observed=state.b_observed,
        b_value=state.b if state.b_observed else None,
        noise_bin=state.noise_bin,
        n_samples=state.n_samples,
        source=state.source,
    )


def facts_from_observation(obs: Observation) -> FactRecord:
    """Build quantized FactRecord F(O) with shared binning for all codecs."""
    return FactRecord(
        a_obs=obs.a_observed,
        a_val_q=quantize_value(obs.a_value) if obs.a_observed and obs.a_value is not None else None,
        b_obs=obs.b_observed,
        b_val_q=quantize_value(obs.b_value) if obs.b_observed and obs.b_value is not None else None,
        noise_bin=obs.noise_bin,
        n_samples=obs.n_samples,
        source=obs.source,
    )


def observation_field_names() -> tuple[str, ...]:
    """Stable ordered field names of Observation (schema checksum input)."""
    return tuple(Observation.__dataclass_fields__.keys())


def schema_checksum() -> str:
    """Checksum over schema version and Observation field names."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "observation_fields": list(observation_field_names()),
        "value_bin_edges": list(VALUE_BIN_EDGES),
        "fact_fields": list(FactRecord.__dataclass_fields__.keys()),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def observation_as_dict(obs: Observation) -> dict[str, object]:
    """JSON-friendly Observation dict (enums as values)."""
    raw = asdict(obs)
    raw["noise_bin"] = obs.noise_bin.value
    raw["source"] = obs.source.value
    return raw
