"""HDF5 persistence for raw State episode datasets (EXPERIMENT_PLAN.md §9).

Flat column schema mirrors ``State`` fields; ``LatentState`` is flattened to
``latent_*`` columns. Codec encoding / tokenization is out of scope here.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Iterable, Sequence
from pathlib import Path

import h5py
import numpy as np

from world.schema import (
    CausalRegime,
    Intervention,
    InterventionTarget,
    LatentState,
    NoiseBin,
    SensorSource,
    State,
)

_PICKLE_PROTOCOL = 5
_SCHEMA_V0 = "world.State.flat.v0"


def _as_str(value: object) -> str:
    """Decode HDF5 / numpy string cells to ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    # numpy bytes_ / str_ and h5py wrappers
    raw = getattr(value, "item", lambda: value)()
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _parse_oracle_posterior(raw: object) -> dict[str, float] | None:
    text = _as_str(raw).strip()
    if not text:
        return None
    out: dict[str, float] = {}
    for part in text.split(","):
        key, _, val = part.partition(":")
        if not key or not val:
            raise ValueError(f"Invalid oracle posterior cell: {text!r}")
        out[key] = float(val)
    return out


def _state_from_row(
    *,
    t: int,
    a: float,
    b: float,
    a_observed: bool,
    b_observed: bool,
    noise_bin: object,
    n_samples: int,
    source: object,
    latent_hidden_c: float,
    latent_regime: object,
    latent_oracle_regime_posterior: object,
    active_intervention_present: bool,
    active_intervention_target: object,
    active_intervention_value: float,
    active_intervention_timestep: int,
) -> State:
    intervention: Intervention | None = None
    if bool(active_intervention_present):
        ts = int(active_intervention_timestep)
        intervention = Intervention(
            target=InterventionTarget(_as_str(active_intervention_target)),
            value=float(active_intervention_value),
            timestep=None if ts < 0 else ts,
        )
    return State(
        t=int(t),
        a=float(a),
        b=float(b),
        a_observed=bool(a_observed),
        b_observed=bool(b_observed),
        noise_bin=NoiseBin(_as_str(noise_bin)),
        n_samples=int(n_samples),
        source=SensorSource(_as_str(source)),
        latent=LatentState(
            hidden_c=float(latent_hidden_c),
            regime=CausalRegime(_as_str(latent_regime)),
            oracle_regime_posterior=_parse_oracle_posterior(
                latent_oracle_regime_posterior
            ),
        ),
        active_intervention=intervention,
    )


def read_episodes_hdf5(path: Path) -> list[tuple[int, list[State]]]:
    """Load one split written by ``write_episodes_hdf5*`` as ``(seed, states)``."""
    with h5py.File(path, "r") as f:
        schema = _as_str(f.attrs.get("schema", ""))
        if schema and schema != _SCHEMA_V0:
            raise ValueError(f"Unsupported HDF5 schema {schema!r} in {path}")
        n_episodes = int(f.attrs["n_episodes"])
        n_steps = int(f.attrs["n_steps"])
        episode_seeds = f["episode_seed"][:]
        t = f["t"][:]
        a = f["a"][:]
        b = f["b"][:]
        a_observed = f["a_observed"][:]
        b_observed = f["b_observed"][:]
        noise_bin = f["noise_bin"][:]
        n_samples = f["n_samples"][:]
        source = f["source"][:]
        latent_hidden_c = f["latent_hidden_c"][:]
        latent_regime = f["latent_regime"][:]
        latent_oracle = f["latent_oracle_regime_posterior"][:]
        inter_present = f["active_intervention_present"][:]
        inter_target = f["active_intervention_target"][:]
        inter_value = f["active_intervention_value"][:]
        inter_timestep = f["active_intervention_timestep"][:]

    if len(episode_seeds) != n_episodes:
        raise ValueError(
            f"{path}: episode_seed length {len(episode_seeds)} != n_episodes={n_episodes}"
        )

    episodes: list[tuple[int, list[State]]] = []
    for i in range(n_episodes):
        states: list[State] = []
        for j in range(n_steps):
            states.append(
                _state_from_row(
                    t=int(t[i, j]),
                    a=float(a[i, j]),
                    b=float(b[i, j]),
                    a_observed=bool(a_observed[i, j]),
                    b_observed=bool(b_observed[i, j]),
                    noise_bin=noise_bin[i, j],
                    n_samples=int(n_samples[i, j]),
                    source=source[i, j],
                    latent_hidden_c=float(latent_hidden_c[i, j]),
                    latent_regime=latent_regime[i, j],
                    latent_oracle_regime_posterior=latent_oracle[i, j],
                    active_intervention_present=bool(inter_present[i, j]),
                    active_intervention_target=inter_target[i, j],
                    active_intervention_value=float(inter_value[i, j]),
                    active_intervention_timestep=int(inter_timestep[i, j]),
                )
            )
        episodes.append((int(episode_seeds[i]), states))
    return episodes


def episodes_sha256(episodes: Sequence[tuple[int, list[State]]]) -> str:
    """Canonical checksum over ``(episode_seed, states)`` in list order."""
    digest = hashlib.sha256()
    for item in episodes:
        digest.update(pickle.dumps(item, protocol=_PICKLE_PROTOCOL))
    return digest.hexdigest()


def _fill_timestep_row(
    *,
    i: int,
    j: int,
    state: State,
    t: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    a_observed: np.ndarray,
    b_observed: np.ndarray,
    noise_bin: np.ndarray,
    n_samples: np.ndarray,
    source: np.ndarray,
    latent_hidden_c: np.ndarray,
    latent_regime: np.ndarray,
    latent_oracle_regime_posterior: np.ndarray,
    active_intervention_present: np.ndarray,
    active_intervention_target: np.ndarray,
    active_intervention_value: np.ndarray,
    active_intervention_timestep: np.ndarray,
) -> None:
    t[i, j] = state.t
    a[i, j] = state.a
    b[i, j] = state.b
    a_observed[i, j] = state.a_observed
    b_observed[i, j] = state.b_observed
    noise_bin[i, j] = state.noise_bin.value
    n_samples[i, j] = state.n_samples
    source[i, j] = state.source.value
    latent_hidden_c[i, j] = state.latent.hidden_c
    latent_regime[i, j] = state.latent.regime.value
    if state.latent.oracle_regime_posterior is None:
        latent_oracle_regime_posterior[i, j] = ""
    else:
        items = sorted(state.latent.oracle_regime_posterior.items())
        latent_oracle_regime_posterior[i, j] = ",".join(
            f"{k}:{v}" for k, v in items
        )
    if state.active_intervention is None:
        active_intervention_present[i, j] = False
        active_intervention_target[i, j] = ""
        active_intervention_value[i, j] = np.nan
        active_intervention_timestep[i, j] = -1
    else:
        inter = state.active_intervention
        active_intervention_present[i, j] = True
        active_intervention_target[i, j] = inter.target.value
        active_intervention_value[i, j] = inter.value
        active_intervention_timestep[i, j] = (
            -1 if inter.timestep is None else int(inter.timestep)
        )


def write_episodes_hdf5(
    path: Path,
    episodes: Sequence[tuple[int, list[State]]],
    *,
    split_name: str,
    compression: str | None = "gzip",
) -> None:
    """Write one split to ``path`` with shape ``(n_episodes, n_steps)`` arrays."""
    if not episodes:
        raise ValueError("Cannot write empty episode list")
    write_episodes_hdf5_streaming(
        path,
        episodes,
        n_episodes=len(episodes),
        n_steps=len(episodes[0][1]),
        split_name=split_name,
        compression=compression,
    )


def write_episodes_hdf5_streaming(
    path: Path,
    episodes: Iterable[tuple[int, list[State]]],
    *,
    n_episodes: int,
    n_steps: int,
    split_name: str,
    compression: str | None = "gzip",
) -> str:
    """Stream episodes into HDF5; return canonical sha256 over the stream.

    Keeps at most one Python episode in memory beyond the destination arrays.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be > 0, got {n_episodes}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")

    episode_seeds = np.empty(n_episodes, dtype=np.int64)
    t = np.empty((n_episodes, n_steps), dtype=np.int32)
    a = np.empty((n_episodes, n_steps), dtype=np.float64)
    b = np.empty((n_episodes, n_steps), dtype=np.float64)
    a_observed = np.empty((n_episodes, n_steps), dtype=np.bool_)
    b_observed = np.empty((n_episodes, n_steps), dtype=np.bool_)
    str_dt = h5py.string_dtype(encoding="utf-8")
    noise_bin = np.empty((n_episodes, n_steps), dtype=str_dt)
    n_samples = np.empty((n_episodes, n_steps), dtype=np.int32)
    source = np.empty((n_episodes, n_steps), dtype=str_dt)
    latent_hidden_c = np.empty((n_episodes, n_steps), dtype=np.float64)
    latent_regime = np.empty((n_episodes, n_steps), dtype=str_dt)
    latent_oracle_regime_posterior = np.empty((n_episodes, n_steps), dtype=str_dt)
    active_intervention_present = np.empty((n_episodes, n_steps), dtype=np.bool_)
    active_intervention_target = np.empty((n_episodes, n_steps), dtype=str_dt)
    active_intervention_value = np.empty((n_episodes, n_steps), dtype=np.float64)
    active_intervention_timestep = np.empty((n_episodes, n_steps), dtype=np.int32)

    digest = hashlib.sha256()
    count = 0
    for seed, states in episodes:
        if count >= n_episodes:
            raise ValueError(
                f"Received more than n_episodes={n_episodes} from episode stream"
            )
        if len(states) != n_steps:
            raise ValueError(
                f"Episode seed={seed} has {len(states)} steps; expected {n_steps}"
            )
        digest.update(pickle.dumps((seed, states), protocol=_PICKLE_PROTOCOL))
        episode_seeds[count] = seed
        for j, state in enumerate(states):
            _fill_timestep_row(
                i=count,
                j=j,
                state=state,
                t=t,
                a=a,
                b=b,
                a_observed=a_observed,
                b_observed=b_observed,
                noise_bin=noise_bin,
                n_samples=n_samples,
                source=source,
                latent_hidden_c=latent_hidden_c,
                latent_regime=latent_regime,
                latent_oracle_regime_posterior=latent_oracle_regime_posterior,
                active_intervention_present=active_intervention_present,
                active_intervention_target=active_intervention_target,
                active_intervention_value=active_intervention_value,
                active_intervention_timestep=active_intervention_timestep,
            )
        count += 1

    if count != n_episodes:
        raise ValueError(
            f"Episode stream ended early: got {count}, expected {n_episodes}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"compression": compression} if compression else {}
    with h5py.File(path, "w") as f:
        f.attrs["split_name"] = split_name
        f.attrs["n_episodes"] = n_episodes
        f.attrs["n_steps"] = n_steps
        f.attrs["schema"] = "world.State.flat.v0"
        f.create_dataset("episode_seed", data=episode_seeds, **kw)
        f.create_dataset("t", data=t, **kw)
        f.create_dataset("a", data=a, **kw)
        f.create_dataset("b", data=b, **kw)
        f.create_dataset("a_observed", data=a_observed, **kw)
        f.create_dataset("b_observed", data=b_observed, **kw)
        f.create_dataset("noise_bin", data=noise_bin, **kw)
        f.create_dataset("n_samples", data=n_samples, **kw)
        f.create_dataset("source", data=source, **kw)
        f.create_dataset("latent_hidden_c", data=latent_hidden_c, **kw)
        f.create_dataset("latent_regime", data=latent_regime, **kw)
        f.create_dataset(
            "latent_oracle_regime_posterior",
            data=latent_oracle_regime_posterior,
            **kw,
        )
        f.create_dataset(
            "active_intervention_present", data=active_intervention_present, **kw
        )
        f.create_dataset(
            "active_intervention_target", data=active_intervention_target, **kw
        )
        f.create_dataset(
            "active_intervention_value", data=active_intervention_value, **kw
        )
        f.create_dataset(
            "active_intervention_timestep",
            data=active_intervention_timestep,
            **kw,
        )
    return digest.hexdigest()
