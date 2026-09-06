"""Locked v0 dataset split parameters (PROGRAM_SPEC.md Dataset splits).

Single source of truth for ``scripts/generate_dataset.py``. Difficulty for
train/val is ``DEFAULT_DIFFICULTY_V0`` from ``world.simulate`` — do not
restate its field literals here. OOD cells only override the single axis
locked in the PROGRAM_SPEC table.

``ood_noise`` / ``ood_confound`` / ``ood_regime_switch`` are inventoried as
deferred and are not generated until the corresponding axis is unlocked.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from world.schema import Difficulty
from world.simulate import DEFAULT_DIFFICULTY_V0

# Dry-run scale (prefix of production under the same base seeds).
N_EPISODES_DRY_RUN: Final[int] = 2000
N_STEPS_V0: Final[int] = 30

# Production v0 scale: train/val at §9 tens-of-thousands; OOD eval-only smaller.
N_EPISODES_TRAIN_V0: Final[int] = 50_000
N_EPISODES_VAL_V0: Final[int] = 50_000
N_EPISODES_OOD_PARTIAL_OBS_V0: Final[int] = 5_000

# Only OOD literal from PROGRAM_SPEC axis table (partial_obs_rate OOD cell).
_OOD_PARTIAL_OBS_RATE: Final[float] = 0.5

OOD_PARTIAL_OBS_DIFFICULTY: Final[Difficulty] = replace(
    DEFAULT_DIFFICULTY_V0,
    partial_obs_rate=_OOD_PARTIAL_OBS_RATE,
)

# Distinct base seeds per split. Derived episode seeds are
# ``base_seed + i`` for i in [0, n_episodes). Spacing of 1_000_000 leaves
# headroom well beyond dry-run and §9 "tens of thousands" scales so ranges
# never collide when n_episodes stays below that gap.
#
#   train:           [1_000_000, 1_000_000 + n)
#   val:             [2_000_000, 2_000_000 + n)
#   ood_partial_obs: [3_000_000, 3_000_000 + n)
#   ood_noise:       [4_000_000, ...)  reserved when unlocked
BASE_SEED_TRAIN: Final[int] = 1_000_000
BASE_SEED_VAL: Final[int] = 2_000_000
BASE_SEED_OOD_PARTIAL_OBS: Final[int] = 3_000_000
BASE_SEED_OOD_NOISE_RESERVED: Final[int] = 4_000_000


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """One dataset split ready for generation (active) or inventory-only."""

    name: str
    status: str  # "active" | "deferred"
    difficulty: Difficulty | None
    base_seed: int
    note: str = ""


# PROGRAM_SPEC inventory order. Only status=="active" rows are emitted.
SPLIT_SPECS: Final[tuple[SplitSpec, ...]] = (
    SplitSpec(
        name="train",
        status="active",
        difficulty=DEFAULT_DIFFICULTY_V0,
        base_seed=BASE_SEED_TRAIN,
    ),
    SplitSpec(
        name="val",
        status="active",
        difficulty=DEFAULT_DIFFICULTY_V0,
        base_seed=BASE_SEED_VAL,
        note="same Difficulty as train; disjoint base_seed",
    ),
    SplitSpec(
        name="ood_partial_obs",
        status="active",
        difficulty=OOD_PARTIAL_OBS_DIFFICULTY,
        base_seed=BASE_SEED_OOD_PARTIAL_OBS,
        note="single-axis OOD: partial_obs_rate=0.5; other axes = train",
    ),
    SplitSpec(
        name="ood_noise",
        status="deferred",
        difficulty=None,
        base_seed=BASE_SEED_OOD_NOISE_RESERVED,
        note="PROGRAM_SPEC: no locked OOD noise_scale yet - do not generate",
    ),
)


def active_split_specs() -> tuple[SplitSpec, ...]:
    return tuple(s for s in SPLIT_SPECS if s.status == "active")


def n_episodes_by_split_for_scale(scale: str) -> dict[str, int]:
    """Episode counts per active split.

    ``dry_run``: uniform ``N_EPISODES_DRY_RUN`` (strict prefix of ``v0`` under
    the same base seeds).
    ``v0``: train/val ``N_EPISODES_*_V0``; ``ood_partial_obs`` smaller (eval-only).
    """
    if scale == "dry_run":
        return {s.name: N_EPISODES_DRY_RUN for s in active_split_specs()}
    if scale == "v0":
        return {
            "train": N_EPISODES_TRAIN_V0,
            "val": N_EPISODES_VAL_V0,
            "ood_partial_obs": N_EPISODES_OOD_PARTIAL_OBS_V0,
        }
    raise ValueError(f"Unknown scale={scale!r}; expected 'dry_run' or 'v0'")


def assert_episode_seed_ranges_disjoint(
    specs: tuple[SplitSpec, ...],
    n_episodes: int | dict[str, int],
) -> None:
    """Fail if any two splits' ``base_seed + [0, n)`` ranges overlap.

    ``n_episodes`` may be a uniform int or a ``{split_name: n}`` mapping.
    """
    intervals: list[tuple[int, int, str]] = []
    for spec in specs:
        if isinstance(n_episodes, int):
            n = n_episodes
        else:
            if spec.name not in n_episodes:
                raise ValueError(f"Missing n_episodes for split {spec.name!r}")
            n = n_episodes[spec.name]
        if n < 0:
            raise ValueError(f"n_episodes must be >= 0, got {n} for {spec.name}")
        if n == 0:
            continue
        start = spec.base_seed
        end = spec.base_seed + n - 1
        for other_start, other_end, other_name in intervals:
            if not (end < other_start or start > other_end):
                raise ValueError(
                    f"Episode seed ranges overlap: {spec.name} "
                    f"[{start}, {end}] vs {other_name} "
                    f"[{other_start}, {other_end}]"
                )
        intervals.append((start, end, spec.name))
