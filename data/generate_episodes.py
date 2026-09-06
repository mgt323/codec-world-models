"""Raw State episode generation for dataset splits (EXPERIMENT_PLAN.md §9).

This module produces ``list[State]`` episodes only. It must not import or call
``obs_codecs`` — codec strings are tokenized offline or on-the-fly as a later
step over this dataset.

Difficulty values for active splits are defined in PROGRAM_SPEC.md
(``DEFAULT_DIFFICULTY_V0`` + OOD table). This function accepts an explicit
``Difficulty``; mapping split names to those locked values is orchestration
(not implemented here).

Seed derivation (normative, for inspection / checksums)
------------------------------------------------------
For episode index ``i`` in ``0 .. n_episodes - 1``::

    episode_seed = base_seed + i

Each episode is ``simulate(episode_seed, n_steps, difficulty=difficulty,
interventions=...)``. Same
``(split_name, n_episodes, difficulty, base_seed, n_steps,
intervention_fraction)`` always yields the same sequence of
``(episode_seed, states)`` (byte-identical when pickled).

``split_name`` is validated against the PROGRAM_SPEC inventory but does not
enter the seed formula — disjoint train/val seeds come from distinct
``base_seed`` values chosen by the caller.

Separate intervention / counterfactual sets (EXPERIMENT_PLAN.md §9) use
``generate_intervention_episode`` and ``generate_counterfactual_pair`` —
not ``generate_split``.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Final

from world.schema import CausalRegime, Difficulty, Intervention, State
from world.simulate import simulate

# Active v0 splits (PROGRAM_SPEC Dataset splits).
ACTIVE_SPLIT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "train",
        "val",
        "ood_partial_obs",
    }
)

# Named in the API / PROGRAM_SPEC inventory but not unlocked for generation yet.
DEFERRED_SPLIT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "ood_noise",
        "ood_confound",
        "ood_regime_switch",
    }
)

KNOWN_SPLIT_NAMES: Final[frozenset[str]] = ACTIVE_SPLIT_NAMES | DEFERRED_SPLIT_NAMES

# Post-generation regime balance check (report only — never resample).
REGIME_BALANCE_MIN_EPISODES: Final[int] = 1000
REGIME_BALANCE_TOLERANCE: Final[float] = 0.10  # absolute fraction vs uniform 1/K


def episode_seed(base_seed: int, episode_index: int) -> int:
    """Map ``(base_seed, episode_index)`` → per-episode world seed.

    Formula: ``base_seed + episode_index`` (integer addition).
    """
    if episode_index < 0:
        raise ValueError(f"episode_index must be >= 0, got {episode_index}")
    return base_seed + episode_index


def generate_split(
    split_name: str,
    n_episodes: int,
    difficulty: Difficulty,
    base_seed: int,
    n_steps: int,
    intervention_fraction: float = 0.0,
) -> Iterator[tuple[int, list[State]]]:
    """Generate raw State episodes for one split.

    Yields ``(episode_seed, states)`` in episode-index order. See module
    docstring for the seed formula.

    Regime balance: when ``n_episodes >= REGIME_BALANCE_MIN_EPISODES``, after
    all episodes are simulated the realized per-regime share is compared to
    uniform ``1 / n_regimes``. If any class differs by more than
    ``REGIME_BALANCE_TOLERANCE`` (absolute), a ``UserWarning`` is emitted with
    the realized counts. Episodes are never discarded or resampled.
    """
    _validate_split_name(split_name)
    if n_episodes < 0:
        raise ValueError(f"n_episodes must be >= 0, got {n_episodes}")
    if not 0.0 <= intervention_fraction <= 1.0:
        raise ValueError(
            f"intervention_fraction must be in [0, 1], got {intervention_fraction}"
        )
    if intervention_fraction != 0.0:
        raise NotImplementedError(
            "intervention_fraction > 0 is reserved for Prompt B2 "
            f"(got {intervention_fraction})"
        )

    def _iter() -> Iterator[tuple[int, list[State]]]:
        # Stream one episode at a time; retain only regime labels for the
        # post-pass balance check (full State lists are not held).
        regime_by_episode: list[CausalRegime | None] = []
        for i in range(n_episodes):
            seed = episode_seed(base_seed, i)
            states = simulate(
                seed,
                n_steps,
                difficulty=difficulty,
                interventions=None,
            )
            regime_by_episode.append(
                states[0].latent.regime if states else None
            )
            yield seed, states
        _warn_if_regime_labels_imbalanced(regime_by_episode)

    return _iter()


def generate_intervention_episode(
    seed: int,
    n_steps: int,
    difficulty: Difficulty,
    intervention: Intervention | None,
) -> list[State]:
    """Simulate one episode with an optional scheduled intervention.

    Named entrypoint for the separate intervention episode set
    (EXPERIMENT_PLAN.md §9). Thin wrapper over ``simulate`` — does not
    participate in ``generate_split``.
    """
    interventions = (intervention,) if intervention is not None else None
    return simulate(
        seed,
        n_steps,
        difficulty=difficulty,
        interventions=interventions,
    )


def generate_counterfactual_pair(
    seed: int,
    n_steps: int,
    difficulty: Difficulty,
    intervention: Intervention,
) -> tuple[list[State], list[State]]:
    """Observational / intervened twins that share the same noise seed.

    Eval task #4 primitive (EXPERIMENT_PLAN.md §7 / §9): both episodes use
    ``seed``, so ``simulate``'s fixed per-step draw order yields identical
    noise realizations; only the ``do()`` schedule differs.

    Returns ``(observational_episode, intervened_episode)``.
    """
    observational = generate_intervention_episode(
        seed, n_steps, difficulty, intervention=None
    )
    intervened = generate_intervention_episode(
        seed, n_steps, difficulty, intervention=intervention
    )
    return observational, intervened


def _validate_split_name(split_name: str) -> None:
    if split_name in DEFERRED_SPLIT_NAMES:
        raise ValueError(
            f"split_name={split_name!r} is deferred in PROGRAM_SPEC.md v0 "
            "(axis not unlocked); do not generate this split yet"
        )
    if split_name not in ACTIVE_SPLIT_NAMES:
        known = ", ".join(sorted(KNOWN_SPLIT_NAMES))
        raise ValueError(
            f"Unknown split_name={split_name!r}; expected one of: {known}"
        )


def _warn_if_regime_imbalanced(
    episodes: Sequence[tuple[int, list[State]]],
) -> None:
    labels: list[CausalRegime | None] = [
        states[0].latent.regime if states else None for _, states in episodes
    ]
    _warn_if_regime_labels_imbalanced(labels)


def _warn_if_regime_labels_imbalanced(
    labels: Sequence[CausalRegime | None],
) -> None:
    n = len(labels)
    if n < REGIME_BALANCE_MIN_EPISODES:
        return
    if any(label is None for label in labels):
        raise ValueError(
            "Cannot check regime balance on an empty episode "
            "(n_steps must be > 0 when n_episodes >= "
            f"{REGIME_BALANCE_MIN_EPISODES})"
        )

    regimes = tuple(CausalRegime)
    expected = 1.0 / len(regimes)
    counts: Counter[CausalRegime] = Counter(
        label for label in labels if label is not None
    )

    offenders: list[str] = []
    for regime in regimes:
        share = counts[regime] / n
        if abs(share - expected) > REGIME_BALANCE_TOLERANCE:
            offenders.append(
                f"{regime.value}={counts[regime]}/{n} ({share:.3f})"
            )

    if offenders:
        realized = ", ".join(f"{r.value}={counts[r]}" for r in regimes)
        warnings.warn(
            "Regime distribution across "
            f"{n} episodes deviates from uniform "
            f"(expected ~{expected:.3f} per class, "
            f"tolerance ±{REGIME_BALANCE_TOLERANCE}): "
            f"{'; '.join(offenders)}. "
            f"Full counts: {realized}. "
            "Episodes were not resampled (determinism preserved).",
            UserWarning,
            stacklevel=2,
        )
