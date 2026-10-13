"""RunConfig schema matching TRAINING_RUNNER_CONTRACT.md §2.

Budgets are **derived**, never hardcoded per tier: one formula from
``COMPUTE_MATCH_BUDGETS.md`` §0 instantiated at a profile's pool size.
At ``pool_episodes = 50_000`` it reproduces the locked v0 table exactly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any, Literal, Mapping

from model.architecture import ARCHITECTURE_ID, CONTEXT_LENGTH_L
from obs_codecs.diagnostics import TOKENIZATION_SCHEME_VERSION

MatchRegime = Literal["compute_matched", "info_matched"]
CodecName = Literal["A", "B", "C", "D", "B_shuffle", "B_reverse", "B_bag"]

OPTIMIZER_ID = "adamw_v0_lr3e4_wd0.1_bs8"
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)

# Episode length (configs.dataset_v0.N_STEPS_V0); not a scaling axis.
N_STEPS_V0 = 30

# COMPUTE_MATCH_BUDGETS.md §0: treat the three-decimal rates as exact decimals
# for budget arithmetic (not IEEE float intermediates).
AVG_TOKENS_PER_OBSERVATION: dict[str, Decimal] = {
    "A": Decimal("72.292"),
    "B": Decimal("45.888"),
    "C": Decimal("49.292"),
    "D": Decimal("55.292"),
}

# E-variants inherit B's rate (COMPUTE_MATCH_BUDGETS.md §4.2).
RATE_GROUP: dict[str, str] = {
    "A": "A",
    "B": "B",
    "B_shuffle": "B",
    "B_reverse": "B",
    "B_bag": "B",
    "C": "C",
    "D": "D",
}

# Codec anchoring T_target: one full pass of A over the pool.
ANCHOR_CODEC = "A"

MODEL_SEED_BASE = 1_000_000
SHUFFLE_SEED_BASE = 2_000_000

ALLOWED_CODECS = frozenset(RATE_GROUP)
ALLOWED_REGIMES = frozenset({"compute_matched", "info_matched"})


@dataclass(frozen=True, slots=True)
class BudgetProfile:
    """One matched-budget scale.

    ``pool_episodes`` is the unique train-episode pool at this scale; it is
    simultaneously the ``info_matched`` episode count and the anchor for
    ``T_target``. Scaling is a single constant shared by every codec and
    replication, with floor re-applied per codec (COMPUTE_MATCH_BUDGETS.md §4.7).
    """

    name: str
    pool_episodes: int
    scientific: bool

    @property
    def t_target(self) -> Decimal:
        """``T_target = pool_episodes x n_steps x avg_A`` (exact decimal)."""
        return (
            Decimal(self.pool_episodes)
            * Decimal(N_STEPS_V0)
            * AVG_TOKENS_PER_OBSERVATION[ANCHOR_CODEC]
        )


BUDGET_PROFILES: dict[str, BudgetProfile] = {
    "tier0": BudgetProfile("tier0", 200, scientific=False),
    "tier1": BudgetProfile("tier1", 2_000, scientific=False),
    "full_v0": BudgetProfile("full_v0", 50_000, scientific=True),
}

DEFAULT_BUDGET_PROFILE = "full_v0"


def get_budget_profile(name: str) -> BudgetProfile:
    try:
        return BUDGET_PROFILES[name]
    except KeyError:
        raise ValueError(
            f"Unknown budget_profile {name!r}; "
            f"expected one of {sorted(BUDGET_PROFILES)}"
        ) from None


def avg_tokens_for_codec(codec: str) -> Decimal:
    if codec not in RATE_GROUP:
        raise ValueError(f"Unknown codec {codec!r}")
    return AVG_TOKENS_PER_OBSERVATION[RATE_GROUP[codec]]


def derive_budget(
    codec: str,
    match_regime: str,
    profile: BudgetProfile,
) -> tuple[int, Decimal, Decimal | None]:
    """Return ``(n_episodes, T_eff, T_target)`` for one cell.

    compute_matched: ``n_episodes = floor(T_target / (avg x n_steps))``, so no
    codec exceeds ``T_target`` (locked floor policy, never round-nearest).
    info_matched: every codec sees the same ``pool_episodes`` unique episodes.
    """
    if match_regime not in ALLOWED_REGIMES:
        raise ValueError(f"Unknown match_regime {match_regime!r}")
    avg = avg_tokens_for_codec(codec)
    tokens_per_episode = avg * Decimal(N_STEPS_V0)

    if match_regime == "compute_matched":
        t_target = profile.t_target
        n_episodes = int(
            (t_target / tokens_per_episode).to_integral_value(rounding=ROUND_FLOOR)
        )
        return n_episodes, Decimal(n_episodes) * tokens_per_episode, t_target

    n_episodes = profile.pool_episodes
    return n_episodes, Decimal(n_episodes) * tokens_per_episode, None


def budget_table(
    profile: BudgetProfile,
    codecs: tuple[str, ...] | list[str] | None = None,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Derived budgets keyed ``[codec][match_regime]`` (JSON-ready floats)."""
    names = tuple(codecs) if codecs is not None else tuple(sorted(RATE_GROUP))
    table: dict[str, dict[str, dict[str, float | int]]] = {}
    for codec in names:
        table[codec] = {}
        for regime in ("compute_matched", "info_matched"):
            n_episodes, t_eff, t_target = derive_budget(codec, regime, profile)
            table[codec][regime] = {
                "n_episodes": n_episodes,
                "T_eff": float(t_eff),
                "T_target": float(t_target) if t_target is not None else None,
            }
    return table

# Schema field names from runner contract §2 (identity + budget + locked knobs
# + checkpoint policy recommendations). Training code must not read others.
_SCHEMA_FIELDS = frozenset(
    {
        "codec",
        "match_regime",
        "replication_id",
        "model_init_seed",
        "data_shuffle_seed",
        "n_episodes",
        "T_eff",
        "T_target",
        "budget_profile",
        "avg_tokens_per_observation",
        "n_steps",
        "token_rate_artifact",
        "tokenization_scheme_version",
        "vocab_policy",
        "context_length_L",
        "document_masking",
        "boundary_markers",
        "mask_source",
        "reshuffle_policy",
        "difficulty",
        "dataset_split_train",
        "architecture_id",
        "optimizer_id",
        "scoring_checkpoint",
        "early_stopping",
        "best_checkpoint_selection",
    }
)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Single-run config; fields mirror TRAINING_RUNNER_CONTRACT.md §2."""

    codec: str
    match_regime: str
    replication_id: int
    model_init_seed: int
    data_shuffle_seed: int
    n_episodes: int
    T_eff: float
    avg_tokens_per_observation: float
    budget_profile: str = DEFAULT_BUDGET_PROFILE
    n_steps: int = N_STEPS_V0
    token_rate_artifact: str = "obs_codecs/token_rates_train_v4.json"
    tokenization_scheme_version: str = TOKENIZATION_SCHEME_VERSION
    vocab_policy: str = "train_only_b_family_shared"
    context_length_L: int = CONTEXT_LENGTH_L
    document_masking: bool = True
    boundary_markers: bool = True
    mask_source: str = "segment_ids"
    reshuffle_policy: str = "reshuffle_per_epoch"
    difficulty: str = "DEFAULT_DIFFICULTY_V0"
    dataset_split_train: str = "artifacts/encoded/train_{codec}.parquet"
    architecture_id: str = ARCHITECTURE_ID
    optimizer_id: str = OPTIMIZER_ID
    scoring_checkpoint: str = "final"
    early_stopping: bool = False
    best_checkpoint_selection: bool = False
    T_target: float | None = 108_438_000.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunConfig:
        unknown = set(payload) - _SCHEMA_FIELDS
        if unknown:
            raise ValueError(f"Unknown RunConfig fields: {sorted(unknown)}")
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in payload.items() if k in allowed}
        return cls(**kwargs)


def seeds_for_replication(replication_id: int) -> tuple[int, int]:
    return (
        MODEL_SEED_BASE + replication_id,
        SHUFFLE_SEED_BASE + replication_id,
    )


def build_run_config(
    *,
    codec: str,
    match_regime: str,
    replication_id: int,
    budget_profile: str = DEFAULT_BUDGET_PROFILE,
) -> RunConfig:
    """Construct a derived-budget config for one matrix cell × replication."""
    if codec not in ALLOWED_CODECS:
        raise ValueError(f"Unknown codec {codec!r}")
    if match_regime not in ALLOWED_REGIMES:
        raise ValueError(f"Unknown match_regime {match_regime!r}")
    if replication_id < 0:
        raise ValueError("replication_id must be >= 0")

    profile = get_budget_profile(budget_profile)
    model_init_seed, data_shuffle_seed = seeds_for_replication(replication_id)
    n_episodes, t_eff, t_target = derive_budget(codec, match_regime, profile)

    return RunConfig(
        codec=codec,
        match_regime=match_regime,
        replication_id=replication_id,
        model_init_seed=model_init_seed,
        data_shuffle_seed=data_shuffle_seed,
        n_episodes=n_episodes,
        T_eff=float(t_eff),
        T_target=float(t_target) if t_target is not None else None,
        budget_profile=profile.name,
        avg_tokens_per_observation=float(avg_tokens_for_codec(codec)),
        dataset_split_train=f"artifacts/encoded/train_{codec}.parquet",
    )


def validate_run_config(config: RunConfig) -> None:
    """Reject configs that violate locked knobs / the profile's budget table."""
    expected = build_run_config(
        codec=config.codec,
        match_regime=config.match_regime,
        replication_id=config.replication_id,
        budget_profile=config.budget_profile,
    )
    if config.n_episodes != expected.n_episodes:
        raise ValueError(
            f"n_episodes {config.n_episodes} != derived {expected.n_episodes} "
            f"for profile {config.budget_profile!r}"
        )
    if abs(config.T_eff - expected.T_eff) > 1e-6:
        raise ValueError(
            f"T_eff {config.T_eff} != derived {expected.T_eff} "
            f"for profile {config.budget_profile!r}"
        )
    if config.T_target != expected.T_target:
        raise ValueError(
            f"T_target {config.T_target} != derived {expected.T_target}"
        )
    if config.avg_tokens_per_observation != expected.avg_tokens_per_observation:
        raise ValueError(
            f"avg_tokens_per_observation {config.avg_tokens_per_observation} != "
            f"locked rate {expected.avg_tokens_per_observation}"
        )
    if config.model_init_seed != expected.model_init_seed:
        raise ValueError(
            f"model_init_seed {config.model_init_seed} != "
            f"{expected.model_init_seed} for replication {config.replication_id}"
        )
    if config.data_shuffle_seed != expected.data_shuffle_seed:
        raise ValueError(
            f"data_shuffle_seed {config.data_shuffle_seed} != "
            f"{expected.data_shuffle_seed} for replication {config.replication_id}"
        )
    if config.n_steps != N_STEPS_V0:
        raise ValueError(f"n_steps must be {N_STEPS_V0}, got {config.n_steps}")
    if config.context_length_L != CONTEXT_LENGTH_L:
        raise ValueError(
            f"context_length_L must be {CONTEXT_LENGTH_L}, got {config.context_length_L}"
        )
    if not config.document_masking:
        raise ValueError("document_masking must be true")
    if config.mask_source != "segment_ids":
        raise ValueError("mask_source must be 'segment_ids'")
    if config.reshuffle_policy != "reshuffle_per_epoch":
        raise ValueError("reshuffle_policy must be reshuffle_per_epoch")
    if config.scoring_checkpoint != "final":
        raise ValueError("scoring_checkpoint must be 'final'")
    if config.early_stopping or config.best_checkpoint_selection:
        raise ValueError("early stopping / best-checkpoint selection forbidden")
    if config.architecture_id != ARCHITECTURE_ID:
        raise ValueError(f"architecture_id must be {ARCHITECTURE_ID!r}")
    if config.optimizer_id != OPTIMIZER_ID:
        raise ValueError(f"optimizer_id must be {OPTIMIZER_ID!r}")
    if not config.boundary_markers:
        raise ValueError("boundary_markers must be true")
