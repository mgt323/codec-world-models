"""Training package: RunConfig, packer, stage-1 loop."""

from train.config import (
    BUDGET_PROFILES,
    BudgetProfile,
    RunConfig,
    budget_table,
    build_run_config,
    derive_budget,
    get_budget_profile,
    validate_run_config,
)
from train.loop import (
    PackingAccount,
    TrainResult,
    config_sha256,
    project_optimizer_steps,
    run_dir_for,
    run_training,
)
from train.data import file_sha256, pool_fingerprint
from train.packer import pack_episodes

__all__ = [
    "BUDGET_PROFILES",
    "BudgetProfile",
    "PackingAccount",
    "RunConfig",
    "TrainResult",
    "budget_table",
    "build_run_config",
    "config_sha256",
    "derive_budget",
    "file_sha256",
    "get_budget_profile",
    "pool_fingerprint",
    "pack_episodes",
    "project_optimizer_steps",
    "run_dir_for",
    "run_training",
    "validate_run_config",
]
