"""Training-matrix expansion, resume state, and manifest I/O.

Implements TRAINING_RUNNER_CONTRACT.md §4 (expansion, per-run outputs,
entrypoint responsibilities). Aggregation (§5) is deliberately not implemented:
no ``metrics_eval.json`` producer exists yet, so there is nothing to aggregate.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from train.config import (
    ALLOWED_CODECS,
    ALLOWED_REGIMES,
    BudgetProfile,
    RunConfig,
    budget_table,
    build_run_config,
    get_budget_profile,
    seeds_for_replication,
)
from train.loop import config_sha256, run_dir_for

# Contract §4.1 default codec list (E-variants included).
DEFAULT_CODECS: tuple[str, ...] = (
    "A",
    "B",
    "C",
    "D",
    "B_shuffle",
    "B_reverse",
    "B_bag",
)
DEFAULT_REGIMES: tuple[str, ...] = ("compute_matched", "info_matched")

MANIFEST_NAME = "manifest.json"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class MatrixCell:
    """One ``(codec, match_regime, replication_id)`` run."""

    config: RunConfig
    run_dir: Path

    @property
    def key(self) -> str:
        c = self.config
        return f"{c.match_regime}/{c.codec}/rep{c.replication_id}"


def expand_matrix(
    *,
    profile: BudgetProfile,
    codecs: Sequence[str],
    match_regimes: Sequence[str],
    n_replications: int,
    output_root: Path | str,
    runs_subdir: str | None = None,
) -> list[MatrixCell]:
    """Contract §4.2 expansion: regimes x codecs x replications."""
    if n_replications < 1:
        raise ValueError("n_replications must be >= 1")
    unknown_codecs = sorted(set(codecs) - ALLOWED_CODECS)
    if unknown_codecs:
        raise ValueError(f"Unknown codecs: {unknown_codecs}")
    unknown_regimes = sorted(set(match_regimes) - ALLOWED_REGIMES)
    if unknown_regimes:
        raise ValueError(f"Unknown match_regimes: {unknown_regimes}")

    cells: list[MatrixCell] = []
    for match_regime in match_regimes:
        for codec in codecs:
            for rep in range(n_replications):
                config = build_run_config(
                    codec=codec,
                    match_regime=match_regime,
                    replication_id=rep,
                    budget_profile=profile.name,
                )
                cells.append(
                    MatrixCell(
                        config=config,
                        run_dir=run_dir_for(
                            config, output_root, runs_subdir=runs_subdir
                        ),
                    )
                )
    return cells


def assert_seed_pairing(cells: Sequence[MatrixCell]) -> None:
    """Contract §4.4.2 / §3: shuffle seed shared per replication, seeds paired."""
    by_rep: dict[int, set[int]] = {}
    for cell in cells:
        c = cell.config
        expected_model, expected_shuffle = seeds_for_replication(c.replication_id)
        if c.model_init_seed != expected_model:
            raise ValueError(
                f"{cell.key}: model_init_seed {c.model_init_seed} != {expected_model}"
            )
        if c.data_shuffle_seed != expected_shuffle:
            raise ValueError(
                f"{cell.key}: data_shuffle_seed {c.data_shuffle_seed} != "
                f"{expected_shuffle}"
            )
        by_rep.setdefault(c.replication_id, set()).add(c.data_shuffle_seed)

    for rep, seeds in sorted(by_rep.items()):
        if len(seeds) != 1:
            raise ValueError(
                f"replication {rep}: data_shuffle_seed must be identical across "
                f"codecs, got {sorted(seeds)}"
            )
    all_shuffle = {next(iter(s)) for s in by_rep.values()}
    if len(all_shuffle) != len(by_rep):
        raise ValueError("data_shuffle_seed collides across replications")


def run_is_complete(cell: MatrixCell) -> tuple[bool, str]:
    """Return ``(complete, reason)`` for resume.

    Complete requires config.json + reproducibility.json + checkpoints/final.pt,
    parseable JSON, and an on-disk config hash equal to the expected config.
    """
    run_dir = cell.run_dir
    config_path = run_dir / "config.json"
    repro_path = run_dir / "reproducibility.json"
    final_ckpt = run_dir / "checkpoints" / "final.pt"

    for path in (config_path, repro_path, final_ckpt):
        if not path.is_file():
            return False, f"missing {path.name}"

    try:
        on_disk = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return False, f"unparseable config.json ({exc.__class__.__name__})"
    try:
        json.loads(repro_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return False, f"unparseable reproducibility.json ({exc.__class__.__name__})"

    try:
        on_disk_config = RunConfig.from_dict(on_disk)
    except (ValueError, TypeError) as exc:
        return False, f"config.json does not match schema ({exc})"

    expected_hash = config_sha256(cell.config)
    if config_sha256(on_disk_config) != expected_hash:
        return False, "config hash differs from expected"
    return True, "complete"


def write_manifest_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Single writer for manifest.json: tmp (same dir) -> fsync -> os.replace.

    Every caller — normal transitions, failures, and the KeyboardInterrupt
    flush — goes through this function, so the target file is never observed
    truncated. The temp file must share the directory (hence the volume) with
    the target, otherwise ``os.replace`` degrades to a non-atomic copy.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def build_manifest(
    *,
    profile: BudgetProfile,
    cells: Sequence[MatrixCell],
    codecs: Sequence[str],
    match_regimes: Sequence[str],
    n_replications: int,
    git_commit: str,
    smoke_max_steps: int | None,
) -> dict[str, Any]:
    """Fresh manifest with every cell pending."""
    return {
        "budget_profile": profile.name,
        "pool_episodes": profile.pool_episodes,
        "T_target": float(profile.t_target),
        "n_steps": cells[0].config.n_steps if cells else None,
        "codecs": list(codecs),
        "match_regimes": list(match_regimes),
        "n_replications": n_replications,
        "git_commit": git_commit,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "budget_table": budget_table(profile, list(codecs)),
        "eval_status": "pending_no_producer",
        "interpretation": (
            "scientific" if profile.scientific else "plumbing_only"
        ),
        "harness_step_cap": smoke_max_steps,
        "cells": [
            {
                "codec": cell.config.codec,
                "match_regime": cell.config.match_regime,
                "replication_id": cell.config.replication_id,
                "status": STATUS_PENDING,
                "run_dir": cell.run_dir.as_posix(),
                "config_sha256": config_sha256(cell.config),
                "n_episodes": cell.config.n_episodes,
                "T_eff": cell.config.T_eff,
                "optimizer_steps": None,
                "tokens_consumed": None,
                "wall_clock_seconds": None,
                "error": None,
            }
            for cell in cells
        ],
    }


def manifest_cell_index(manifest: dict[str, Any]) -> dict[str, int]:
    """Map ``regime/codec/repN`` to its index in ``manifest['cells']``."""
    index: dict[str, int] = {}
    for i, entry in enumerate(manifest.get("cells", [])):
        key = (
            f"{entry['match_regime']}/{entry['codec']}"
            f"/rep{entry['replication_id']}"
        )
        index[key] = i
    return index


def update_cell(
    manifest: dict[str, Any],
    index: dict[str, int],
    cell: MatrixCell,
    **fields: Any,
) -> None:
    """Mutate one cell entry in-place (caller persists via write_manifest_atomic)."""
    entry = manifest["cells"][index[cell.key]]
    entry.update(fields)
    manifest["updated_utc"] = _utc_now()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_pool_requirements(cells: Sequence[MatrixCell]) -> Iterator[str]:
    """Distinct codecs appearing in the matrix, in stable order."""
    seen: set[str] = set()
    for cell in cells:
        codec = cell.config.codec
        if codec not in seen:
            seen.add(codec)
            yield codec
