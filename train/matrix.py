"""Training-matrix expansion, resume state, manifest I/O, and §5 aggregation.

Implements docs/TRAINING_RUNNER_CONTRACT.md §4 (expansion, per-run outputs,
entrypoint responsibilities) and §5 (cross-replication median/IQR summaries).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
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

# Invariant I14: scientific aggregation needs at least this many replications.
MIN_SCIENTIFIC_REPLICATIONS = 3

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
    min_replications: int | None = None,
) -> list[MatrixCell]:
    """Contract §4.2 expansion: regimes x codecs x replications.

    ``min_replications`` enforces invariant I14 for scientific profiles. The
    matrix entrypoint sets it to 3 for ``full_v0`` unless a smoke step cap is
    in effect. Unit tests that expand a single cell leave it unset.
    """
    if n_replications < 1:
        raise ValueError("n_replications must be >= 1")
    if min_replications is not None and n_replications < min_replications:
        raise ValueError(
            f"n_replications must be >= {min_replications} for profile "
            f"{profile.name} (got {n_replications})"
        )
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
    git_dirty: bool = False,
    hardware: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fresh manifest with every cell pending."""
    payload: dict[str, Any] = {
        "budget_profile": profile.name,
        "pool_episodes": profile.pool_episodes,
        "T_target": float(profile.t_target),
        "n_steps": cells[0].config.n_steps if cells else None,
        "codecs": list(codecs),
        "match_regimes": list(match_regimes),
        "n_replications": n_replications,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "budget_table": budget_table(profile, list(codecs)),
        "eval_status": "pending",
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
    if hardware is not None:
        payload.update(dict(hardware))
    return payload


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


def aggregate_scientific_metrics(
    runs_root: Path,
    *,
    profile_name: str,
    output_root: Path | str,
    git_commit: str,
    git_dirty: bool,
    allow_pilot: bool = False,
    aggregates_dir: Path | str | None = None,
    eval_schema_version: str | None = None,
    decoder_mode: str | None = None,
    eval_git_commit: str | None = None,
    eval_git_dirty: bool | None = None,
) -> list[Path]:
    """Write contract §5 summaries for a scientific profile.

    ``tier0`` and ``tier1`` are refused unless ``allow_pilot=True``. Quartiles
    use ``numpy.percentile(..., method="linear")``. A metric that is null for
    every replication is omitted rather than aggregated.

    Contract §2.6: any ``metrics_eval.json`` whose ``scoring_checkpoint`` is not
    ``"final"`` aborts aggregation. Invariant I14: unless ``allow_pilot`` is set,
    every aggregated (regime, split, metric, codec) cell needs at least 3
    replication values.
    """
    import numpy as np

    profile = get_budget_profile(profile_name)
    if not profile.scientific and not allow_pilot:
        raise ValueError(
            f"aggregation is refused for plumbing profile {profile.name}"
        )
    written: list[Path] = []
    root = Path(runs_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    # runs_root is artifacts/runs/full_v0: match_regime / codec / repN
    by_cell: dict[tuple[str, str, str, str], list[float]] = {}
    for metrics_path in sorted(root.glob("*/*/*/metrics_eval.json")):
        rep_dir = metrics_path.parent
        codec = rep_dir.parent.name
        match_regime = rep_dir.parent.parent.name
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        scoring_checkpoint = payload.get("scoring_checkpoint")
        if scoring_checkpoint != "final":
            raise ValueError(
                f"{metrics_path}: scoring_checkpoint must be 'final' to aggregate "
                f"(contract §2.6), got {scoring_checkpoint!r}"
            )
        splits = payload.get("splits", payload)
        for split_name, metrics in splits.items():
            if not isinstance(metrics, dict):
                continue
            for metric_name, value in metrics.items():
                if value is None:
                    continue
                by_cell.setdefault(
                    (match_regime, str(split_name), str(metric_name), codec),
                    [],
                ).append(float(value))
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for (match_regime, split_name, metric_name, codec), values in sorted(by_cell.items()):
        if not allow_pilot and len(values) < MIN_SCIENTIFIC_REPLICATIONS:
            raise ValueError(
                f"{match_regime}/{split_name}/{metric_name}/{codec}: need >= "
                f"{MIN_SCIENTIFIC_REPLICATIONS} replications to aggregate, got "
                f"{len(values)} (Invariant I14)"
            )
        array = np.asarray(values, dtype=np.float64)
        q1, median, q3 = np.percentile(array, [25.0, 50.0, 75.0], method="linear")
        bucket = summaries.setdefault(
            (match_regime, split_name),
            {
                "match_regime": match_regime,
                "eval_split": split_name,
                "budget_profile": profile.name,
                "n_replications": len(values),
                "aggregation": "median_iqr",
                "percentile_method": "linear",
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                **(
                    {"eval_schema_version": eval_schema_version}
                    if eval_schema_version is not None
                    else {}
                ),
                **({"decoder_mode": decoder_mode} if decoder_mode is not None else {}),
                **(
                    {"eval_git_commit": eval_git_commit}
                    if eval_git_commit is not None
                    else {}
                ),
                **(
                    {"eval_git_dirty": eval_git_dirty}
                    if eval_git_dirty is not None
                    else {}
                ),
                "metrics": {},
            },
        )
        bucket["metrics"].setdefault(metric_name, {})[codec] = {
            "median": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "values": [float(value) for value in values],
        }
        bucket["n_replications"] = max(bucket["n_replications"], len(values))
    out_root = Path(output_root)
    agg_base = (
        Path(aggregates_dir)
        if aggregates_dir is not None
        else out_root / "runs" / "aggregates"
    )
    for (match_regime, split_name), payload in sorted(summaries.items()):
        path = agg_base / match_regime / split_name / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def iter_pool_requirements(cells: Sequence[MatrixCell]) -> Iterator[str]:
    """Distinct codecs appearing in the matrix, in stable order."""
    seen: set[str] = set()
    for cell in cells:
        codec = cell.config.codec
        if codec not in seen:
            seen.add(codec)
            yield codec
