"""Training-matrix expansion, budget derivation, resume, and manifest atomicity.

Covers TRAINING_RUNNER_CONTRACT.md §4 for the pilot tiers plus the drift guard
that the derived ``full_v0`` table still equals COMPUTE_MATCH_BUDGETS.md.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest

from train.config import (
    ALLOWED_CODECS,
    RunConfig,
    budget_table,
    build_run_config,
    derive_budget,
    get_budget_profile,
    validate_run_config,
)
from train.loop import config_sha256, run_dir_for
from train.matrix import (
    DEFAULT_CODECS,
    DEFAULT_REGIMES,
    MANIFEST_NAME,
    STATUS_COMPLETED,
    MatrixCell,
    assert_seed_pairing,
    build_manifest,
    expand_matrix,
    load_manifest,
    manifest_cell_index,
    run_is_complete,
    update_cell,
    write_manifest_atomic,
)

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "run_training_matrix.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_training_matrix", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# COMPUTE_MATCH_BUDGETS.md §1 / §2 tables, transcribed as the drift reference.
_LOCKED_FULL_V0 = {
    "A": {"cm_n": 50_000, "cm_t": 108_438_000.0, "im_t": 108_438_000.0},
    "B": {"cm_n": 78_770, "cm_t": 108_437_932.8, "im_t": 68_832_000.0},
    "B_shuffle": {"cm_n": 78_770, "cm_t": 108_437_932.8, "im_t": 68_832_000.0},
    "B_reverse": {"cm_n": 78_770, "cm_t": 108_437_932.8, "im_t": 68_832_000.0},
    "B_bag": {"cm_n": 78_770, "cm_t": 108_437_932.8, "im_t": 68_832_000.0},
    "C": {"cm_n": 73_330, "cm_t": 108_437_470.8, "im_t": 73_938_000.0},
    "D": {"cm_n": 65_372, "cm_t": 108_436_458.72, "im_t": 82_938_000.0},
}

_TIER0_CM = {"A": (200, 433_752.00), "B": (315, 433_641.60),
             "C": (293, 433_276.68), "D": (261, 432_936.36)}
_TIER1_CM = {"A": (2_000, 4_337_520.00), "B": (3_150, 4_336_416.00),
             "C": (2_933, 4_337_203.08), "D": (2_614, 4_335_998.64)}


def test_full_v0_derivation_matches_locked_doc_table() -> None:
    """Formula must reproduce the locked table exactly (no silent drift)."""
    assert set(_LOCKED_FULL_V0) == set(ALLOWED_CODECS)
    profile = get_budget_profile("full_v0")
    for codec, expected in _LOCKED_FULL_V0.items():
        cm_n, cm_t, cm_target = derive_budget(codec, "compute_matched", profile)
        im_n, im_t, im_target = derive_budget(codec, "info_matched", profile)
        assert cm_n == expected["cm_n"], codec
        assert float(cm_t) == expected["cm_t"], codec
        assert float(cm_target) == 108_438_000.0
        assert im_n == 50_000, codec
        assert float(im_t) == expected["im_t"], codec
        assert im_target is None


@pytest.mark.parametrize(
    ("profile_name", "pool", "t_target", "table"),
    [
        ("tier0", 200, 433_752.0, _TIER0_CM),
        ("tier1", 2_000, 4_337_520.0, _TIER1_CM),
    ],
)
def test_tier_tables_exact(
    profile_name: str, pool: int, t_target: float, table: dict[str, tuple[int, float]]
) -> None:
    profile = get_budget_profile(profile_name)
    assert profile.pool_episodes == pool
    assert float(profile.t_target) == t_target
    assert profile.scientific is False
    for codec, (n_episodes, t_eff) in table.items():
        cm_n, cm_t, cm_target = derive_budget(codec, "compute_matched", profile)
        assert (cm_n, float(cm_t)) == (n_episodes, t_eff), codec
        # Floor policy: never exceed the tier's own T_target.
        assert float(cm_t) <= float(cm_target)
        im_n, _im_t, _ = derive_budget(codec, "info_matched", profile)
        assert im_n == pool


def test_e_variants_inherit_b_budget() -> None:
    profile = get_budget_profile("tier0")
    b = derive_budget("B", "compute_matched", profile)
    for variant in ("B_shuffle", "B_reverse", "B_bag"):
        assert derive_budget(variant, "compute_matched", profile) == b


def test_expansion_shape_and_seed_pairing(tmp_path: Path) -> None:
    profile = get_budget_profile("tier0")
    cells = expand_matrix(
        profile=profile,
        codecs=DEFAULT_CODECS,
        match_regimes=DEFAULT_REGIMES,
        n_replications=3,
        output_root=tmp_path,
        runs_subdir="tier0",
    )
    assert len(cells) == len(DEFAULT_CODECS) * len(DEFAULT_REGIMES) * 3
    assert len({c.key for c in cells}) == len(cells)
    assert_seed_pairing(cells)

    for rep in range(3):
        rep_cells = [c for c in cells if c.config.replication_id == rep]
        shuffle_seeds = {c.config.data_shuffle_seed for c in rep_cells}
        assert shuffle_seeds == {2_000_000 + rep}
        assert {c.config.model_init_seed for c in rep_cells} == {1_000_000 + rep}

    for cell in cells:
        validate_run_config(cell.config)


def test_seed_pairing_rejects_desynced_shuffle_seed(tmp_path: Path) -> None:
    profile = get_budget_profile("tier0")
    cells = expand_matrix(
        profile=profile,
        codecs=["A", "B"],
        match_regimes=["compute_matched"],
        n_replications=1,
        output_root=tmp_path,
        runs_subdir="tier0",
    )
    from dataclasses import replace

    bad = MatrixCell(
        config=replace(cells[1].config, data_shuffle_seed=999),
        run_dir=cells[1].run_dir,
    )
    with pytest.raises(ValueError, match="data_shuffle_seed"):
        assert_seed_pairing([cells[0], bad])


def test_run_dir_layout_includes_tier_segment(tmp_path: Path) -> None:
    config = build_run_config(
        codec="A",
        match_regime="compute_matched",
        replication_id=0,
        budget_profile="tier0",
    )
    path = run_dir_for(config, tmp_path, runs_subdir="tier0")
    assert path == tmp_path / "runs" / "tier0" / "compute_matched" / "A" / "rep0"
    # Default (no segment) preserves the pre-existing single-run layout.
    assert run_dir_for(config, tmp_path) == (
        tmp_path / "runs" / "compute_matched" / "A" / "rep0"
    )


def _fake_completed_run(cell: MatrixCell, *, config: RunConfig | None = None) -> None:
    """Write a run directory that satisfies the completeness check."""
    written = config if config is not None else cell.config
    (cell.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    written.to_json(cell.run_dir / "config.json")
    (cell.run_dir / "reproducibility.json").write_text(
        json.dumps({"config_sha256": config_sha256(written)}), encoding="utf-8"
    )
    (cell.run_dir / "checkpoints" / "final.pt").write_bytes(b"stub")


@pytest.fixture()
def one_cell(tmp_path: Path) -> MatrixCell:
    return expand_matrix(
        profile=get_budget_profile("tier0"),
        codecs=["A"],
        match_regimes=["compute_matched"],
        n_replications=1,
        output_root=tmp_path,
        runs_subdir="tier0",
    )[0]


def test_resume_detects_complete_run(one_cell: MatrixCell) -> None:
    assert run_is_complete(one_cell)[0] is False
    _fake_completed_run(one_cell)
    complete, reason = run_is_complete(one_cell)
    assert complete is True, reason


def test_resume_reruns_on_missing_or_corrupt_files(one_cell: MatrixCell) -> None:
    _fake_completed_run(one_cell)
    (one_cell.run_dir / "checkpoints" / "final.pt").unlink()
    assert run_is_complete(one_cell)[0] is False

    _fake_completed_run(one_cell)
    (one_cell.run_dir / "reproducibility.json").write_text(
        '{"truncated": ', encoding="utf-8"
    )
    complete, reason = run_is_complete(one_cell)
    assert complete is False
    assert "unparseable reproducibility.json" in reason


def test_resume_reruns_when_budget_profile_changed(one_cell: MatrixCell) -> None:
    other = build_run_config(
        codec="A",
        match_regime="compute_matched",
        replication_id=0,
        budget_profile="tier1",
    )
    _fake_completed_run(one_cell, config=other)
    complete, reason = run_is_complete(one_cell)
    assert complete is False
    assert "config hash differs" in reason


# --- manifest atomicity -----------------------------------------------------


def _manifest_payload(tag: str) -> dict[str, Any]:
    return {"marker": tag, "cells": []}


def test_manifest_tmp_shares_directory_with_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """os.replace is only atomic within one volume; tmp must be a sibling."""
    target = tmp_path / "runs" / "tier0" / MANIFEST_NAME
    seen: dict[str, Path] = {}
    real_replace = os.replace

    def spy(src, dst):  # type: ignore[no-untyped-def]
        seen["src"] = Path(src)
        seen["dst"] = Path(dst)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    write_manifest_atomic(target, _manifest_payload("v1"))
    assert seen["src"].parent == seen["dst"].parent == target.parent


def test_manifest_survives_interrupt_mid_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial bytes land in the tmp file; the target keeps the prior payload."""
    target = tmp_path / MANIFEST_NAME
    write_manifest_atomic(target, _manifest_payload("v1"))

    import train.matrix as matrix_mod

    def dump_then_interrupt(payload, fh, **kwargs):  # type: ignore[no-untyped-def]
        fh.write('{"marker": "partial"')  # truncated on purpose
        raise KeyboardInterrupt

    monkeypatch.setattr(matrix_mod.json, "dump", dump_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        write_manifest_atomic(target, _manifest_payload("v2"))

    monkeypatch.undo()
    assert json.loads(target.read_text(encoding="utf-8"))["marker"] == "v1"


def test_manifest_survives_failed_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / MANIFEST_NAME
    write_manifest_atomic(target, _manifest_payload("v1"))

    def boom(src, dst):  # type: ignore[no-untyped-def]
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_manifest_atomic(target, _manifest_payload("v2"))

    monkeypatch.undo()
    assert json.loads(target.read_text(encoding="utf-8"))["marker"] == "v1"


def test_load_manifest_returns_none_for_corrupt_file(tmp_path: Path) -> None:
    target = tmp_path / MANIFEST_NAME
    target.write_text("{not json", encoding="utf-8")
    assert load_manifest(target) is None


def test_manifest_marks_pilot_as_plumbing_only(tmp_path: Path) -> None:
    profile = get_budget_profile("tier0")
    cells = expand_matrix(
        profile=profile,
        codecs=["A"],
        match_regimes=["compute_matched"],
        n_replications=1,
        output_root=tmp_path,
        runs_subdir="tier0",
    )
    manifest = build_manifest(
        profile=profile,
        cells=cells,
        codecs=["A"],
        match_regimes=["compute_matched"],
        n_replications=1,
        git_commit="deadbeef",
        smoke_max_steps=None,
    )
    assert manifest["interpretation"] == "plumbing_only"
    assert manifest["eval_status"] == "pending_no_producer"
    assert manifest["budget_table"]["A"]["compute_matched"]["n_episodes"] == 200

    full_cells = expand_matrix(
        profile=get_budget_profile("full_v0"),
        codecs=["A"],
        match_regimes=["compute_matched"],
        n_replications=1,
        output_root=tmp_path,
        runs_subdir="full_v0",
    )
    full_manifest = build_manifest(
        profile=get_budget_profile("full_v0"),
        cells=full_cells,
        codecs=["A"],
        match_regimes=["compute_matched"],
        n_replications=1,
        git_commit="deadbeef",
        smoke_max_steps=None,
    )
    assert full_manifest["interpretation"] == "scientific"


def test_update_cell_roundtrips_through_manifest(tmp_path: Path) -> None:
    profile = get_budget_profile("tier0")
    cells = expand_matrix(
        profile=profile,
        codecs=["A", "B"],
        match_regimes=["compute_matched"],
        n_replications=1,
        output_root=tmp_path,
        runs_subdir="tier0",
    )
    manifest = build_manifest(
        profile=profile,
        cells=cells,
        codecs=["A", "B"],
        match_regimes=["compute_matched"],
        n_replications=1,
        git_commit="x",
        smoke_max_steps=None,
    )
    index = manifest_cell_index(manifest)
    update_cell(manifest, index, cells[0], status=STATUS_COMPLETED, optimizer_steps=7)
    path = tmp_path / MANIFEST_NAME
    write_manifest_atomic(path, manifest)

    reloaded = load_manifest(path)
    assert reloaded is not None
    entry = reloaded["cells"][manifest_cell_index(reloaded)[cells[0].key]]
    assert entry["status"] == STATUS_COMPLETED
    assert entry["optimizer_steps"] == 7


# --- dataset provenance -----------------------------------------------------


def test_pool_fingerprint_is_deterministic_and_order_sensitive() -> None:
    from train.data import pool_fingerprint

    pool = [[4, 5, 6], [7, 8]]
    assert pool_fingerprint(pool) == pool_fingerprint([[4, 5, 6], [7, 8]])
    assert pool_fingerprint(pool) != pool_fingerprint([[7, 8], [4, 5, 6]])
    # Truncation must be visible, so a short corpus cannot masquerade as full.
    assert pool_fingerprint(pool) != pool_fingerprint(pool[:1])
    # Episode boundaries must not be collapsible into one flat stream.
    assert pool_fingerprint([[4, 5], [6]]) != pool_fingerprint([[4], [5, 6]])
    assert pool_fingerprint([[], [4]]) != pool_fingerprint([[4], []])
    # Pinned digest: encoding is fixed little-endian int32, so the value must
    # not drift with platform word size, endianness, or a refactor.
    assert pool_fingerprint([[1], [2, 3]]) == "4b04bc60ea2135acc9cd32a87cdd6571"


def test_file_sha256_missing_returns_none(tmp_path: Path) -> None:
    from train.data import file_sha256

    assert file_sha256(tmp_path / "nope.parquet") is None
    target = tmp_path / "x.bin"
    target.write_bytes(b"abc")
    assert file_sha256(target) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


# --- CLI integration --------------------------------------------------------


@pytest.mark.integration
def test_pool_guard_aborts_when_corpus_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pool shorter than profile.pool_episodes must abort during preflight."""
    script = _load_script()
    profile = get_budget_profile("tier1")
    cells = expand_matrix(
        profile=profile,
        codecs=["A"],
        match_regimes=["compute_matched"],
        n_replications=1,
        output_root=tmp_path,
        runs_subdir="tier1",
    )
    monkeypatch.setattr(
        script, "episodes_from_parquet", lambda *a, **k: [[4, 5]] * 10
    )
    with pytest.raises(SystemExit, match="requires exactly 2000"):
        script._preflight(
            cells,
            profile,
            vocab_dir=_ROOT / "artifacts" / "vocabs",
            batch_size=8,
        )
    assert not (tmp_path / "runs").exists()


@pytest.mark.integration
def test_micro_matrix_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """2 codecs x 2 regimes x 1 rep, step-capped, writes tree + manifest."""
    script = _load_script()
    profile = get_budget_profile("tier0")

    tiny_pool = [[4 + (i % 20)] * 60 for i in range(profile.pool_episodes)]
    monkeypatch.setattr(script, "episodes_from_parquet", lambda *a, **k: tiny_pool)

    rc = script.main(
        [
            "--tier", "0",
            "--codecs", "A", "B",
            "--replications", "1",
            "--output-root", str(tmp_path),
            "--smoke-max-steps", "1",
        ]
    )
    assert rc == 0

    for regime in ("compute_matched", "info_matched"):
        for codec in ("A", "B"):
            run_dir = tmp_path / "runs" / "tier0" / regime / codec / "rep0"
            assert (run_dir / "config.json").is_file()
            assert (run_dir / "metrics_train.jsonl").is_file()
            assert (run_dir / "reproducibility.json").is_file()
            assert (run_dir / "checkpoints" / "final.pt").is_file()
            # No eval producer exists yet; must not be stubbed.
            assert not (run_dir / "metrics_eval.json").exists()
            repro = json.loads((run_dir / "reproducibility.json").read_text("utf-8"))
            assert repro["budget_profile"] == "tier0"
            assert repro["harness_step_cap"] == 1
            assert repro["config_sha256"]
            # Injected pool: the file checksum cannot describe what was consumed.
            assert repro["episode_pool_source"] == "injected"
            assert repro["episode_pool_size"] == profile.pool_episodes
            assert repro["episode_pool_fingerprint"]

    manifest = load_manifest(tmp_path / "runs" / "tier0" / MANIFEST_NAME)
    assert manifest is not None
    assert manifest["interpretation"] == "plumbing_only"
    assert manifest["harness_step_cap"] == 1
    assert len(manifest["cells"]) == 4
    assert all(c["status"] == STATUS_COMPLETED for c in manifest["cells"])


@pytest.mark.integration
def test_micro_matrix_resume_skips_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    profile = get_budget_profile("tier0")
    tiny_pool = [[4 + (i % 20)] * 60 for i in range(profile.pool_episodes)]
    monkeypatch.setattr(script, "episodes_from_parquet", lambda *a, **k: tiny_pool)

    argv = [
        "--tier", "0",
        "--codecs", "A",
        "--replications", "1",
        "--output-root", str(tmp_path),
        "--smoke-max-steps", "1",
    ]
    assert script.main(argv) == 0
    run_dir = tmp_path / "runs" / "tier0" / "compute_matched" / "A" / "rep0"
    first_mtime = (run_dir / "checkpoints" / "final.pt").stat().st_mtime_ns

    assert script.main(argv) == 0
    assert (run_dir / "checkpoints" / "final.pt").stat().st_mtime_ns == first_mtime
