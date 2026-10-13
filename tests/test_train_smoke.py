"""Single-run smoke: Codec A, compute_matched, replication 0."""

from __future__ import annotations

import json
from pathlib import Path

from model.architecture import SHARED_VOCAB_SIZE
from train.config import build_run_config, validate_run_config
from train.data import synthetic_episodes
from train.loop import run_training


def test_run_config_a_compute_matched_rep0_validates() -> None:
    cfg = build_run_config(
        codec="A", match_regime="compute_matched", replication_id=0
    )
    assert cfg.model_init_seed == 1_000_000
    assert cfg.data_shuffle_seed == 2_000_000
    assert cfg.n_episodes == 50_000
    assert cfg.context_length_L == 1024
    validate_run_config(cfg)


def test_single_run_smoke_writes_artifacts(tmp_path: Path) -> None:
    cfg = build_run_config(
        codec="A", match_regime="compute_matched", replication_id=0
    )
    # Enough tokens for >=2 L=1024 windows so batch_size=1 yields two steps.
    pool = synthetic_episodes(
        n_episodes=8,
        tokens_per_episode=300,
        vocab_size_native=42,
        seed=0,
    )
    result = run_training(
        cfg,
        output_root=tmp_path,
        episode_pool=pool,
        episode_limit=8,
        max_optimizer_steps=2,
        batch_size=1,
    )

    run_dir = result.run_dir
    assert run_dir == tmp_path / "runs" / "compute_matched" / "A" / "rep0"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics_train.jsonl").is_file()
    assert (run_dir / "reproducibility.json").is_file()
    assert (run_dir / "checkpoints" / "final.pt").is_file()

    config_disk = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config_disk["codec"] == "A"
    assert config_disk["match_regime"] == "compute_matched"
    assert config_disk["replication_id"] == 0

    repro = json.loads(
        (run_dir / "reproducibility.json").read_text(encoding="utf-8")
    )
    assert repro["model_param_count"] == result.model_param_count
    assert repro["shared_vocab_size"] == SHARED_VOCAB_SIZE
    assert repro["vocab_pad_policy"] == "shared_max_pad_48"
    assert repro["scoring_checkpoint"] == "final"
    assert repro["optimizer_steps"] == 2
    assert "git_commit" in repro
    assert "flops_estimate" in repro

    lines = (
        (run_dir / "metrics_train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(lines) == 2
