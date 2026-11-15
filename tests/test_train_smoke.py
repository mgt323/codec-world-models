"""Single-run smoke: Codec A, compute_matched, replication 0."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from model.architecture import SHARED_VOCAB_SIZE
from train.config import build_run_config, get_budget_profile, validate_run_config
from train.data import load_codec_vocab, synthetic_episodes
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
    assert isinstance(repro["git_dirty"], bool)
    assert "flops_estimate" in repro

    lines = (
        (run_dir / "metrics_train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(lines) == 2


def test_run_training_missing_parquet_raises_without_allow_synthetic(
    tmp_path: Path,
) -> None:
    cfg = build_run_config(
        codec="A",
        match_regime="compute_matched",
        replication_id=0,
        budget_profile="tier0",
    )
    with pytest.raises(FileNotFoundError, match="encoded train split not found"):
        run_training(
            cfg,
            output_root=tmp_path,
            encoded_dir=tmp_path / "missing_encoded",
            episode_pool=None,
            allow_synthetic=False,
        )


def test_run_training_direct_parquet_truncates_to_profile_pool_episodes(
    tmp_path: Path,
) -> None:
    profile = get_budget_profile("tier0")
    vocab = load_codec_vocab("A")
    toks = [k for k in vocab.token_to_id if not k.startswith("<")]
    sample_str = " ".join(toks[:3])

    enc_dir = tmp_path / "encoded"
    enc_dir.mkdir()
    parquet_path = enc_dir / "train_A.parquet"

    # Too few episodes -> ValueError.
    few_rows = [
        {"episode_seed": ep, "timestep": 0, "encoded_string": sample_str}
        for ep in range(profile.pool_episodes - 1)
    ]
    pd.DataFrame(few_rows).to_parquet(parquet_path, index=False)
    cfg = build_run_config(
        codec="A",
        match_regime="compute_matched",
        replication_id=0,
        budget_profile="tier0",
    )
    with pytest.raises(ValueError, match="requires exactly"):
        run_training(
            cfg,
            output_root=tmp_path / "out_few",
            encoded_dir=enc_dir,
            episode_pool=None,
            device="cpu",
            max_optimizer_steps=1,
            batch_size=1,
        )

    # More than pool_episodes -> truncated cleanly to profile.pool_episodes.
    many_rows = [
        {"episode_seed": ep, "timestep": 0, "encoded_string": sample_str}
        for ep in range(profile.pool_episodes + 7)
    ]
    pd.DataFrame(many_rows).to_parquet(parquet_path, index=False)
    res = run_training(
        cfg,
        output_root=tmp_path / "out_many",
        encoded_dir=enc_dir,
        episode_pool=None,
        device="cpu",
        max_optimizer_steps=1,
        batch_size=1,
    )
    repro = json.loads((res.run_dir / "reproducibility.json").read_text(encoding="utf-8"))
    assert repro["episode_pool_source"] == "parquet"
    assert repro["episode_pool_size"] == profile.pool_episodes
