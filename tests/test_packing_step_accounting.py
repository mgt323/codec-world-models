"""Verify packer window / optimizer-step accounting at real batch_size=8.

Confirms ``ceil(stream/L)`` / ``ceil(windows/batch_size)`` against observed
behavior (partial last batch included — no drop_last).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from model.architecture import CONTEXT_LENGTH_L
from train.config import BATCH_SIZE, build_run_config
from train.data import (
    episodes_from_parquet,
    load_codec_vocab,
    present_episode_order,
    select_episodes,
)
from train.loop import project_optimizer_steps, run_training
from train.packer import PackedWindow, iter_packed_batches, pack_episodes

_TRAIN_A = Path("artifacts/encoded/train_A.parquet")
_VOCAB_DIR = Path("artifacts/vocabs")


def _require_artifacts() -> None:
    if not _TRAIN_A.is_file():
        pytest.skip(f"missing {_TRAIN_A}")
    if not (_VOCAB_DIR / "vocab_A.json").is_file():
        pytest.skip("missing vocab_A.json")


def test_iter_packed_batches_includes_partial_last_batch() -> None:
    """Unit guard on batching policy: no drop_last."""
    windows = [
        PackedWindow(
            input_ids=torch.zeros(4, dtype=torch.long),
            segment_ids=torch.zeros(4, dtype=torch.long),
            n_content_tokens=4,
        )
        for _ in range(10)
    ]
    batches = list(iter_packed_batches(windows, batch_size=8))
    assert len(batches) == 2
    assert batches[0][0].shape[0] == 8
    assert batches[1][0].shape[0] == 2
    assert math.ceil(10 / 8) == 2
    assert math.floor(10 / 8) == 1  # floor would silently drop the remainder


@pytest.mark.integration
def test_pack_10k_presentations_matches_ceil_formula() -> None:
    """Observed windows for 10k A presentations match ceil(stream/L)."""
    _require_artifacts()
    vocab = load_codec_vocab("A", _VOCAB_DIR)
    pool = episodes_from_parquet(_TRAIN_A, vocab)  # dry-run pool (~2k)
    n_present = 10_000
    cfg = build_run_config(
        codec="A", match_regime="compute_matched", replication_id=0
    )
    order = present_episode_order(len(pool), n_present, cfg.data_shuffle_seed)
    presented = select_episodes(pool, order)
    windows = pack_episodes(presented, context_length=CONTEXT_LENGTH_L)
    stream = sum(w.n_content_tokens for w in windows)
    proj_win, proj_steps = project_optimizer_steps(
        stream, context_length=CONTEXT_LENGTH_L, batch_size=BATCH_SIZE
    )
    assert len(windows) == proj_win == math.ceil(stream / CONTEXT_LENGTH_L)
    assert proj_steps == math.ceil(proj_win / BATCH_SIZE)
    assert proj_win % BATCH_SIZE != 0, (
        "10k slice happened to be divisible by batch_size; "
        "re-check ceil/floor discrimination"
    )
    out = Path("artifacts/runs/_accounting/a_10k_pack_projection.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "episodes_presented": n_present,
                "pool_size": len(pool),
                "stream_tokens": stream,
                "n_windows_observed": len(windows),
                "n_windows_projected": proj_win,
                "optimizer_steps_projected_bs8": proj_steps,
                "optimizer_steps_if_drop_last_floor": proj_win // BATCH_SIZE,
                "windows_mod_batch_size": proj_win % BATCH_SIZE,
                "batch_size": BATCH_SIZE,
                "L": CONTEXT_LENGTH_L,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.integration
def test_real_bs8_train_slice_matches_ceil_projection(tmp_path: Path) -> None:
    """Actual train at BATCH_SIZE=8: observed steps == ceil(windows/8).

    Uses a substantial real-A slice (enough windows that windows % 8 != 0).
    Full 10k presentation training is ~hours on CPU; packing for 10k is
    checked separately in ``test_pack_10k_presentations_matches_ceil_formula``.
    """
    _require_artifacts()
    vocab = load_codec_vocab("A", _VOCAB_DIR)
    pool = episodes_from_parquet(_TRAIN_A, vocab)
    cfg = build_run_config(
        codec="A", match_regime="compute_matched", replication_id=0
    )

    n_present = None
    for candidate in (64, 80, 96, 112, 128, 160, 192, 224, 256):
        order = present_episode_order(
            len(pool), candidate, cfg.data_shuffle_seed
        )
        presented = select_episodes(pool, order)
        n_win = len(pack_episodes(presented, context_length=CONTEXT_LENGTH_L))
        if n_win % BATCH_SIZE != 0:
            n_present = candidate
            break
    assert n_present is not None, "could not find slice with windows % 8 != 0"

    result = run_training(
        cfg,
        output_root=tmp_path,
        episode_pool=pool,
        episode_limit=n_present,
        batch_size=BATCH_SIZE,
    )
    packing = result.packing
    assert packing.batch_size == BATCH_SIZE
    assert packing.n_windows % BATCH_SIZE != 0
    assert packing.last_batch_size == packing.n_windows % BATCH_SIZE
    assert packing.last_batch_size < BATCH_SIZE
    assert packing.partial_last_batch_included is True
    assert packing.observed_optimizer_steps == packing.projected_optimizer_steps
    assert packing.observed_optimizer_steps == math.ceil(
        packing.n_windows / BATCH_SIZE
    )
    assert packing.observed_optimizer_steps != packing.n_windows // BATCH_SIZE

    repro = json.loads(
        (result.run_dir / "reproducibility.json").read_text(encoding="utf-8")
    )
    assert repro["packing_account"]["observed_optimizer_steps"] == (
        packing.observed_optimizer_steps
    )
