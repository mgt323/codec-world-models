"""Packer segment isolation and attention-mask leak tests (Decision 1)."""

from __future__ import annotations

import torch

from model.architecture import PAD_SEGMENT_ID, attention_allow_mask
from obs_codecs.vocab import BOS_ID, EOS_ID, PAD_ID
from train.packer import pack_episodes, pack_stream_to_windows


def test_packer_segment_ids_increment_at_episode_boundaries() -> None:
    # Two short episodes; L large enough to hold both in one window.
    episodes = [
        [10, 11, 12],
        [20, 21],
    ]
    windows = pack_episodes(episodes, context_length=16)
    assert len(windows) == 1
    w = windows[0]
    # [bos,10,11,12,eos] seg0 + [bos,20,21,eos] seg1 + pads
    expected_tokens = [
        BOS_ID,
        10,
        11,
        12,
        EOS_ID,
        BOS_ID,
        20,
        21,
        EOS_ID,
    ]
    n = len(expected_tokens)
    assert w.input_ids[:n].tolist() == expected_tokens
    assert w.segment_ids[:5].tolist() == [0, 0, 0, 0, 0]
    assert w.segment_ids[5:9].tolist() == [1, 1, 1, 1]
    assert w.segment_ids[n:].tolist() == [PAD_SEGMENT_ID] * (16 - n)
    assert w.input_ids[n:].tolist() == [PAD_ID] * (16 - n)


def test_attention_mask_blocks_cross_segment_and_pad() -> None:
    tokens = [BOS_ID, 10, EOS_ID, BOS_ID, 20, EOS_ID]
    segs = [0, 0, 0, 1, 1, 1]
    windows = pack_stream_to_windows(tokens, segs, context_length=8)
    assert len(windows) == 1
    seg = windows[0].segment_ids.unsqueeze(0)  # (1, 8)
    allow = attention_allow_mask(seg)[0]  # (8, 8)

    # Position 4 (seg1, token 20) must not attend into seg0 positions 0..2.
    for key in (0, 1, 2):
        assert not bool(allow[4, key]), f"cross-segment leak at (4,{key})"

    # Causal within segment: pos 2 (eos seg0) may attend 0,1,2; not future.
    assert bool(allow[2, 0]) and bool(allow[2, 1]) and bool(allow[2, 2])
    assert not bool(allow[1, 2])  # no forward attention

    # Content must not attend to pad keys (positions 6,7).
    for query in range(6):
        assert not bool(allow[query, 6])
        assert not bool(allow[query, 7])

    # Diagonal always allowed (numerical stability for pad rows).
    assert bool(allow[6, 6]) and bool(allow[7, 7])


def test_long_episode_split_preserves_segment_within_window() -> None:
    episode = list(range(100, 120))  # 20 content tokens
    windows = pack_episodes([episode], context_length=8)
    # bos + 20 + eos = 22 tokens -> 3 windows (8+8+6padded)
    assert len(windows) == 3
    # First window is contiguous fragment of segment 0 only.
    assert set(windows[0].segment_ids.tolist()) == {0}
    # Last window: remaining content + pads.
    last = windows[2]
    content = last.segment_ids[last.segment_ids != PAD_SEGMENT_ID]
    assert set(content.tolist()) == {0}
    assert PAD_SEGMENT_ID in last.segment_ids.tolist()
