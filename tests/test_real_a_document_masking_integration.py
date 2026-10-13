"""Integration: real Codec A train data → packer → model document masking.

Checks the locked COMPUTE_MATCH_BUDGETS claim that attention cannot cross
episode segments inside a packed L=1024 window — on real encoded train rows,
not synthetic fixtures.

**Boundary count note (A @ L=1024):** mean Codec A episode length is ~2170
tokens, so a single window holds at most **one** episode boundary (two distinct
content ``segment_ids``). Two boundaries in one window would require ~3 episode
fragments within 1024 tokens, which is impossible under the locked L for A/B/C/D
(all are in split mode at 1024). This test therefore requires ≥1 real boundary
(≥2 distinct content segments) and asserts every cross-segment pair is masked /
near-zero in attention weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model.architecture import (
    CONTEXT_LENGTH_L,
    PAD_SEGMENT_ID,
    SHARED_VOCAB_SIZE,
    TransformerConfig,
    build_model,
)
from model.init import initialize_model
from train.data import episodes_from_parquet, load_codec_vocab
from train.packer import PackedWindow, pack_episodes

_TRAIN_A = Path("artifacts/encoded/train_A.parquet")
_VOCAB_DIR = Path("artifacts/vocabs")
# Softmax mass on blocked keys should be numerically ~0 (float32).
_ATTN_CROSS_SEG_TOL = 1e-6


def _content_segment_ids(window: PackedWindow) -> list[int]:
    return [
        int(s)
        for s in window.segment_ids.tolist()
        if int(s) != PAD_SEGMENT_ID
    ]


def _n_episode_boundaries(window: PackedWindow) -> int:
    """Count segment changes along the content prefix (pad excluded)."""
    segs = _content_segment_ids(window)
    if not segs:
        return 0
    changes = 0
    for a, b in zip(segs, segs[1:]):
        if a != b:
            changes += 1
    return changes


def _find_window_with_boundary(
    windows: list[PackedWindow],
) -> PackedWindow:
    for w in windows:
        if _n_episode_boundaries(w) >= 1:
            return w
    raise AssertionError(
        "No packed L=1024 window from real Codec A train data contained an "
        "episode boundary; cannot validate document masking on real data."
    )


@pytest.mark.integration
def test_real_codec_a_packer_model_cross_segment_attention_blocked() -> None:
    if not _TRAIN_A.is_file():
        pytest.skip(f"missing real encoded train data: {_TRAIN_A}")
    if not (_VOCAB_DIR / "vocab_A.json").is_file():
        pytest.skip(f"missing vocab_A.json under {_VOCAB_DIR}")

    vocab = load_codec_vocab("A", _VOCAB_DIR)
    # Enough episodes to guarantee at least one mid-stream episode junction
    # lands inside an L-window (A ≈ 2170 tok/ep → junction every ~3 windows).
    episodes = episodes_from_parquet(_TRAIN_A, vocab, max_episodes=12)
    assert len(episodes) >= 2
    assert all(len(ep) > CONTEXT_LENGTH_L for ep in episodes[:2]), (
        "expected real A episodes longer than L (split mode)"
    )

    windows = pack_episodes(episodes, context_length=CONTEXT_LENGTH_L)
    assert windows, "packer produced no windows from real A episodes"

    # Literal "≥2 boundaries / window" is unreachable for A@L=1024 (documented
    # above). Record that fact, then validate on a real ≥1-boundary window.
    max_boundaries = max(_n_episode_boundaries(w) for w in windows)
    assert max_boundaries < 2, (
        "unexpected: A@L=1024 produced ≥2 boundaries in one window — "
        "revisit density assumptions in COMPUTE_MATCH_BUDGETS"
    )
    window = _find_window_with_boundary(windows)
    assert _n_episode_boundaries(window) == 1
    distinct = sorted(set(_content_segment_ids(window)))
    assert len(distinct) == 2

    input_ids = window.input_ids.unsqueeze(0)
    segment_ids = window.segment_ids.unsqueeze(0)
    assert input_ids.shape == (1, CONTEXT_LENGTH_L)

    model = build_model(
        TransformerConfig(
            vocab_size=SHARED_VOCAB_SIZE,
            context_length=CONTEXT_LENGTH_L,
        )
    )
    initialize_model(model, model_init_seed=1_000_000, codec_name="A")
    model.eval()

    logits = model(input_ids, segment_ids)
    assert logits.shape == (1, CONTEXT_LENGTH_L, SHARED_VOCAB_SIZE)
    assert model.last_allow_mask is not None
    allow = model.last_allow_mask
    assert allow.shape == (1, CONTEXT_LENGTH_L, CONTEXT_LENGTH_L)

    seg = segment_ids[0]
    cross = (seg.unsqueeze(1) != seg.unsqueeze(0)) & (
        seg.unsqueeze(1) != PAD_SEGMENT_ID
    ) & (seg.unsqueeze(0) != PAD_SEGMENT_ID)
    # Mask source of truth from the forward that just ran: no cross-segment allow.
    assert not bool(allow[0][cross].any()), (
        "forward-derived allow mask permits cross-segment attention on real A data"
    )

    # Actual softmax weights at layer 0 under the same mask path.
    probs = model.attention_probs_at_layer(
        input_ids, segment_ids, layer_index=0
    )
    assert probs.shape[0] == 1
    cross_mass = probs[0][:, cross]
    assert torch.isfinite(cross_mass).all()
    max_leak = float(cross_mass.max().item())
    assert max_leak < _ATTN_CROSS_SEG_TOL, (
        f"cross-segment attention mass {max_leak} on real Codec A packed window "
        f"(tol={_ATTN_CROSS_SEG_TOL})"
    )

    # Within-segment causal attention must still be usable (not a trivial all-zero).
    same = (seg.unsqueeze(1) == seg.unsqueeze(0)) & (
        seg.unsqueeze(0) != PAD_SEGMENT_ID
    )
    causal_same = same & torch.tril(
        torch.ones(CONTEXT_LENGTH_L, CONTEXT_LENGTH_L, dtype=torch.bool)
    )
    assert bool(allow[0][causal_same].all())
    assert float(probs[0][:, causal_same].sum().item()) > 0.0
