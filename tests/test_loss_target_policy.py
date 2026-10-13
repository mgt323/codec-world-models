"""Lock the next-token loss target policy and its per-run diagnostics.

`COMPUTE_MATCH_BUDGETS.md` §4 item 4 accepts cross-segment boundary targets as
a measured residual rather than masking them. That is a methodological choice,
so the default must not drift silently, and the diagnostic counters that make
the residual auditable must stay correct.
"""

from __future__ import annotations

import torch

from model.architecture import PAD_SEGMENT_ID, SHARED_VOCAB_SIZE
from train.loop import _next_token_loss

L = 8


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One window: segment 0 (4 tokens), segment 1 (2 tokens), then 2 pads."""
    segment_ids = torch.tensor([[0, 0, 0, 0, 1, 1, PAD_SEGMENT_ID, PAD_SEGMENT_ID]])
    input_ids = torch.tensor([[1, 5, 6, 2, 1, 2, 0, 0]])
    torch.manual_seed(0)
    logits = torch.randn(1, L, SHARED_VOCAB_SIZE)
    return logits, input_ids, segment_ids


def test_default_policy_scores_cross_segment_targets() -> None:
    """Locked default: every non-pad target counts, boundary pairs included."""
    logits, input_ids, segment_ids = _fixture()
    _loss, n_selected, diag = _next_token_loss(logits, input_ids, segment_ids)

    # Shifted targets are positions 1..7: segments [0,0,0,1,1,PAD,PAD].
    # Non-pad targets = 5; of those, the 0->1 transition is cross-segment.
    assert n_selected == 5
    assert diag["n_cross"] == 1
    assert diag["n_within"] == 4
    assert n_selected == diag["n_within"] + diag["n_cross"]


def test_within_segment_policy_drops_only_boundary_targets() -> None:
    logits, input_ids, segment_ids = _fixture()
    _loss, n_selected, diag = _next_token_loss(
        logits, input_ids, segment_ids, restrict_to_segment=True
    )
    assert n_selected == 4
    assert diag["n_within"] == 4
    assert diag["n_cross"] == 1


def test_reported_losses_reconcile_as_weighted_mean() -> None:
    """`loss_all` must equal the count-weighted mix of within and cross."""
    logits, input_ids, segment_ids = _fixture()
    loss, _n, diag = _next_token_loss(logits, input_ids, segment_ids)

    n_w, n_c = diag["n_within"], diag["n_cross"]
    expected = (
        diag["loss_within_segment"] * n_w + diag["loss_cross_segment"] * n_c
    ) / (n_w + n_c)
    assert abs(diag["loss_all_targets"] - expected) < 1e-5
    assert abs(float(loss) - diag["loss_all_targets"]) < 1e-6


def test_selected_loss_follows_the_policy() -> None:
    logits, input_ids, segment_ids = _fixture()
    _l, _n, diag = _next_token_loss(logits, input_ids, segment_ids)
    loss_within, _n2, _d2 = _next_token_loss(
        logits, input_ids, segment_ids, restrict_to_segment=True
    )
    assert abs(float(loss_within) - diag["loss_within_segment"]) < 1e-6
    # The two policies must actually differ, else the residual is unmeasurable.
    assert abs(diag["loss_all_targets"] - diag["loss_within_segment"]) > 0


def test_all_pad_window_yields_no_targets() -> None:
    segment_ids = torch.full((1, L), PAD_SEGMENT_ID)
    input_ids = torch.zeros(1, L, dtype=torch.long)
    logits = torch.zeros(1, L, SHARED_VOCAB_SIZE)
    loss, n_selected, _diag = _next_token_loss(logits, input_ids, segment_ids)
    assert n_selected == 0
    assert float(loss) == 0.0
