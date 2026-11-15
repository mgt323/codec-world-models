"""Tests for GPU/SDPA optimizations, async loss logging, and checkpoint compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from eval.checkpoint import load_scoring_checkpoint
from model.architecture import (
    PAD_SEGMENT_ID,
    SHARED_VOCAB_SIZE,
    TransformerConfig,
    build_model,
)
from model.init import initialize_model
from obs_codecs.vocab import BOS_ID, EOS_ID, PAD_ID
from train.config import build_run_config
from train.data import synthetic_episodes
from train.loop import _next_token_loss, _save_checkpoint, run_training


def _additive_attention_forward(
    attn_mod: torch.nn.Module, x: torch.Tensor, allow: torch.Tensor
) -> torch.Tensor:
    batch, length, width = x.shape
    q, k, v = attn_mod._project_qkv(x)
    scale = attn_mod.head_dim**-0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    additive = torch.zeros(
        batch, 1, length, length, device=x.device, dtype=torch.float32
    ).masked_fill(~allow.unsqueeze(1), float("-inf"))
    probs = torch.softmax(scores + additive, dim=-1)
    y = torch.matmul(probs, v.float()).to(x.dtype)
    y = y.transpose(1, 2).contiguous().view(batch, length, width)
    return attn_mod.out_proj(y)


@pytest.mark.parametrize(
    "device_type",
    ["cpu", *(["cuda"] if torch.cuda.is_available() else [])],
)
def test_boolean_sdpa_matches_additive_reference_and_blocks_pads(
    device_type: str,
) -> None:
    device = torch.device(device_type)
    cfg = TransformerConfig(
        vocab_size=SHARED_VOCAB_SIZE,
        context_length=16,
        n_layers=2,
        d_model=24,
        n_heads=3,
        d_ff=48,
    )
    model_sdpa = build_model(cfg).float()
    model_ref = build_model(cfg).float()
    initialize_model(model_sdpa, model_init_seed=100, codec_name="A")
    model_ref.load_state_dict(model_sdpa.state_dict())
    model_sdpa = model_sdpa.to(device)
    model_ref = model_ref.to(device)

    input_ids = torch.tensor(
        [
            [BOS_ID, 10, 11, EOS_ID, BOS_ID, 12, 13, EOS_ID, PAD_ID, PAD_ID],
            [BOS_ID, 14, 15, 16, 17, EOS_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
        ],
        dtype=torch.long,
        device=device,
    )
    segment_ids = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 1, 1, PAD_SEGMENT_ID, PAD_SEGMENT_ID],
            [0, 0, 0, 0, 0, 0, PAD_SEGMENT_ID, PAD_SEGMENT_ID, PAD_SEGMENT_ID, PAD_SEGMENT_ID],
        ],
        dtype=torch.long,
        device=device,
    )

    logits_sdpa = model_sdpa(input_ids, segment_ids)
    loss_sdpa, _, _ = _next_token_loss(logits_sdpa, input_ids, segment_ids)
    loss_sdpa.backward()

    # Reference forward using explicit additive -inf softmax.
    x_ref = model_ref._embed(input_ids)
    allow_ref = model_ref.derive_allow_mask(segment_ids)
    for block in model_ref.blocks:
        x_ref = x_ref + _additive_attention_forward(block.attn, block.ln1(x_ref), allow_ref)
        x_ref = x_ref + block.mlp(block.ln2(x_ref))
    logits_ref = model_ref.lm_head(model_ref.ln_f(x_ref))
    loss_ref, _, _ = _next_token_loss(logits_ref, input_ids, segment_ids)
    loss_ref.backward()

    assert torch.allclose(logits_sdpa, logits_ref, atol=1e-5, rtol=1e-4)
    for (name, p_sdpa), (_, p_ref) in zip(
        model_sdpa.named_parameters(), model_ref.named_parameters(), strict=True
    ):
        assert p_sdpa.grad is not None and p_ref.grad is not None, name
        assert torch.isfinite(p_sdpa.grad).all(), name
        assert torch.allclose(p_sdpa.grad, p_ref.grad, atol=1e-4, rtol=1e-3), name

    # Verify zero attention weight across segments and into pad keys.
    probs = model_sdpa.attention_probs_at_layer(input_ids, segment_ids, layer_index=0)
    assert probs[0, :, 5, :4].abs().max().item() == 0.0
    assert probs[0, :, :8, 8:].abs().max().item() == 0.0


def test_compiled_model_checkpoint_saves_unwrapped_keys_and_loads_in_eval(
    tmp_path: Path,
) -> None:
    cfg = build_run_config(codec="A", match_regime="compute_matched", replication_id=0)
    raw_model = build_model()
    initialize_model(raw_model, model_init_seed=cfg.model_init_seed, codec_name=cfg.codec)
    compiled = torch.compile(raw_model, dynamic=False)

    ckpt_dir = tmp_path / "checkpoints"
    final_path = ckpt_dir / "final.pt"
    _save_checkpoint(final_path, compiled, scoring=True, step=5, config=cfg)

    payload = torch.load(final_path, map_location="cpu", weights_only=False)
    state_dict = payload["model_state_dict"]
    assert not any(k.startswith("_orig_mod.") for k in state_dict)
    assert all(v.dtype == torch.float32 for v in state_dict.values())

    loaded, _ = load_scoring_checkpoint(final_path, torch.device("cpu"))
    assert loaded.param_count() == raw_model.param_count()

    # Verify load_scoring_checkpoint rejects _orig_mod. keys explicitly.
    bad_path = ckpt_dir / "bad" / "final.pt"
    bad_path.parent.mkdir(parents=True)
    torch.save(
        {
            **payload,
            "model_state_dict": {f"_orig_mod.{k}": v for k, v in state_dict.items()},
        },
        bad_path,
    )
    with pytest.raises(ValueError, match="_orig_mod\\."):
        load_scoring_checkpoint(bad_path, torch.device("cpu"))


def test_batched_async_loss_logging_matches_per_step_reference(tmp_path: Path) -> None:
    cfg = build_run_config(
        codec="A",
        match_regime="compute_matched",
        replication_id=0,
        budget_profile="tier0",
    )
    # 50 windows at L=1024 -> batch_size=4 yields 13 steps (10 + 3, with final B=2 < 4).
    pool = synthetic_episodes(
        n_episodes=100,
        tokens_per_episode=500,
        vocab_size_native=42,
        seed=7,
    )
    res = run_training(
        cfg,
        output_root=tmp_path,
        device="cpu",
        episode_pool=pool,
        episode_limit=100,
        batch_size=4,
    )
    assert res.optimizer_steps == 13
    assert res.packing.last_batch_size == 2

    lines = (
        (res.run_dir / "metrics_train.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(lines) == 13
    rows = [json.loads(line) for line in lines]
    assert [r["step"] for r in rows] == list(range(1, 14))
    assert rows[-1]["batch_size"] == 2
    assert all(r["loss"] > 0.0 and r["loss_all_targets"] > 0.0 for r in rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA GPU required")
def test_rocm_bf16_compiled_step_smoke(tmp_path: Path) -> None:
    cfg = build_run_config(
        codec="A",
        match_regime="compute_matched",
        replication_id=0,
        budget_profile="tier0",
    )
    # 25 windows at L=1024 -> batch_size=8 yields 4 steps (3 compiled B=8 + 1 eager B=1).
    pool = synthetic_episodes(
        n_episodes=50,
        tokens_per_episode=500,
        vocab_size_native=42,
        seed=11,
    )
    res = run_training(
        cfg,
        output_root=tmp_path,
        device="rocm",
        episode_pool=pool,
        episode_limit=50,
        batch_size=8,
        precision="bf16_mixed",
        compile_model=True,
    )
    assert res.optimizer_steps == 4
    assert res.packing.last_batch_size == 1
    repro = json.loads((res.run_dir / "reproducibility.json").read_text(encoding="utf-8"))
    assert repro["precision"] == "bf16_mixed"
    assert repro["torch_compile"] is True
    assert repro["fused_adamw"] is True

    ckpt_path = res.run_dir / "checkpoints" / "final.pt"
    loaded, _ = load_scoring_checkpoint(ckpt_path, torch.device("cuda"))
    assert next(loaded.parameters()).dtype == torch.float32
