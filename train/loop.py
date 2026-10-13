"""Stage-1 next-token training loop (single run).

Consumes only ``RunConfig`` fields from TRAINING_RUNNER_CONTRACT.md §2.
Stopping criterion: exactly ``n_episodes`` episode presentations.
Scoring checkpoint: final only; intermediate files tagged ``non_scoring``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from model.architecture import (
    PAD_SEGMENT_ID,
    SHARED_VOCAB_SIZE,
    TransformerConfig,
    build_model,
    estimate_flops_proxy_token_updates,
)
from model.init import initialize_model
from train.config import (
    ADAM_BETAS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RunConfig,
    validate_run_config,
)
from train.data import (
    episodes_from_parquet,
    file_sha256,
    load_codec_vocab,
    pool_fingerprint,
    present_episode_order,
    select_episodes,
    synthetic_episodes,
)
from train.packer import iter_packed_batches, pack_episodes


@dataclass(frozen=True, slots=True)
class PackingAccount:
    """Observed packing / batching stats for one run (vs projection formula)."""

    stream_tokens: int
    n_windows: int
    batch_size: int
    last_batch_size: int
    projected_optimizer_steps: int
    observed_optimizer_steps: int
    partial_last_batch_included: bool


@dataclass(frozen=True, slots=True)
class TrainResult:
    run_dir: Path
    model_param_count: int
    optimizer_steps: int
    tokens_consumed: int
    episodes_presented: int
    packing: PackingAccount


def project_optimizer_steps(
    stream_tokens: int,
    *,
    context_length: int,
    batch_size: int,
) -> tuple[int, int]:
    """Return ``(n_windows, n_steps)`` using ceil packing / ceil batching."""
    if stream_tokens < 0:
        raise ValueError("stream_tokens must be non-negative")
    if context_length < 1 or batch_size < 1:
        raise ValueError("context_length and batch_size must be positive")
    if stream_tokens == 0:
        return 0, 0
    n_windows = (stream_tokens + context_length - 1) // context_length
    n_steps = (n_windows + batch_size - 1) // batch_size
    return n_windows, n_steps


def run_dir_for(
    config: RunConfig,
    output_root: Path | str,
    *,
    runs_subdir: str | None = None,
) -> Path:
    """Contract §4.3 layout, optionally nested under a tier/profile segment."""
    base = Path(output_root) / "runs"
    if runs_subdir:
        base = base / runs_subdir
    return base / config.match_regime / config.codec / f"rep{config.replication_id}"


def config_sha256(config: RunConfig) -> str:
    """Stable hash of the full config dump (contract §2.4)."""
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _next_token_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    restrict_to_segment: bool = False,
) -> tuple[torch.Tensor, int, dict[str, float | int]]:
    """Causal LM loss; ignore pad positions and the final position per row.

    A target at position ``t+1`` is *cross-segment* when it belongs to a
    different episode than the source position ``t`` — i.e. the ``<eos> -> <bos>``
    transition at a packed episode boundary. Attention never crosses that
    boundary, so such a target is trivially predictable, and its density is a
    function of packing density (hence codec-dependent).

    ``restrict_to_segment`` selects which mask is *optimized*. Both variants are
    always measured and returned so the residual can be audited from any run.
    """
    # Predict token t+1 from position t.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_seg = segment_ids[:, 1:].contiguous()
    src_seg = segment_ids[:, :-1].contiguous()

    valid = shift_seg != PAD_SEGMENT_ID
    within = valid & (shift_seg == src_seg)
    cross = valid & (shift_seg != src_seg)

    selected = within if restrict_to_segment else valid
    n_selected = int(selected.sum().item())
    n_within = int(within.sum().item())
    n_cross = int(cross.sum().item())
    if n_selected == 0:
        return logits.sum() * 0.0, 0, {"n_within": n_within, "n_cross": n_cross}

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view_as(shift_labels)

    diagnostics: dict[str, float | int] = {
        "n_within": n_within,
        "n_cross": n_cross,
        "loss_all_targets": float(loss.masked_select(valid).mean().detach().cpu())
        if bool(valid.any())
        else 0.0,
        "loss_within_segment": float(
            loss.masked_select(within).mean().detach().cpu()
        )
        if n_within
        else 0.0,
        "loss_cross_segment": float(loss.masked_select(cross).mean().detach().cpu())
        if n_cross
        else 0.0,
    }
    return loss.masked_select(selected).mean(), n_selected, diagnostics


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    scoring: bool,
    step: int,
    config: RunConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "scoring": scoring,
        "step": step,
        "codec": config.codec,
        "match_regime": config.match_regime,
        "replication_id": config.replication_id,
    }
    torch.save(payload, path)


def run_training(
    config: RunConfig,
    *,
    output_root: Path | str = "artifacts",
    device: str | torch.device | None = None,
    episode_pool: Sequence[Sequence[int]] | None = None,
    episode_limit: int | None = None,
    max_optimizer_steps: int | None = None,
    batch_size: int | None = None,
    save_intermediate_every: int | None = None,
    runs_subdir: str | None = None,
    restrict_loss_to_segment: bool = False,
) -> TrainResult:
    """Train one run.

    ``episode_limit``, ``max_optimizer_steps``, and ``batch_size`` are
    harness-only overrides for smoke tests — not RunConfig fields.
    ``episode_pool`` lets a caller inject an already-tokenized pool (the matrix
    runner caches one pool per codec); ``runs_subdir`` nests the run directory
    under a tier/profile segment.

    ``restrict_loss_to_segment`` is a **diagnostic** knob for sizing the
    cross-segment-target residual; it defaults to the locked behaviour and is
    not a ``RunConfig`` field. Adopting it would be a methodological change and
    must be applied to every codec at once.
    """
    validate_run_config(config)
    if not config.document_masking:
        raise ValueError("document_masking must be true")

    n_target = config.n_episodes if episode_limit is None else episode_limit
    if n_target < 1:
        raise ValueError("episode presentation count must be positive")
    bs = BATCH_SIZE if batch_size is None else batch_size
    if bs < 1:
        raise ValueError("batch_size must be positive")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = run_dir_for(config, output_root, runs_subdir=runs_subdir)
    ckpt_dir = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config.to_json(out / "config.json")

    vocab = load_codec_vocab(config.codec)
    native_size = vocab.vocab_size
    if native_size > SHARED_VOCAB_SIZE:
        raise ValueError(
            f"codec {config.codec} vocab_size {native_size} exceeds pad "
            f"{SHARED_VOCAB_SIZE}"
        )

    dataset_path = Path(config.dataset_split_train.format(codec=config.codec))
    pool_source = "injected"
    if episode_pool is None:
        if dataset_path.exists():
            episode_pool = episodes_from_parquet(dataset_path, vocab)
            pool_source = "parquet"
        else:
            episode_pool = synthetic_episodes(
                n_episodes=min(32, n_target),
                tokens_per_episode=64,
                vocab_size_native=native_size,
                seed=config.data_shuffle_seed,
            )
            pool_source = "synthetic"

    pool_list = [list(ep) for ep in episode_pool]
    order = present_episode_order(
        len(pool_list), n_target, config.data_shuffle_seed
    )
    presented = select_episodes(pool_list, order)

    model = build_model(
        TransformerConfig(
            vocab_size=SHARED_VOCAB_SIZE,
            context_length=config.context_length_L,
        )
    )
    initialize_model(
        model,
        model_init_seed=config.model_init_seed,
        codec_name=config.codec,
    )
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=ADAM_BETAS,
    )

    metrics_path = out / "metrics_train.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    t0 = time.perf_counter()
    optimizer_steps = 0
    tokens_consumed = 0
    # Pack all presentations then train — for full budgets this is large;
    # smoke tests use tiny episode_limit. Streaming packer can replace later.
    windows = pack_episodes(
        presented, context_length=config.context_length_L
    )
    stream_tokens = sum(w.n_content_tokens for w in windows)
    n_windows = len(windows)
    projected_windows, projected_steps = project_optimizer_steps(
        stream_tokens,
        context_length=config.context_length_L,
        batch_size=bs,
    )
    if projected_windows != n_windows:
        raise RuntimeError(
            f"packer window count {n_windows} != ceil(stream/L) "
            f"projection {projected_windows} (stream_tokens={stream_tokens})"
        )
    # Partial final batches are yielded (no drop_last) — matches ceil(windows/bs).
    batch_sizes: list[int] = []

    for input_ids, segment_ids in iter_packed_batches(
        windows, batch_size=bs
    ):
        batch_sizes.append(int(input_ids.shape[0]))
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        logits = model(input_ids, segment_ids)
        loss, n_tok, loss_diag = _next_token_loss(
            logits,
            input_ids,
            segment_ids,
            restrict_to_segment=restrict_loss_to_segment,
        )
        if n_tok == 0:
            # Still count as an optimizer step opportunity skipped — should not
            # happen for real packed content; raise so accounting cannot drift.
            raise RuntimeError(
                "batch produced zero loss tokens; step accounting would diverge"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer_steps += 1
        tokens_consumed += n_tok

        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "step": optimizer_steps,
                        "loss": float(loss.detach().cpu()),
                        "tokens_in_step": n_tok,
                        "tokens_consumed": tokens_consumed,
                        "batch_size": batch_sizes[-1],
                        **loss_diag,
                    }
                )
                + "\n"
            )

        if (
            save_intermediate_every is not None
            and optimizer_steps % save_intermediate_every == 0
        ):
            _save_checkpoint(
                ckpt_dir / f"step_{optimizer_steps}_non_scoring.pt",
                model,
                scoring=False,
                step=optimizer_steps,
                config=config,
            )

        if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
            break

    if max_optimizer_steps is None and optimizer_steps != projected_steps:
        raise RuntimeError(
            f"observed optimizer_steps {optimizer_steps} != projected "
            f"ceil(windows/batch_size)={projected_steps} "
            f"(n_windows={n_windows}, batch_size={bs}, "
            f"batch_sizes={batch_sizes[:3]}...{batch_sizes[-3:]})"
        )

    packing = PackingAccount(
        stream_tokens=stream_tokens,
        n_windows=n_windows,
        batch_size=bs,
        last_batch_size=batch_sizes[-1] if batch_sizes else 0,
        projected_optimizer_steps=projected_steps,
        observed_optimizer_steps=optimizer_steps,
        # iter_packed_batches has no drop_last; partial finals are trained.
        partial_last_batch_included=True,
    )
    _save_checkpoint(
        ckpt_dir / "final.pt",
        model,
        scoring=True,
        step=optimizer_steps,
        config=config,
    )

    param_count = model.param_count()
    flops = estimate_flops_proxy_token_updates(
        tokens_consumed, context_length=config.context_length_L
    )
    flops["approx_6nd"] = 6 * param_count * tokens_consumed

    reproducibility: dict[str, Any] = {
        "git_commit": git_commit(),
        "config": config.to_dict(),
        "config_sha256": config_sha256(config),
        "budget_profile": config.budget_profile,
        "dataset_split_train": dataset_path.as_posix(),
        "dataset_checksum": file_sha256(dataset_path),
        "episode_pool_source": pool_source,
        "episode_pool_size": len(pool_list),
        "episode_pool_fingerprint": pool_fingerprint(pool_list),
        "model_init_seed": config.model_init_seed,
        "data_shuffle_seed": config.data_shuffle_seed,
        "codec": config.codec,
        "match_regime": config.match_regime,
        "replication_id": config.replication_id,
        "model_param_count": param_count,
        "shared_vocab_size": SHARED_VOCAB_SIZE,
        "native_vocab_size": native_size,
        "vocab_pad_policy": "shared_max_pad_48",
        "tokens_consumed": tokens_consumed,
        "optimizer_steps": optimizer_steps,
        "episodes_presented": n_target,
        "packing_account": {
            "stream_tokens": packing.stream_tokens,
            "n_windows": packing.n_windows,
            "batch_size": packing.batch_size,
            "last_batch_size": packing.last_batch_size,
            "projected_optimizer_steps": packing.projected_optimizer_steps,
            "observed_optimizer_steps": packing.observed_optimizer_steps,
            "partial_last_batch_included": packing.partial_last_batch_included,
            "formula": "windows=ceil(stream/L); steps=ceil(windows/batch_size); no drop_last",
        },
        "flops_estimate": flops,
        "wall_clock_seconds": time.perf_counter() - t0,
        "scoring_checkpoint": "final",
        "harness_step_cap": max_optimizer_steps,
        "loss_target_policy": (
            "within_segment" if restrict_loss_to_segment else "all_non_pad"
        ),
        "built_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return TrainResult(
        run_dir=out,
        model_param_count=param_count,
        optimizer_steps=optimizer_steps,
        tokens_consumed=tokens_consumed,
        episodes_presented=n_target,
        packing=packing,
    )
