"""Stage-1 next-token training loop (single run).

Consumes only ``RunConfig`` fields from docs/TRAINING_RUNNER_CONTRACT.md §2.
Stopping criterion: exactly ``n_episodes`` episode presentations.
Scoring checkpoint: final only; intermediate files tagged ``non_scoring``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

from configs.dataset_v0 import BASE_SEED_TRAIN
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
    get_budget_profile,
    validate_run_config,
)
from train.device import (
    PrecisionMode,
    hardware_metadata,
    resolve_device,
    resolve_precision,
)
from train.data import (
    episodes_from_parquet,
    file_sha256,
    iter_present_episode_order,
    load_codec_vocab,
    pool_fingerprint,
    synthetic_episodes,
)
from train.packer import iter_packed_batches_streaming, prefetch_batches
from world.schema import SCHEMA_VERSION

LOG_FLUSH_INTERVAL = 10


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
    wall_clock_seconds: float | None = None


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


def git_dirty() -> bool:
    """True when the worktree has uncommitted changes (invariant I13).

    A missing git binary is recorded as dirty so a result is not labeled clean
    when provenance could not be checked.
    """
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True
    return bool(output.strip())


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
        shift_logits.float().reshape(-1, shift_logits.size(-1)),
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


def _next_token_loss_step(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    restrict_to_segment: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static-shape FP32 loss and detached diagnostic vector ``(4,)`` on device."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_seg = segment_ids[:, 1:]
    src_seg = segment_ids[:, :-1]

    valid = shift_seg != PAD_SEGMENT_ID
    within = valid & (shift_seg == src_seg)
    cross = valid & (shift_seg != src_seg)
    selected = within if restrict_to_segment else valid

    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view_as(shift_labels)

    valid_f = valid.float()
    within_f = within.float()
    cross_f = cross.float()
    selected_f = selected.float()

    selected_loss = (token_loss * selected_f).sum() / selected_f.sum().clamp_min(1.0)
    valid_mean = (token_loss.detach() * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    within_mean = (token_loss.detach() * within_f).sum() / within_f.sum().clamp_min(1.0)
    cross_mean = (token_loss.detach() * cross_f).sum() / cross_f.sum().clamp_min(1.0)
    diag_gpu = torch.stack([selected_loss.detach(), valid_mean, within_mean, cross_mean])
    return selected_loss, diag_gpu


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap ``torch.compile`` (``_orig_mod``) wrappers to the base module."""
    raw: torch.nn.Module = model
    while hasattr(raw, "_orig_mod"):
        raw = getattr(raw, "_orig_mod")
    return raw


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
        "model_state_dict": unwrap_model(model).state_dict(),
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
    allow_synthetic: bool = False,
    vocab_dir: Path | str = "artifacts/vocabs",
    encoded_dir: Path | str | None = None,
    precision: PrecisionMode | Literal["auto"] = "auto",
    compile_model: bool | None = None,
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

    ``allow_synthetic`` is test-only. A missing parquet raises
    ``FileNotFoundError`` unless it is set. Direct parquet loads are truncated
    to the profile's ``pool_episodes``.
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

    if device is None:
        device = resolve_device("auto")
    elif isinstance(device, str):
        if device in ("auto", "rocm", "cuda", "cpu"):
            device = resolve_device(device)  # type: ignore[arg-type]
        else:
            device = torch.device(device)
    resolved_precision = resolve_precision(device, precision)
    dataset_path = (
        Path(encoded_dir) / f"train_{config.codec}.parquet"
        if encoded_dir is not None
        else Path(config.dataset_split_train.format(codec=config.codec))
    )
    if episode_pool is None and not dataset_path.is_file() and not allow_synthetic:
        raise FileNotFoundError(
            f"encoded train split not found: {dataset_path}. "
            "Pass allow_synthetic=True only from unit tests."
        )

    out = run_dir_for(config, output_root, runs_subdir=runs_subdir)
    ckpt_dir = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config.to_json(out / "config.json")

    vocab = load_codec_vocab(config.codec, vocab_dir)
    native_size = vocab.vocab_size
    if native_size > SHARED_VOCAB_SIZE:
        raise ValueError(
            f"codec {config.codec} vocab_size {native_size} exceeds pad "
            f"{SHARED_VOCAB_SIZE}"
        )

    pool_source = "injected"
    if episode_pool is None:
        profile = get_budget_profile(config.budget_profile)
        if dataset_path.is_file():
            episode_pool = episodes_from_parquet(
                dataset_path, vocab, max_episodes=profile.pool_episodes
            )
            if len(episode_pool) != profile.pool_episodes:
                raise ValueError(
                    f"loaded pool has {len(episode_pool)} episodes, profile "
                    f"{profile.name} requires exactly {profile.pool_episodes} "
                    f"(source {dataset_path})"
                )
            pool_source = "parquet"
        elif allow_synthetic:
            episode_pool = synthetic_episodes(
                n_episodes=min(32, n_target),
                tokens_per_episode=64,
                vocab_size_native=native_size,
                seed=config.data_shuffle_seed,
            )
            pool_source = "synthetic"

    pool_list = [list(ep) for ep in episode_pool]
    ep_framed_lens = [len(ep) + 2 for ep in pool_list]
    stream_tokens = sum(
        ep_framed_lens[index]
        for index in iter_present_episode_order(
            len(pool_list), n_target, config.data_shuffle_seed
        )
    )

    def _presented() -> Iterator[list[int]]:
        for index in iter_present_episode_order(
            len(pool_list), n_target, config.data_shuffle_seed
        ):
            yield pool_list[index]

    raw_model = build_model(
        TransformerConfig(
            vocab_size=SHARED_VOCAB_SIZE,
            context_length=config.context_length_L,
        )
    )
    initialize_model(
        raw_model,
        model_init_seed=config.model_init_seed,
        codec_name=config.codec,
    )
    raw_model.to(device)
    raw_model.float()
    raw_model.train()

    use_compile = (
        device.type == "cuda"
        and (max_optimizer_steps is None or max_optimizer_steps >= 16)
        if compile_model is None
        else bool(compile_model)
    )
    compiled_model = (
        torch.compile(raw_model, dynamic=False) if use_compile else None
    )
    fused_adamw = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=ADAM_BETAS,
        fused=fused_adamw,
    )

    metrics_path = out / "metrics_train.jsonl"
    t0 = time.perf_counter()
    optimizer_steps = 0
    tokens_consumed = 0
    pin_memory = device.type == "cuda"
    n_windows, projected_steps = project_optimizer_steps(
        stream_tokens,
        context_length=config.context_length_L,
        batch_size=bs,
    )
    # Partial final batches are yielded (no drop_last) — matches ceil(windows/bs).
    first_batch_size = 0
    last_batch_size = 0
    pending_steps: list[tuple[int, int, int, int, int, int, torch.Tensor]] = []

    def _flush_pending_metrics(metrics_file: Any) -> None:
        if not pending_steps:
            return
        diag_rows = torch.stack([item[6] for item in pending_steps]).cpu().tolist()
        for (step_idx, step_tok, step_consumed, step_bs, n_w, n_c, _), diag in zip(
            pending_steps, diag_rows, strict=True
        ):
            loss_val, valid_val, within_val, cross_val = diag
            metrics_file.write(
                json.dumps(
                    {
                        "step": step_idx,
                        "loss": float(loss_val),
                        "tokens_in_step": step_tok,
                        "tokens_consumed": step_consumed,
                        "batch_size": step_bs,
                        "n_within": n_w,
                        "n_cross": n_c,
                        "loss_all_targets": float(valid_val),
                        "loss_within_segment": float(within_val) if n_w else 0.0,
                        "loss_cross_segment": float(cross_val) if n_c else 0.0,
                    }
                )
                + "\n"
            )
        metrics_file.flush()
        pending_steps.clear()

    batches = iter_packed_batches_streaming(
        _presented(),
        batch_size=bs,
        context_length=config.context_length_L,
        pin_memory=pin_memory,
    )
    if pin_memory:
        batches = prefetch_batches(batches, max_prefetch=2)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if resolved_precision == "bf16_mixed"
        else contextlib.nullcontext()
    )

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for input_ids_cpu, segment_ids_cpu in batches:
            batch_len = int(input_ids_cpu.shape[0])
            if optimizer_steps == 0:
                first_batch_size = batch_len
            last_batch_size = batch_len

            shift_seg_cpu = segment_ids_cpu[:, 1:]
            src_seg_cpu = segment_ids_cpu[:, :-1]
            valid_cpu = shift_seg_cpu != PAD_SEGMENT_ID
            within_cpu = valid_cpu & (shift_seg_cpu == src_seg_cpu)
            cross_cpu = valid_cpu & (shift_seg_cpu != src_seg_cpu)
            selected_cpu = within_cpu if restrict_loss_to_segment else valid_cpu
            n_tok = int(selected_cpu.sum().item())
            n_within = int(within_cpu.sum().item())
            n_cross = int(cross_cpu.sum().item())
            if n_tok == 0:
                # Still count as an optimizer step opportunity skipped — should not
                # happen for real packed content; raise so accounting cannot drift.
                raise RuntimeError(
                    "batch produced zero loss tokens; step accounting would diverge"
                )

            input_ids = input_ids_cpu.to(device, non_blocking=pin_memory)
            segment_ids = segment_ids_cpu.to(device, non_blocking=pin_memory)
            forward_fn = (
                compiled_model
                if (compiled_model is not None and batch_len == bs)
                else raw_model
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = forward_fn(input_ids, segment_ids)
            loss, diag_gpu = _next_token_loss_step(
                logits,
                input_ids,
                segment_ids,
                restrict_to_segment=restrict_loss_to_segment,
            )
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            tokens_consumed += n_tok

            pending_steps.append(
                (
                    optimizer_steps,
                    n_tok,
                    tokens_consumed,
                    batch_len,
                    n_within,
                    n_cross,
                    diag_gpu,
                )
            )
            if len(pending_steps) >= LOG_FLUSH_INTERVAL:
                _flush_pending_metrics(metrics_file)

            if (
                save_intermediate_every is not None
                and optimizer_steps % save_intermediate_every == 0
            ):
                _save_checkpoint(
                    ckpt_dir / f"step_{optimizer_steps}_non_scoring.pt",
                    raw_model,
                    scoring=False,
                    step=optimizer_steps,
                    config=config,
                )

            if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
                break

        _flush_pending_metrics(metrics_file)

    if max_optimizer_steps is None and optimizer_steps != projected_steps:
        raise RuntimeError(
            f"observed optimizer_steps {optimizer_steps} != projected "
            f"ceil(windows/batch_size)={projected_steps} "
            f"(n_windows={n_windows}, batch_size={bs}, "
            f"first_batch={first_batch_size}, last_batch={last_batch_size})"
        )

    packing = PackingAccount(
        stream_tokens=stream_tokens,
        n_windows=n_windows,
        batch_size=bs,
        last_batch_size=last_batch_size,
        projected_optimizer_steps=projected_steps,
        observed_optimizer_steps=optimizer_steps,
        # iter_packed_batches has no drop_last; partial finals are trained.
        partial_last_batch_included=True,
    )
    _save_checkpoint(
        ckpt_dir / "final.pt",
        raw_model,
        scoring=True,
        step=optimizer_steps,
        config=config,
    )

    param_count = raw_model.param_count()
    flops = estimate_flops_proxy_token_updates(
        tokens_consumed, context_length=config.context_length_L
    )
    flops["approx_6nd"] = 6 * param_count * tokens_consumed

    wall_clock_seconds = time.perf_counter() - t0
    reproducibility: dict[str, Any] = {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        **hardware_metadata(
            device,
            precision=resolved_precision,
            torch_compile=use_compile,
            fused_adamw=fused_adamw,
        ),
        "attention_mask_impl": "sdpa_bool_segment_ids",
        "config": config.to_dict(),
        "config_sha256": config_sha256(config),
        "budget_profile": config.budget_profile,
        "dataset_split_train": dataset_path.as_posix(),
        "dataset_checksum": file_sha256(dataset_path),
        # Provenance only. Not RunConfig fields, so config_sha256 is unchanged.
        # dataset_base_seed is the locked train-split world seed
        # (episode_seed = dataset_base_seed + episode_index).
        "dataset_base_seed": BASE_SEED_TRAIN,
        "schema_version": SCHEMA_VERSION,
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
        "wall_clock_seconds": wall_clock_seconds,
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
        wall_clock_seconds=wall_clock_seconds,
    )
