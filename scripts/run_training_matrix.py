"""Run the stage-1 training matrix for one budget profile (tier).

Implements docs/TRAINING_RUNNER_CONTRACT.md §4. Budgets are derived from the §0
formula at the tier's pool size — no per-tier literals. See ``scripts/evaluate_runs.py``
for post-training scoring and §5 aggregation.

Example::

    python scripts/run_training_matrix.py --tier 0 --replications 3
    python scripts/run_training_matrix.py --tier 0 --dry-run
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.config import (  # noqa: E402
    BATCH_SIZE,
    BUDGET_PROFILES,
    BudgetProfile,
    get_budget_profile,
    validate_run_config,
)
from train.data import episodes_from_parquet, load_codec_vocab  # noqa: E402
from train.device import (  # noqa: E402
    hardware_metadata,
    resolve_device,
    resolve_precision,
)
from train.loop import (  # noqa: E402
    git_commit,
    git_dirty,
    project_optimizer_steps,
    run_training,
)
from train.matrix import (  # noqa: E402
    DEFAULT_CODECS,
    DEFAULT_REGIMES,
    MANIFEST_NAME,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    MatrixCell,
    assert_seed_pairing,
    build_manifest,
    expand_matrix,
    iter_pool_requirements,
    load_manifest,
    manifest_cell_index,
    run_is_complete,
    update_cell,
    write_manifest_atomic,
)

TIER_PROFILES = {0: "tier0", 1: "tier1"}
# Measured compiled bf16_mixed AdamW step on RX 9070 XT (gfx1201),
# torch 2.14.1+rocm7.2, batch 8, L=1024, boolean SDPA mask. Engineering estimate.
SECONDS_PER_STEP_ESTIMATE = 0.025


def _tier_segment(profile: BudgetProfile) -> str:
    return profile.name


def _projected_steps(cell: MatrixCell, batch_size: int) -> int:
    """Windows/steps the packer will produce for this cell's full budget."""
    stream = cell.config.T_eff + 2 * cell.config.n_episodes
    _windows, steps = project_optimizer_steps(
        int(stream),
        context_length=cell.config.context_length_L,
        batch_size=batch_size,
    )
    return steps


def _preflight(
    cells: list[MatrixCell],
    profile: BudgetProfile,
    *,
    vocab_dir: Path,
    encoded_dir: Path | None = None,
    batch_size: int,
) -> dict[str, list[list[int]]]:
    """Contract §4.4.1-§4.4.3 gates plus the pool-size guard.

    Returns the in-process episode-pool cache (one entry per codec). The cache
    is local to this invocation: nothing is persisted, so every ``--resume``
    re-invocation rebuilds it from the parquet.
    """
    for cell in cells:
        validate_run_config(cell.config)
    assert_seed_pairing(cells)

    enc_root = _ROOT / "artifacts" / "encoded" if encoded_dir is None else Path(encoded_dir)
    pools: dict[str, list[list[int]]] = {}
    for codec in iter_pool_requirements(cells):
        parquet = enc_root / f"train_{codec}.parquet"
        if not parquet.is_file():
            raise SystemExit(f"missing encoded train split: {parquet}")
        vocab = load_codec_vocab(codec, vocab_dir)
        pool = episodes_from_parquet(
            parquet, vocab, max_episodes=profile.pool_episodes
        )
        if len(pool) != profile.pool_episodes:
            raise SystemExit(
                f"{codec}: loaded pool has {len(pool)} episodes, profile "
                f"{profile.name} requires exactly {profile.pool_episodes} "
                f"(source {parquet})"
            )
        print(
            f"pool {codec:<10} {len(pool)} episodes == "
            f"{profile.name}.pool_episodes"
        )
        pools[codec] = pool

    total_steps = sum(_projected_steps(c, batch_size) for c in cells)
    print()
    print(f"profile          {profile.name}")
    print(f"pool_episodes    {profile.pool_episodes}")
    print(f"T_target         {float(profile.t_target):,.2f}")
    print(f"interpretation   "
          f"{'scientific' if profile.scientific else 'plumbing_only'}")
    print(f"cells            {len(cells)}")
    print(f"projected steps  {total_steps:,}")
    print(
        f"est wall-clock   ~{total_steps * SECONDS_PER_STEP_ESTIMATE / 3600:.1f} h "
        f"at {SECONDS_PER_STEP_ESTIMATE:.2f}s/step"
    )
    print()
    return pools


def _print_budget_table(cells: list[MatrixCell]) -> None:
    seen: set[tuple[str, str]] = set()
    print(f"{'codec':<11}{'regime':<17}{'n_episodes':>11}{'T_eff':>18}")
    for cell in cells:
        c = cell.config
        key = (c.codec, c.match_regime)
        if key in seen:
            continue
        seen.add(key)
        print(f"{c.codec:<11}{c.match_regime:<17}{c.n_episodes:>11,}{c.T_eff:>18,.2f}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Expand and run the training matrix for one budget profile. "
            "Pilot tiers are plumbing validation only."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tier", type=int, choices=sorted(TIER_PROFILES))
    group.add_argument("--budget-profile", choices=sorted(BUDGET_PROFILES))
    parser.add_argument("--codecs", nargs="+", default=list(DEFAULT_CODECS))
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES))
    parser.add_argument("--replications", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=_ROOT / "artifacts")
    parser.add_argument("--vocab-dir", type=Path, default=_ROOT / "artifacts" / "vocabs")
    parser.add_argument(
        "--encoded-dir",
        type=Path,
        default=_ROOT / "artifacts" / "encoded",
        help="Directory containing train_{codec}.parquet splits.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--device",
        choices=("auto", "rocm", "cuda", "cpu"),
        default="auto",
        help="auto prefers ROCm/HIP (torch.cuda). rocm and cuda are the same device.",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16_mixed", "fp32"),
        default="auto",
        help="auto uses bf16_mixed on ROCm GPU and fp32 on CPU.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile during training (diagnostic escape hatch).",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Abort when the resolved device is CPU.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every cell even if a valid completed run exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preflight and print the plan; launch nothing",
    )
    parser.add_argument(
        "--smoke-max-steps",
        type=int,
        default=None,
        help=(
            "Harness cap on optimizer steps per run. Taints the manifest: "
            "capped runs never consume the full budget."
        ),
    )
    args = parser.parse_args(argv)

    profile_name = (
        args.budget_profile
        if args.budget_profile
        else TIER_PROFILES[args.tier]
    )
    profile = get_budget_profile(profile_name)
    segment = _tier_segment(profile)

    min_replications = (
        3 if profile.scientific and args.smoke_max_steps is None else None
    )
    try:
        device = resolve_device(args.device, require_gpu=args.require_gpu)
        resolved_precision = resolve_precision(device, args.precision)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"device           {device} ({args.device})")
    print(f"precision        {resolved_precision} ({args.precision})")

    cells = expand_matrix(
        profile=profile,
        codecs=args.codecs,
        match_regimes=args.regimes,
        n_replications=args.replications,
        output_root=args.output_root,
        runs_subdir=segment,
        min_replications=min_replications,
    )

    _print_budget_table(cells)
    pools = _preflight(
        cells,
        profile,
        vocab_dir=args.vocab_dir,
        encoded_dir=args.encoded_dir,
        batch_size=args.batch_size,
    )

    use_compile = (
        False
        if args.no_compile
        else (
            device.type == "cuda"
            and (args.smoke_max_steps is None or args.smoke_max_steps >= 16)
        )
    )
    hw_meta = hardware_metadata(
        device,
        precision=resolved_precision,
        torch_compile=use_compile,
        fused_adamw=(device.type == "cuda"),
    )
    manifest_path = Path(args.output_root) / "runs" / segment / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    expected_keys = {c.key for c in cells}
    if manifest is None or set(manifest_cell_index(manifest)) != expected_keys:
        manifest = build_manifest(
            profile=profile,
            cells=cells,
            codecs=args.codecs,
            match_regimes=args.regimes,
            n_replications=args.replications,
            git_commit=git_commit(),
            smoke_max_steps=args.smoke_max_steps,
            git_dirty=git_dirty(),
            hardware=hw_meta,
        )
    else:
        prior_precision = manifest.get("precision")
        if (
            prior_precision is not None
            and prior_precision != resolved_precision
            and not args.force
        ):
            raise SystemExit(
                f"refusing to mix precision {resolved_precision!r} with existing "
                f"manifest precision {prior_precision!r} at {manifest_path}; "
                "pass --force or use a separate --output-root"
            )
        manifest.update(hw_meta)
    index = manifest_cell_index(manifest)

    if args.dry_run:
        print("dry run: no cells launched")
        return 0

    write_manifest_atomic(manifest_path, manifest)

    n_done = n_skipped = n_failed = 0
    for position, cell in enumerate(cells, start=1):
        complete, reason = run_is_complete(cell)
        if complete and not args.force:
            n_skipped += 1
            update_cell(manifest, index, cell, status=STATUS_COMPLETED, error=None)
            write_manifest_atomic(manifest_path, manifest)
            print(f"[{position}/{len(cells)}] skip     {cell.key} ({reason})")
            continue

        update_cell(manifest, index, cell, status=STATUS_RUNNING, error=None)
        write_manifest_atomic(manifest_path, manifest)
        print(f"[{position}/{len(cells)}] run      {cell.key} ({reason})")

        try:
            result = run_training(
                cell.config,
                output_root=args.output_root,
                device=device,
                episode_pool=pools[cell.config.codec],
                batch_size=args.batch_size,
                max_optimizer_steps=args.smoke_max_steps,
                runs_subdir=segment,
                vocab_dir=args.vocab_dir,
                encoded_dir=args.encoded_dir,
                precision=resolved_precision,
                compile_model=(False if args.no_compile else None),
            )
        except KeyboardInterrupt:
            update_cell(
                manifest,
                index,
                cell,
                status=STATUS_INTERRUPTED,
                error="KeyboardInterrupt",
            )
            write_manifest_atomic(manifest_path, manifest)
            print(f"interrupted during {cell.key}; manifest flushed")
            raise
        except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the matrix
            n_failed += 1
            update_cell(
                manifest,
                index,
                cell,
                status=STATUS_FAILED,
                error=f"{exc.__class__.__name__}: {exc}",
            )
            write_manifest_atomic(manifest_path, manifest)
            print(f"[{position}/{len(cells)}] FAILED   {cell.key}: {exc}")
            traceback.print_exc()
            continue

        n_done += 1
        update_cell(
            manifest,
            index,
            cell,
            status=STATUS_COMPLETED,
            optimizer_steps=result.optimizer_steps,
            tokens_consumed=result.tokens_consumed,
            wall_clock_seconds=result.wall_clock_seconds,
            error=None,
        )
        write_manifest_atomic(manifest_path, manifest)
        print(
            f"[{position}/{len(cells)}] done     {cell.key} "
            f"steps={result.optimizer_steps} tokens={result.tokens_consumed:,}"
        )

    print()
    print(f"completed {n_done}, skipped {n_skipped}, failed {n_failed}")
    print(f"manifest: {manifest_path}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
