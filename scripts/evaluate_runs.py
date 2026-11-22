"""Score final checkpoints and write metrics_eval.json.

Refuses non-final checkpoints. ``--aggregate`` writes contract §5 summaries
and refuses plumbing profiles (tier0 / tier1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from data.hdf5_episodes import read_episodes_hdf5  # noqa: E402
from eval.checkpoint import load_scoring_checkpoint  # noqa: E402
from eval.diagnostics import DIAGNOSTICS_VERSION, compute_eval_diagnostics  # noqa: E402
from eval.metrics import METRIC_KEYS, empty_metrics  # noqa: E402
from eval.score import (  # noqa: E402
    COUNTERFACTUAL_BASE_SEED,
    DECODER_MODES,
    EVAL_MAX_EPISODES,
    EVAL_SCHEMA_VERSION,
    EVAL_SUITE_EPISODES,
    INTERVENTION_BASE_SEED,
    PROBE_TRAIN_EPISODES,
    counterfactual_divergence_mse,
    intervention_accuracy,
    make_counterfactual_pairs,
    make_intervention_episodes,
    protocol_record,
    score_passive_split,
)
from train.config import N_STEPS_V0, get_budget_profile  # noqa: E402
from train.data import load_codec_vocab  # noqa: E402
from train.device import hardware_metadata, resolve_device  # noqa: E402
from train.loop import git_commit, git_dirty  # noqa: E402
from train.matrix import MANIFEST_NAME, _utc_now, aggregate_scientific_metrics, load_manifest, write_manifest_atomic  # noqa: E402

_TRANSFER_PARTNER = {"A": "B", "B": "A"}


def _load_prefix(path: Path, limit: int) -> list:
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_episodes_hdf5(path, max_episodes=limit)


def _score_run(
    run_dir: Path,
    *,
    device: torch.device,
    raw_dir: Path,
    eval_dir: Path,
    vocab_dir: Path,
    commit: str | None = None,
    dirty: bool | None = None,
    hardware: dict[str, Any] | None = None,
    decoder_mode: str = "skeleton",
    metrics_out_dir: Path | None = None,
    episode_cache: dict[str, Any] | None = None,
    diagnostics: bool = True,
) -> None:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    codec = str(config["codec"])
    vocab = load_codec_vocab(codec, vocab_dir)
    model, _payload = load_scoring_checkpoint(run_dir / "checkpoints" / "final.pt", device)
    cache = episode_cache if episode_cache is not None else {}
    if "train" not in cache:
        cache["train"] = _load_prefix(raw_dir / "train.h5", PROBE_TRAIN_EPISODES)
    if "val" not in cache:
        cache["val"] = _load_prefix(raw_dir / "val.h5", EVAL_MAX_EPISODES)
    if "ood" not in cache:
        cache["ood"] = _load_prefix(raw_dir / "ood_partial_obs.h5", EVAL_MAX_EPISODES)
    train_eps = cache["train"]
    val_eps = cache["val"]
    ood_eps = cache["ood"]
    partner = _TRANSFER_PARTNER.get(codec)
    transfer_eps = val_eps if partner is not None else None
    val_metrics = score_passive_split(
        model,
        codec,
        val_eps,
        vocab,
        device,
        train_episodes=train_eps,
        partial_obs=False,
        transfer_model_codec=partner,
        transfer_episodes=transfer_eps,
        decoder_mode=decoder_mode,
    )
    ood_metrics = score_passive_split(
        model,
        codec,
        ood_eps,
        vocab,
        device,
        train_episodes=train_eps,
        partial_obs=True,
        transfer_model_codec=partner,
        transfer_episodes=ood_eps if partner is not None else None,
        decoder_mode=decoder_mode,
    )
    if "intervention" not in cache:
        intervention_path = eval_dir / "intervention.h5"
        if intervention_path.is_file():
            cache["intervention"] = _load_prefix(intervention_path, EVAL_SUITE_EPISODES)
        else:
            cache["intervention"] = make_intervention_episodes(
                EVAL_SUITE_EPISODES,
                base_seed=INTERVENTION_BASE_SEED,
                n_steps=N_STEPS_V0,
            )
    intervention_eps = cache["intervention"]
    if "counterfactual" not in cache:
        cf_obs = eval_dir / "counterfactual_observational.h5"
        cf_int = eval_dir / "counterfactual_intervened.h5"
        if cf_obs.is_file() and cf_int.is_file():
            left = _load_prefix(cf_obs, EVAL_SUITE_EPISODES)
            right = _load_prefix(cf_int, EVAL_SUITE_EPISODES)
            cache["counterfactual"] = list(zip(left, right, strict=False))
        else:
            cache["counterfactual"] = make_counterfactual_pairs(
                EVAL_SUITE_EPISODES,
                base_seed=COUNTERFACTUAL_BASE_SEED,
                n_steps=N_STEPS_V0,
            )
    pairs = cache["counterfactual"]
    intervention_metrics = empty_metrics()
    int_scores = intervention_accuracy(
        model, codec, intervention_eps, vocab, device, decoder_mode=decoder_mode
    )
    intervention_metrics["intervention_accuracy"] = int_scores["accuracy"]
    intervention_metrics["parse_fail_rate"] = int_scores["parse_fail_rate"]
    counterfactual_metrics = empty_metrics()
    counterfactual_metrics["counterfactual_divergence_mse"] = counterfactual_divergence_mse(
        model, codec, pairs, vocab, device, decoder_mode=decoder_mode
    )
    # Additive Tier B2 diagnostics: a separate top-level block. The locked
    # ``splits`` metrics above are not modified and §5 aggregation ignores it.
    diagnostics_block = (
        compute_eval_diagnostics(
            model,
            codec,
            vocab,
            device,
            val_episodes=val_eps,
            ood_episodes=ood_eps,
            intervention_episodes=intervention_eps,
            counterfactual_pairs=pairs,
            decoder_mode=decoder_mode,
        )
        if diagnostics
        else None
    )
    document = {
        "eval_schema_version": EVAL_SCHEMA_VERSION,
        "decoder_mode": decoder_mode,
        "git_commit": commit if commit is not None else git_commit(),
        "git_dirty": dirty if dirty is not None else git_dirty(),
        **(
            hardware
            if hardware is not None
            else hardware_metadata(device, precision="fp32")
        ),
        "protocol": protocol_record(decoder_mode=decoder_mode),
        "scoring_checkpoint": "final",
        "codec": codec,
        "splits": {
            "val": val_metrics,
            "ood_partial_obs": ood_metrics,
            "intervention": intervention_metrics,
            "counterfactual": counterfactual_metrics,
        },
        **({"diagnostics": diagnostics_block} if diagnostics_block is not None else {}),
    }
    for split_metrics in document["splits"].values():
        missing = [key for key in METRIC_KEYS if key not in split_metrics]
        if missing:
            raise RuntimeError(f"missing metric keys {missing}")
    target_dir = run_dir if metrics_out_dir is None else metrics_out_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "metrics_eval.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _iter_run_dirs(runs_root: Path) -> list[Path]:
    return sorted(
        path.parent.parent
        for path in runs_root.glob("*/*/*/checkpoints/final.pt")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=_ROOT / "artifacts" / "eval_splits",
    )
    parser.add_argument("--vocab-dir", type=Path, default=_ROOT / "artifacts" / "vocabs")
    parser.add_argument("--device", choices=("auto", "rocm", "cuda", "cpu"), default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--budget-profile", default=None)
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Write §5 summaries. Refused unless the profile is scientific.",
    )
    parser.add_argument(
        "--allow-pilot",
        action="store_true",
        help="Allow §5 aggregation on plumbing profiles (tier0/tier1).",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help=f"Do not write the additive {DIAGNOSTICS_VERSION} block to metrics_eval.json.",
    )
    parser.add_argument("--output-root", type=Path, default=_ROOT / "artifacts")
    parser.add_argument(
        "--decoder-mode",
        choices=DECODER_MODES,
        default="skeleton",
        help="Token reconstruction mode (default: skeleton).",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=None,
        help="Optional separate root for writing metrics_eval.json and manifest.json without overwriting --runs-root.",
    )
    parser.add_argument(
        "--aggregates-dir",
        type=Path,
        default=None,
        help="Optional separate directory for §5 aggregates without overwriting output-root/runs/aggregates.",
    )
    args = parser.parse_args(argv)
    device = resolve_device(args.device, require_gpu=args.require_gpu)
    commit = git_commit()
    dirty = git_dirty()
    hardware = hardware_metadata(device, precision="fp32")
    run_dirs = _iter_run_dirs(args.runs_root)
    if not run_dirs:
        raise SystemExit(f"no final checkpoints under {args.runs_root}")
    episode_cache: dict[str, Any] = {}
    metrics_root = args.metrics_root if args.metrics_root is not None else args.runs_root
    for run_dir in run_dirs:
        print(f"score {run_dir}")
        rel = run_dir.relative_to(args.runs_root)
        metrics_out_dir = None if args.metrics_root is None else (args.metrics_root / rel)
        _score_run(
            run_dir,
            device=device,
            raw_dir=args.raw_dir,
            eval_dir=args.eval_dir,
            vocab_dir=args.vocab_dir,
            commit=commit,
            dirty=dirty,
            hardware=hardware,
            decoder_mode=args.decoder_mode,
            metrics_out_dir=metrics_out_dir,
            episode_cache=episode_cache,
            diagnostics=not args.skip_diagnostics,
        )
    src_manifest_path = args.runs_root / MANIFEST_NAME
    manifest = load_manifest(src_manifest_path)
    if manifest is not None:
        manifest["eval_status"] = "scored"
        manifest["eval_schema_version"] = EVAL_SCHEMA_VERSION
        manifest["decoder_mode"] = args.decoder_mode
        manifest["eval_splits_dir"] = str(args.eval_dir)
        manifest["eval_git_commit"] = commit
        manifest["eval_git_dirty"] = dirty
        manifest["updated_utc"] = _utc_now()
        dst_manifest_path = metrics_root / MANIFEST_NAME
        dst_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_manifest_atomic(dst_manifest_path, manifest)
    if args.aggregate:
        profile_name = args.budget_profile or (
            manifest.get("budget_profile") if manifest else None
        )
        if not profile_name:
            raise SystemExit("--aggregate needs --budget-profile or a manifest")
        get_budget_profile(profile_name)
        # summary.json["git_commit"] / ["git_dirty"] always inherit the training
        # manifest so eval/report.py can cross-check them; the commit that ran
        # this evaluation is recorded separately as eval_git_commit / _dirty.
        if manifest is not None and manifest.get("git_commit"):
            agg_commit = str(manifest["git_commit"])
            agg_dirty = bool(manifest.get("git_dirty", False))
        else:
            agg_commit = commit
            agg_dirty = dirty
        written = aggregate_scientific_metrics(
            metrics_root,
            profile_name=profile_name,
            output_root=args.output_root,
            git_commit=agg_commit,
            git_dirty=agg_dirty,
            allow_pilot=args.allow_pilot,
            aggregates_dir=args.aggregates_dir,
            eval_schema_version=EVAL_SCHEMA_VERSION,
            decoder_mode=args.decoder_mode,
            eval_git_commit=commit,
            eval_git_dirty=dirty,
        )
        for path in written:
            print(f"aggregate {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
