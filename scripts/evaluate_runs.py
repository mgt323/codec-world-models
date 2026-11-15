"""Score final checkpoints and write metrics_eval.json.

Refuses non-final checkpoints. ``--aggregate`` writes contract §5 summaries
and refuses plumbing profiles (tier0 / tier1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from data.hdf5_episodes import read_episodes_hdf5  # noqa: E402
from eval.checkpoint import load_scoring_checkpoint  # noqa: E402
from eval.metrics import METRIC_KEYS, empty_metrics  # noqa: E402
from eval.score import (  # noqa: E402
    COUNTERFACTUAL_BASE_SEED,
    EVAL_MAX_EPISODES,
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
) -> None:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    codec = str(config["codec"])
    vocab = load_codec_vocab(codec, vocab_dir)
    model, _payload = load_scoring_checkpoint(run_dir / "checkpoints" / "final.pt", device)
    train_eps = _load_prefix(raw_dir / "train.h5", PROBE_TRAIN_EPISODES)
    val_eps = _load_prefix(raw_dir / "val.h5", EVAL_MAX_EPISODES)
    ood_eps = _load_prefix(raw_dir / "ood_partial_obs.h5", EVAL_MAX_EPISODES)
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
    )
    intervention_path = eval_dir / "intervention.h5"
    if intervention_path.is_file():
        intervention_eps = _load_prefix(intervention_path, EVAL_SUITE_EPISODES)
    else:
        intervention_eps = make_intervention_episodes(
            EVAL_SUITE_EPISODES,
            base_seed=INTERVENTION_BASE_SEED,
            n_steps=N_STEPS_V0,
        )
    cf_obs = eval_dir / "counterfactual_observational.h5"
    cf_int = eval_dir / "counterfactual_intervened.h5"
    if cf_obs.is_file() and cf_int.is_file():
        left = _load_prefix(cf_obs, EVAL_SUITE_EPISODES)
        right = _load_prefix(cf_int, EVAL_SUITE_EPISODES)
        pairs = list(zip(left, right, strict=False))
    else:
        pairs = make_counterfactual_pairs(
            EVAL_SUITE_EPISODES,
            base_seed=COUNTERFACTUAL_BASE_SEED,
            n_steps=N_STEPS_V0,
        )
    intervention_metrics = empty_metrics()
    intervention_metrics["intervention_accuracy"] = intervention_accuracy(
        model, codec, intervention_eps, vocab, device
    )
    counterfactual_metrics = empty_metrics()
    counterfactual_metrics["counterfactual_divergence_mse"] = counterfactual_divergence_mse(
        model, codec, pairs, vocab, device
    )
    document = {
        "git_commit": commit if commit is not None else git_commit(),
        "git_dirty": dirty if dirty is not None else git_dirty(),
        **(hardware if hardware is not None else hardware_metadata(device)),
        "protocol": protocol_record(),
        "scoring_checkpoint": "final",
        "codec": codec,
        "splits": {
            "val": val_metrics,
            "ood_partial_obs": ood_metrics,
            "intervention": intervention_metrics,
            "counterfactual": counterfactual_metrics,
        },
    }
    for split_metrics in document["splits"].values():
        missing = [key for key in METRIC_KEYS if key not in split_metrics]
        if missing:
            raise RuntimeError(f"missing metric keys {missing}")
    (run_dir / "metrics_eval.json").write_text(
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
    parser.add_argument("--output-root", type=Path, default=_ROOT / "artifacts")
    args = parser.parse_args(argv)
    device = resolve_device(args.device, require_gpu=args.require_gpu)
    commit = git_commit()
    dirty = git_dirty()
    hardware = hardware_metadata(device)
    run_dirs = _iter_run_dirs(args.runs_root)
    if not run_dirs:
        raise SystemExit(f"no final checkpoints under {args.runs_root}")
    for run_dir in run_dirs:
        print(f"score {run_dir}")
        _score_run(
            run_dir,
            device=device,
            raw_dir=args.raw_dir,
            eval_dir=args.eval_dir,
            vocab_dir=args.vocab_dir,
            commit=commit,
            dirty=dirty,
            hardware=hardware,
        )
    manifest_path = args.runs_root / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    if manifest is not None:
        manifest["eval_status"] = "scored"
        manifest["updated_utc"] = _utc_now()
        write_manifest_atomic(manifest_path, manifest)
    if args.aggregate:
        profile_name = args.budget_profile or (
            manifest.get("budget_profile") if manifest else None
        )
        if not profile_name:
            raise SystemExit("--aggregate needs --budget-profile or a manifest")
        get_budget_profile(profile_name)
        written = aggregate_scientific_metrics(
            args.runs_root,
            profile_name=profile_name,
            output_root=args.output_root,
            git_commit=commit,
            git_dirty=dirty,
            allow_pilot=args.allow_pilot,
        )
        for path in written:
            print(f"aggregate {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
