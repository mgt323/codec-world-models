"""Post-run audit of one tier's run artifacts.

Re-checks, from the artifacts alone, the invariants the runner asserts in
process: contract §4.3 file set, packing/step accounting, param-count parity,
seed pairing, and dataset provenance. Exits non-zero if anything fails.

Usage::

    python scripts/verify_tier_artifacts.py tier0
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import Counter
from collections.abc import Mapping

import torch

from eval.metrics import METRIC_KEYS
from eval.score import DECODER_MODES, EVAL_SCHEMA_VERSION

EXPECTED_PARAMS = 11_056_896
_EVAL_SPLITS = ("val", "ood_partial_obs", "intervention", "counterfactual")


def _parse_args(argv: list[str] | None = None) -> tuple[str, pathlib.Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", nargs="?", default="tier0")
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path("artifacts"),
        help="Root containing runs/{tier}/manifest.json (default: artifacts).",
    )
    args = parser.parse_args(argv)
    return str(args.tier), args.output_root / "runs" / str(args.tier)


def eval_document_problems(key: str, eval_doc: Mapping[str, object]) -> list[str]:
    """Schema checks for one ``metrics_eval.json`` document."""
    problems: list[str] = []
    splits = eval_doc.get("splits", {})
    if not isinstance(splits, dict):
        return [f"{key}: metrics_eval.json splits must be an object"]
    for split_name in _EVAL_SPLITS:
        metrics = splits.get(split_name)
        if not isinstance(metrics, dict):
            problems.append(f"{key}: metrics_eval.json missing split {split_name}")
            continue
        for metric_name in METRIC_KEYS:
            if metric_name not in metrics:
                problems.append(
                    f"{key}: metrics_eval.json {split_name} missing {metric_name}"
                )
    if eval_doc.get("scoring_checkpoint") != "final":
        problems.append(f"{key}: metrics_eval.json scoring_checkpoint is not final")
    if not eval_doc.get("git_commit"):
        problems.append(f"{key}: metrics_eval.json missing git_commit")
    if not isinstance(eval_doc.get("git_dirty"), bool):
        problems.append(f"{key}: metrics_eval.json git_dirty must be bool")
    if eval_doc.get("eval_schema_version") != EVAL_SCHEMA_VERSION:
        problems.append(
            f"{key}: metrics_eval.json eval_schema_version "
            f"{eval_doc.get('eval_schema_version')!r} != {EVAL_SCHEMA_VERSION!r}"
        )
    if eval_doc.get("decoder_mode") not in DECODER_MODES:
        problems.append(
            f"{key}: metrics_eval.json decoder_mode "
            f"{eval_doc.get('decoder_mode')!r} not in {DECODER_MODES}"
        )
    return problems


def manifest_eval_contract_problems(manifest: Mapping[str, object]) -> list[str]:
    """Require eval_v2 fields once a tier is scored or any metrics file exists."""
    problems: list[str] = []
    if manifest.get("eval_schema_version") != EVAL_SCHEMA_VERSION:
        problems.append(
            "manifest eval_schema_version "
            f"{manifest.get('eval_schema_version')!r} != {EVAL_SCHEMA_VERSION!r}"
        )
    if manifest.get("decoder_mode") not in DECODER_MODES:
        problems.append(
            f"manifest decoder_mode {manifest.get('decoder_mode')!r} not in {DECODER_MODES}"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    tier, root = _parse_args(argv)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    print(f"profile         {manifest['budget_profile']}")
    print(f"interpretation  {manifest['interpretation']}")
    print(f"eval_status     {manifest['eval_status']}")
    print(f"harness_cap     {manifest['harness_step_cap']}")
    print(f"status          {dict(Counter(c['status'] for c in manifest['cells']))}")
    print()

    problems: list[str] = []
    fingerprints: dict[str, set[str]] = {}
    shuffle_by_rep: dict[int, set[int]] = {}
    precisions: set[str] = set()
    saw_eval = False

    header = (
        f"{'cell':<32}{'steps':>7}{'proj':>6}{'windows':>9}{'stream':>11}"
        f"{'T_eff':>12}{'params':>11}"
    )
    print(header)
    print("-" * len(header))

    for cell in sorted(
        manifest["cells"],
        key=lambda x: (x["match_regime"], x["codec"], x["replication_id"]),
    ):
        run_dir = pathlib.Path(cell["run_dir"])
        key = f"{cell['match_regime']}/{cell['codec']}/rep{cell['replication_id']}"

        for name in (
            "config.json",
            "metrics_train.jsonl",
            "reproducibility.json",
            "checkpoints/final.pt",
        ):
            if not (run_dir / name).exists():
                problems.append(f"{key}: missing {name}")
        eval_path = run_dir / "metrics_eval.json"
        if manifest.get("eval_status") == "scored" and not eval_path.is_file():
            problems.append(f"{key}: manifest eval_status=scored but missing metrics_eval.json")
        if eval_path.is_file():
            saw_eval = True
            eval_doc = json.loads(eval_path.read_text(encoding="utf-8"))
            problems.extend(eval_document_problems(key, eval_doc))
            manifest_schema = manifest.get("eval_schema_version")
            if (
                manifest_schema is not None
                and eval_doc.get("eval_schema_version") != manifest_schema
            ):
                problems.append(f"{key}: eval_schema_version disagrees with manifest")
            manifest_decoder = manifest.get("decoder_mode")
            if manifest_decoder is not None and eval_doc.get("decoder_mode") != manifest_decoder:
                problems.append(f"{key}: decoder_mode disagrees with manifest")

        ckpt_path = run_dir / "checkpoints" / "final.pt"
        if ckpt_path.is_file():
            try:
                ckpt_payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt_payload = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt_payload.get("model_state_dict", {})
            if any(str(k).startswith("_orig_mod.") for k in state_dict):
                problems.append(f"{key}: final.pt contains _orig_mod. prefix keys")

        repro = json.loads((run_dir / "reproducibility.json").read_text(encoding="utf-8"))
        if not repro.get("git_commit"):
            problems.append(f"{key}: reproducibility.json missing git_commit")
        if not isinstance(repro.get("git_dirty"), bool):
            problems.append(f"{key}: reproducibility.json git_dirty must be bool")
        cell_precision = repro.get("precision")
        if not isinstance(cell_precision, str) or not cell_precision:
            problems.append(f"{key}: reproducibility.json missing precision")
        else:
            precisions.add(cell_precision)
        config = repro["config"]

        # Basis is the packed stream, not tokens_consumed: the latter counts
        # next-token targets, which is stream minus one target per window.
        account = repro["packing_account"]
        stream = account["stream_tokens"]
        projected_windows = math.ceil(stream / 1024)
        projected = math.ceil(projected_windows / 8)
        if projected_windows != account["n_windows"]:
            problems.append(
                f"{key}: windows {account['n_windows']} != ceil(stream/L) "
                f"{projected_windows}"
            )
        if projected != cell["optimizer_steps"]:
            problems.append(
                f"{key}: steps {cell['optimizer_steps']} != ceil projection {projected}"
            )
        if repro["tokens_consumed"] != stream - account["n_windows"]:
            problems.append(
                f"{key}: tokens_consumed {repro['tokens_consumed']} != "
                f"stream - n_windows {stream - account['n_windows']}"
            )
        if not account["partial_last_batch_included"]:
            problems.append(f"{key}: drop_last policy would under-consume T_eff")
        if repro["model_param_count"] != EXPECTED_PARAMS:
            problems.append(f"{key}: param count {repro['model_param_count']}")
        if repro["episode_pool_size"] != manifest["pool_episodes"]:
            problems.append(f"{key}: pool size {repro['episode_pool_size']}")
        if repro["dataset_checksum"] is None:
            problems.append(f"{key}: dataset_checksum is null")
        if repro["budget_profile"] != tier:
            problems.append(f"{key}: budget_profile {repro['budget_profile']}")
        if repro["scoring_checkpoint"] != "final":
            problems.append(f"{key}: scoring_checkpoint {repro['scoring_checkpoint']}")
        if repro["harness_step_cap"] is not None:
            problems.append(f"{key}: step-capped run ({repro['harness_step_cap']})")

        n_lines = len(
            (run_dir / "metrics_train.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )
        if n_lines != cell["optimizer_steps"]:
            problems.append(
                f"{key}: {n_lines} metric lines != {cell['optimizer_steps']} steps"
            )

        fingerprints.setdefault(cell["codec"], set()).add(
            repro["episode_pool_fingerprint"]
        )
        shuffle_by_rep.setdefault(config["replication_id"], set()).add(
            config["data_shuffle_seed"]
        )

        print(
            f"{key:<32}{cell['optimizer_steps']:>7}{projected:>6}"
            f"{account['n_windows']:>9,}{stream:>11,}{cell['T_eff']:>12,.0f}"
            f"{repro['model_param_count']:>11,}"
        )

    if manifest.get("eval_status") == "scored" or saw_eval:
        problems.extend(manifest_eval_contract_problems(manifest))

    print()
    for codec, fps in sorted(fingerprints.items()):
        if len(fps) != 1:
            problems.append(f"{codec}: {len(fps)} distinct pool fingerprints")
    print(f"one pool fingerprint per codec:   {all(len(v) == 1 for v in fingerprints.values())}")
    distinct = {next(iter(v)) for v in fingerprints.values()}
    print(f"distinct fingerprints by codec:   {len(distinct)} of {len(fingerprints)}")

    for rep, seeds in sorted(shuffle_by_rep.items()):
        if len(seeds) != 1:
            problems.append(f"rep{rep}: data_shuffle_seed not shared across codecs {seeds}")
    print(f"shuffle seed shared within rep:   {all(len(v) == 1 for v in shuffle_by_rep.values())}")
    print(f"shuffle seeds across reps:        "
          f"{sorted(next(iter(v)) for v in shuffle_by_rep.values())}")

    manifest_precision = manifest.get("precision")
    if len(precisions) != 1:
        problems.append(f"mixed precisions across cells: {sorted(precisions)}")
    elif manifest_precision is not None and next(iter(precisions)) != manifest_precision:
        problems.append(
            f"cell precision {next(iter(precisions))!r} != manifest precision {manifest_precision!r}"
        )
    print(f"uniform cell precision:           {sorted(precisions)}")

    print()
    print(f"PROBLEMS: {len(problems)}")
    for problem in problems:
        print(" ", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
