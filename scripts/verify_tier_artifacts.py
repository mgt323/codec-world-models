"""Post-run audit of one tier's run artifacts.

Re-checks, from the artifacts alone, the invariants the runner asserts in
process: contract §4.3 file set, packing/step accounting, param-count parity,
seed pairing, and dataset provenance. Exits non-zero if anything fails.

Usage::

    python scripts/verify_tier_artifacts.py tier0
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import Counter

TIER = sys.argv[1] if len(sys.argv) > 1 else "tier0"
EXPECTED_PARAMS = 11_056_896

root = pathlib.Path("artifacts/runs") / TIER
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
    if (run_dir / "metrics_eval.json").exists():
        problems.append(f"{key}: metrics_eval.json must stay absent (no producer)")

    repro = json.loads((run_dir / "reproducibility.json").read_text(encoding="utf-8"))
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
    if repro["budget_profile"] != TIER:
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

print()
print(f"PROBLEMS: {len(problems)}")
for p in problems:
    print(" ", p)
raise SystemExit(1 if problems else 0)
