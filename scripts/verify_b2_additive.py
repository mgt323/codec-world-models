"""Check that a Tier B2 rerun only added keys and left the locked ones unchanged.

``probes`` mode compares ``probes.json`` cells: the locked per-codec trio
(``hidden_c_r2``, ``regime_accuracy``, ``next_b_r2``) and the CKA matrices.
``eval`` mode compares ``metrics_eval.json`` cells: the whole locked ``splits``
block. In both modes the candidate must also contain the additive keys, and its
provenance (dirty flags) is printed.

Exit code 0 means every locked value agrees within ``--atol`` and all additive
keys are present; exit code 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LOCKED_PROBE_KEYS: tuple[str, ...] = ("hidden_c_r2", "regime_accuracy", "next_b_r2")
ADDITIVE_PROBE_CODEC_KEYS: tuple[str, ...] = ("hidden_c_r2_common_cause",)
ADDITIVE_PROBE_TOP_KEYS: tuple[str, ...] = ("fixture_baselines",)
ADDITIVE_FIXTURE_KEYS: tuple[str, ...] = (
    "persistence_next_b_r2",
    "running_mean_next_b_r2",
    "oracle_next_b_r2",
)
ADDITIVE_EVAL_SPLIT_KEYS: dict[str, tuple[str, ...]] = {
    "val": ("next_state_mse_median", "next_state_mse_winsorized"),
    "ood_partial_obs": ("next_state_mse_median", "next_state_mse_winsorized"),
    "intervention": (
        "intervention_accuracy_both_observed",
        "intervention_accuracy_parsed",
        "zero_baseline_acc",
        "oracle_acc",
    ),
    "counterfactual": ("cf_step_mse_at_intervention",),
}


def _differences(left: Any, right: Any, path: str, atol: float, out: list[tuple[str, float]]) -> None:
    """Collect ``(path, abs_diff)`` for every locked value that disagrees."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                out.append((f"{path}/{key}", float("inf")))
            else:
                _differences(left[key], right[key], f"{path}/{key}", atol, out)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append((path, float("inf")))
            return
        for index, (lv, rv) in enumerate(zip(left, right, strict=True)):
            _differences(lv, rv, f"{path}[{index}]", atol, out)
        return
    if left is None or right is None or isinstance(left, (str, bool)) or isinstance(right, (str, bool)):
        if left != right:
            out.append((path, float("inf")))
        return
    diff = abs(float(left) - float(right))
    if diff > atol:
        out.append((path, diff))


def compare_probe_cell(
    baseline: dict[str, Any], candidate: dict[str, Any], *, atol: float
) -> tuple[list[tuple[str, float]], list[str]]:
    """Return ``(locked_differences, missing_additive_keys)`` for one ``probes.json``."""
    diffs: list[tuple[str, float]] = []
    for codec, base_probes in baseline["probes"].items():
        cand_probes = candidate["probes"].get(codec)
        if cand_probes is None:
            diffs.append((f"probes/{codec}", float("inf")))
            continue
        locked_base = {k: base_probes[k] for k in LOCKED_PROBE_KEYS}
        locked_cand = {k: cand_probes.get(k) for k in LOCKED_PROBE_KEYS}
        _differences(locked_base, locked_cand, f"probes/{codec}", atol, diffs)
    _differences(
        baseline["similarity"]["cka"], candidate["similarity"]["cka"], "similarity/cka", atol, diffs
    )
    missing: list[str] = []
    for codec, cand_probes in candidate["probes"].items():
        missing.extend(
            f"probes/{codec}/{k}" for k in ADDITIVE_PROBE_CODEC_KEYS if k not in cand_probes
        )
    missing.extend(k for k in ADDITIVE_PROBE_TOP_KEYS if k not in candidate)
    fixture = candidate.get("fixture_baselines", {})
    missing.extend(f"fixture_baselines/{k}" for k in ADDITIVE_FIXTURE_KEYS if k not in fixture)
    return diffs, missing


def compare_eval_cell(
    baseline: dict[str, Any], candidate: dict[str, Any], *, atol: float
) -> tuple[list[tuple[str, float]], list[str]]:
    """Return ``(locked_differences, missing_additive_keys)`` for one ``metrics_eval.json``."""
    diffs: list[tuple[str, float]] = []
    _differences(baseline["splits"], candidate["splits"], "splits", atol, diffs)
    missing: list[str] = []
    block = candidate.get("diagnostics")
    if not isinstance(block, dict):
        return diffs, ["diagnostics"]
    for split, keys in ADDITIVE_EVAL_SPLIT_KEYS.items():
        split_block = block.get("splits", {}).get(split, {})
        missing.extend(f"diagnostics/{split}/{k}" for k in keys if k not in split_block)
    return diffs, missing


def _cells(root: Path, pattern: str) -> dict[str, Path]:
    return {str(path.relative_to(root)): path for path in sorted(root.glob(pattern))}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(mode: str, baseline: Path, candidate: Path, atol: float) -> int:
    pattern = "*/rep*/probes.json" if mode == "probes" else "*/*/rep*/metrics_eval.json"
    compare = compare_probe_cell if mode == "probes" else compare_eval_cell
    base_cells = _cells(baseline, pattern)
    cand_cells = _cells(candidate, pattern)
    if not base_cells:
        print(f"no baseline cells under {baseline}", file=sys.stderr)
        return 1
    failures = 0
    dirty_flags: set[str] = set()
    worst = 0.0
    for rel, base_path in base_cells.items():
        cand_path = cand_cells.get(rel)
        if cand_path is None:
            print(f"MISSING candidate cell {rel}")
            failures += 1
            continue
        base_doc, cand_doc = _load(base_path), _load(cand_path)
        diffs, missing = compare(base_doc, cand_doc, atol=atol)
        flag_key = "probe_git_dirty" if mode == "probes" else "git_dirty"
        dirty_flags.add(f"{flag_key}={cand_doc.get(flag_key)}")
        for path, diff in diffs:
            print(f"LOCKED DIFF {rel} {path} |diff|={diff}")
            worst = max(worst, diff)
        for key in missing:
            print(f"MISSING ADDITIVE {rel} {key}")
        failures += int(bool(diffs or missing))
    extra = sorted(set(cand_cells) - set(base_cells))
    for rel in extra:
        print(f"NOTE candidate-only cell {rel}")
    print(
        f"{mode}: {len(base_cells)} baseline cells, {failures} failing, "
        f"{', '.join(sorted(dirty_flags))}"
    )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probes", "eval"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for locked float values (default 1e-6).",
    )
    args = parser.parse_args(argv)
    return _run(args.mode, args.baseline, args.candidate, args.atol)


if __name__ == "__main__":
    raise SystemExit(main())
