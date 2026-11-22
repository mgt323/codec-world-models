"""scripts/verify_b2_additive.py: locked keys equal, additive keys present."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.verify_b2_additive import compare_eval_cell, compare_probe_cell, main


def _probe_doc() -> dict:
    return {
        "probe_git_dirty": False,
        "probes": {
            "A": {"hidden_c_r2": 0.04, "regime_accuracy": 0.52, "next_b_r2": 0.65},
            "B": {"hidden_c_r2": 0.03, "regime_accuracy": 0.61, "next_b_r2": 0.67},
        },
        "similarity": {"cka": {"A": {"A": 1.0, "B": 0.9}, "B": {"A": 0.9, "B": 1.0}}},
    }


def _with_additive(doc: dict) -> dict:
    out = copy.deepcopy(doc)
    for probes in out["probes"].values():
        probes["hidden_c_r2_common_cause"] = 0.9
    out["fixture_baselines"] = {
        "persistence_next_b_r2": 0.42,
        "running_mean_next_b_r2": 0.67,
        "oracle_next_b_r2": 0.72,
    }
    return out


def _eval_doc() -> dict:
    return {
        "git_dirty": False,
        "splits": {
            "val": {"next_state_mse": 0.1, "regime_accuracy": 0.6, "ood_regime_accuracy": None},
            "intervention": {"intervention_accuracy": 0.375, "parse_fail_rate": 0.094},
        },
    }


def _eval_with_additive(doc: dict) -> dict:
    out = copy.deepcopy(doc)
    out["diagnostics"] = {
        "splits": {
            "val": {"next_state_mse_median": 0.01, "next_state_mse_winsorized": 0.02},
            "ood_partial_obs": {"next_state_mse_median": 0.01, "next_state_mse_winsorized": 0.02},
            "intervention": {
                "intervention_accuracy_both_observed": 0.4,
                "intervention_accuracy_parsed": 0.4,
                "zero_baseline_acc": 0.375,
                "oracle_acc": 0.56,
            },
            "counterfactual": {"cf_step_mse_at_intervention": 0.1},
        }
    }
    return out


def test_probe_cell_accepts_additive_only_change() -> None:
    base = _probe_doc()
    diffs, missing = compare_probe_cell(base, _with_additive(base), atol=1e-6)
    assert diffs == [] and missing == []


def test_probe_cell_flags_locked_drift_and_missing_additive() -> None:
    base = _probe_doc()
    drifted = _with_additive(base)
    drifted["probes"]["B"]["next_b_r2"] = 0.68
    diffs, _missing = compare_probe_cell(base, drifted, atol=1e-6)
    assert [path for path, _ in diffs] == ["probes/B/next_b_r2"]
    diffs, missing = compare_probe_cell(base, copy.deepcopy(base), atol=1e-6)
    assert diffs == []
    assert "fixture_baselines" in missing
    assert "probes/A/hidden_c_r2_common_cause" in missing


def test_probe_cell_tolerance() -> None:
    base = _probe_doc()
    near = _with_additive(base)
    near["probes"]["A"]["hidden_c_r2"] += 1e-8
    assert compare_probe_cell(base, near, atol=1e-6)[0] == []
    assert compare_probe_cell(base, near, atol=1e-10)[0] != []


def test_eval_cell_accepts_additive_only_change_and_flags_drift() -> None:
    base = _eval_doc()
    diffs, missing = compare_eval_cell(base, _eval_with_additive(base), atol=1e-6)
    assert diffs == [] and missing == []
    drifted = _eval_with_additive(base)
    drifted["splits"]["intervention"]["parse_fail_rate"] = 0.1
    diffs, _ = compare_eval_cell(base, drifted, atol=1e-6)
    assert [path for path, _ in diffs] == ["splits/intervention/parse_fail_rate"]
    _diffs, missing = compare_eval_cell(base, copy.deepcopy(base), atol=1e-6)
    assert missing == ["diagnostics"]


def test_main_exit_codes_for_probes(tmp_path: Path) -> None:
    base_root, cand_root = tmp_path / "base", tmp_path / "cand"
    for root, doc in ((base_root, _probe_doc()), (cand_root, _with_additive(_probe_doc()))):
        cell = root / "compute_matched" / "rep0"
        cell.mkdir(parents=True)
        (cell / "probes.json").write_text(json.dumps(doc), encoding="utf-8")
    argv = ["probes", "--baseline", str(base_root), "--candidate", str(cand_root)]
    assert main(argv) == 0
    (cand_root / "compute_matched" / "rep0" / "probes.json").write_text(
        json.dumps(_probe_doc()), encoding="utf-8"
    )
    assert main(argv) == 1  # additive keys missing
    assert main(["probes", "--baseline", str(tmp_path / "none"), "--candidate", str(cand_root)]) == 1
