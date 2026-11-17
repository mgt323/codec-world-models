"""Unit tests for eval.report and scripts.summarize_results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval.metrics import median_iqr
from eval.report import (
    MATCH_REGIMES,
    REPORT_CODECS,
    load_and_aggregate_probes,
    load_report_bundle,
)
from scripts.summarize_results import main as summarize_main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_synthetic_artifacts(
    tmp_path: Path,
    *,
    git_commit: str = "abc123def456",
    parse_rate: float = 0.99,
    val_mse_override: float | None = None,
    overlap_e_iqr: bool = False,
    swap_info_rank: bool = False,
) -> tuple[Path, Path, Path]:
    aggregates_dir = tmp_path / "aggregates"
    probes_dir = tmp_path / "probes"
    manifest_path = tmp_path / "manifest.json"

    skewed_base = np.array([0.10, 0.11, 0.14, 0.22, 0.39], dtype=np.float64)

    cells = []
    for regime in MATCH_REGIMES:
        for codec in REPORT_CODECS:
            tokens = 108_400_000 if (regime == "compute_matched" or codec == "A") else 68_800_000
            steps = 13_250 if (regime == "compute_matched" or codec == "A") else 8_400
            for rep_id in range(5):
                cells.append(
                    {
                        "codec": codec,
                        "match_regime": regime,
                        "replication_id": rep_id,
                        "status": "completed",
                        "error": None,
                        "tokens_consumed": tokens,
                        "optimizer_steps": steps,
                        "T_eff": float(tokens),
                        "n_episodes": 50_000 if regime == "info_matched" else 78_770,
                    }
                )
    _write_json(
        manifest_path,
        {
            "budget_profile": "synthetic_v0",
            "git_commit": git_commit,
            "git_dirty": False,
            "n_replications": 5,
            "eval_status": "scored",
            "cells": cells,
        },
    )

    base_reg = {
        "B": [0.80, 0.81, 0.82, 0.83, 0.84],
        "C": [0.72, 0.73, 0.74, 0.75, 0.76],
        "D": [0.64, 0.65, 0.66, 0.67, 0.68],
        "A": [0.56, 0.57, 0.58, 0.59, 0.60],
        "B_reverse": [0.48, 0.49, 0.50, 0.51, 0.52],
        "B_bag": [0.40, 0.41, 0.42, 0.43, 0.44],
        "B_shuffle": [0.32, 0.33, 0.34, 0.35, 0.36],
    }
    base_cf = {
        "B": [1e-4, 1.2e-4, 1.4e-4, 1.6e-4, 1.8e-4],
        "C": [3e-4, 3.2e-4, 3.4e-4, 3.6e-4, 3.8e-4],
        "D": [5e-4, 5.2e-4, 5.4e-4, 5.6e-4, 5.8e-4],
        "A": [7e-4, 7.2e-4, 7.4e-4, 7.6e-4, 7.8e-4],
        "B_reverse": [9e-4, 9.2e-4, 9.4e-4, 9.6e-4, 9.8e-4],
        "B_bag": [1.1e-3, 1.12e-3, 1.14e-3, 1.16e-3, 1.18e-3],
        "B_shuffle": [1.3e-3, 1.32e-3, 1.34e-3, 1.36e-3, 1.38e-3],
    }

    for regime in MATCH_REGIMES:
        reg_vals = {k: list(v) for k, v in base_reg.items()}
        if overlap_e_iqr:
            reg_vals["B_reverse"] = [0.70, 0.75, 0.79, 0.83, 0.85]
        if swap_info_rank and regime == "info_matched":
            reg_vals["B"], reg_vals["C"] = list(base_reg["C"]), list(base_reg["B"])

        val_mse_vals = [0.05, 0.06, 0.07, 0.08, 0.09]
        if val_mse_override is not None and regime == "compute_matched":
            val_mse_vals[2] = val_mse_override

        _write_json(
            aggregates_dir / regime / "val" / "summary.json",
            {
                "git_commit": git_commit,
                "eval_split": "val",
                "match_regime": regime,
                "metrics": {
                    "next_state_parse_rate": {c: median_iqr([parse_rate] * 5) for c in REPORT_CODECS},
                    "next_state_mse": {
                        c: median_iqr(val_mse_vals if c == "B" else [0.05] * 5)
                        for c in REPORT_CODECS
                    },
                    "regime_accuracy": {c: median_iqr(reg_vals[c]) for c in REPORT_CODECS},
                    "calibration_ece": {c: median_iqr([0.05] * 5) for c in REPORT_CODECS},
                    "calibration_brier": {c: median_iqr([0.20] * 5) for c in REPORT_CODECS},
                },
            },
        )
        _write_json(
            aggregates_dir / regime / "intervention" / "summary.json",
            {
                "git_commit": git_commit,
                "eval_split": "intervention",
                "match_regime": regime,
                "metrics": {
                    "intervention_accuracy": {c: median_iqr(reg_vals[c]) for c in REPORT_CODECS},
                },
            },
        )
        _write_json(
            aggregates_dir / regime / "counterfactual" / "summary.json",
            {
                "git_commit": git_commit,
                "eval_split": "counterfactual",
                "match_regime": regime,
                "metrics": {
                    "counterfactual_divergence_mse": {
                        c: median_iqr(base_cf[c]) for c in REPORT_CODECS
                    },
                },
            },
        )
        _write_json(
            aggregates_dir / regime / "ood_partial_obs" / "summary.json",
            {
                "git_commit": git_commit,
                "eval_split": "ood_partial_obs",
                "match_regime": regime,
                "metrics": {
                    "regime_accuracy": {c: median_iqr(reg_vals[c]) for c in REPORT_CODECS},
                    "partial_obs_next_state_mse": {c: median_iqr([0.10] * 5) for c in REPORT_CODECS},
                    "next_state_parse_rate": {c: median_iqr([parse_rate] * 5) for c in REPORT_CODECS},
                },
            },
        )

        for rep_id in range(5):
            probes_map = {}
            for idx, codec in enumerate(REPORT_CODECS):
                shift = 0.05 * (len(REPORT_CODECS) - idx)
                probes_map[codec] = {
                    "hidden_c_r2": float(skewed_base[rep_id] + shift),
                    "regime_accuracy": float(0.45 + 0.03 * (len(REPORT_CODECS) - idx)),
                    "next_b_r2": float(0.85 + 0.01 * (len(REPORT_CODECS) - idx)),
                    "n_states": 10000,
                }
            cka_map: dict[str, dict[str, float]] = {c: {} for c in REPORT_CODECS}
            for i, left in enumerate(REPORT_CODECS):
                for j, right in enumerate(REPORT_CODECS):
                    if i == j:
                        val = 1.0
                    else:
                        val = float(
                            min(0.95, max(0.05, skewed_base[rep_id] + 0.35 - 0.05 * abs(i - j)))
                        )
                    cka_map[left][right] = val
                    cka_map[right][left] = val
            _write_json(
                probes_dir / regime / f"rep{rep_id}" / "probes.json",
                {
                    "frozen": True,
                    "git_commit": git_commit,
                    "git_dirty": False,
                    "match_regime": regime,
                    "n_states": 10000,
                    "replication": rep_id,
                    "probes": probes_map,
                    "similarity": {"cka": cka_map},
                },
            )

    return aggregates_dir, probes_dir, manifest_path


def test_probe_and_cka_aggregation_uses_linear_percentile(tmp_path: Path) -> None:
    _, probes_dir, _ = _write_synthetic_artifacts(tmp_path)
    agg = load_and_aggregate_probes(probes_dir)

    skewed_base = np.array([0.10, 0.11, 0.14, 0.22, 0.39], dtype=np.float64)
    for regime in MATCH_REGIMES:
        reg_agg = agg[regime]
        assert reg_agg.n_replications == 5

        for idx, codec in enumerate(REPORT_CODECS):
            shift = 0.05 * (len(REPORT_CODECS) - idx)
            expected_vals = skewed_base + shift
            q1, med, q3 = np.percentile(expected_vals, [25.0, 50.0, 75.0], method="linear")
            stats = reg_agg.probe_metrics["hidden_c_r2"][codec]
            assert stats.median == pytest.approx(float(med), abs=1e-12)
            assert stats.q1 == pytest.approx(float(q1), abs=1e-12)
            assert stats.q3 == pytest.approx(float(q3), abs=1e-12)
            assert stats.iqr == pytest.approx(float(q3 - q1), abs=1e-12)

        for i in range(len(REPORT_CODECS)):
            for j in range(len(REPORT_CODECS)):
                if i == j:
                    vals = np.ones(5, dtype=np.float64)
                else:
                    vals = np.clip(skewed_base + 0.35 - 0.05 * abs(i - j), 0.05, 0.95)
                q1, med, q3 = np.percentile(vals, [25.0, 50.0, 75.0], method="linear")
                assert reg_agg.cka_median[i, j] == pytest.approx(float(med), abs=1e-12)
                assert reg_agg.cka_iqr[i, j] == pytest.approx(float(q3 - q1), abs=1e-12)


def test_verdict_checks_sanity_iqr_separation_and_rank_shift(tmp_path: Path) -> None:
    # 1. Clean fixture -> sanity passes & clear IQR separation
    agg_dir, probes_dir, manifest_path = _write_synthetic_artifacts(tmp_path / "clean")
    clean_bundle = load_report_bundle(agg_dir, probes_dir, manifest_path)
    assert clean_bundle.verdict.sanity_passed is True
    assert clean_bundle.verdict.completion_ok is True
    assert clean_bundle.verdict.finite_ok is True
    assert clean_bundle.verdict.extreme_outliers == []
    for regime in MATCH_REGIMES:
        assert clean_bundle.verdict.b_vs_e_outcomes[(regime, "val.regime_accuracy")] == "CLEAR_IQR_SEPARATION"
        assert (
            clean_bundle.verdict.b_vs_e_outcomes[(regime, "counterfactual.counterfactual_divergence_mse")]
            == "CLEAR_IQR_SEPARATION"
        )
    assert clean_bundle.verdict.regime_rank_shifts["val.regime_accuracy"]["largest_delta_rank"] == 0.0

    # 2. Parse rate < 0.95 -> sanity fails
    agg_dir, probes_dir, manifest_path = _write_synthetic_artifacts(
        tmp_path / "low_parse", parse_rate=0.80
    )
    low_parse = load_report_bundle(agg_dir, probes_dir, manifest_path)
    assert low_parse.verdict.sanity_passed is False
    assert low_parse.verdict.parse_rate_passing_codecs["compute_matched"] == []

    # 3. Extreme MSE outlier (1e12) -> sanity fails & outlier recorded
    agg_dir, probes_dir, manifest_path = _write_synthetic_artifacts(
        tmp_path / "outlier", val_mse_override=1e12
    )
    outlier_bundle = load_report_bundle(agg_dir, probes_dir, manifest_path)
    assert outlier_bundle.verdict.sanity_passed is False
    assert len(outlier_bundle.verdict.extreme_outliers) == 1

    # 4. NaN value -> finite_ok False & sanity fails
    agg_dir, probes_dir, manifest_path = _write_synthetic_artifacts(
        tmp_path / "nan_case", val_mse_override=float("nan")
    )
    nan_bundle = load_report_bundle(agg_dir, probes_dir, manifest_path)
    assert nan_bundle.verdict.finite_ok is False
    assert nan_bundle.verdict.sanity_passed is False

    # 5. Overlapping E IQR -> MEDIAN_ADVANTAGE_OVERLAPPING_IQR
    agg_dir, probes_dir, manifest_path = _write_synthetic_artifacts(
        tmp_path / "overlap", overlap_e_iqr=True
    )
    overlap_bundle = load_report_bundle(agg_dir, probes_dir, manifest_path)
    assert (
        overlap_bundle.verdict.b_vs_e_outcomes[("compute_matched", "val.regime_accuracy")]
        == "MEDIAN_ADVANTAGE_OVERLAPPING_IQR"
    )

    # 6. Info -> compute rank swap -> changed_codecs populated
    agg_dir, probes_dir, manifest_path = _write_synthetic_artifacts(
        tmp_path / "rank_swap", swap_info_rank=True
    )
    swap_bundle = load_report_bundle(agg_dir, probes_dir, manifest_path)
    rs = swap_bundle.verdict.regime_rank_shifts["val.regime_accuracy"]
    assert rs["largest_delta_rank"] >= 1.0
    assert "B" in rs["changed_codecs"]
    assert "C" in rs["changed_codecs"]


def test_provenance_guard_rejects_missing_git_commit(tmp_path: Path) -> None:
    agg_dir, probes_dir, manifest_path = _write_synthetic_artifacts(
        tmp_path / "no_git", git_commit=""
    )
    with pytest.raises(ValueError, match="Invariant I13"):
        load_report_bundle(agg_dir, probes_dir, manifest_path)


def test_summarize_results_cli_renders_tables_and_three_figures_headless(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    agg_dir, probes_dir, manifest_path = _write_synthetic_artifacts(tmp_path / "cli_data")

    for fmt, magic in (("png", b"\x89PNG\r\n\x1a\n"), ("pdf", b"%PDF-")):
        out_dir = tmp_path / f"out_{fmt}"
        rc = summarize_main(
            [
                "--aggregates-dir",
                str(agg_dir),
                "--probes-dir",
                str(probes_dir),
                "--manifest-path",
                str(manifest_path),
                "--out-dir",
                str(out_dir),
                "--format",
                fmt,
            ]
        )
        assert rc == 0

        captured = capsys.readouterr().out
        assert "[compute_matched]" in captured
        assert "[info_matched]" in captured
        assert "[VERDICT SUMMARY]" in captured
        assert "Provenance: budget=synthetic_v0" in captured

        expected_files = sorted(
            [
                f"fig1_causal_metrics.{fmt}",
                f"fig2_compute_pareto.{fmt}",
                f"fig3_probes_and_rsa.{fmt}",
            ]
        )
        actual_files = sorted(p.name for p in out_dir.iterdir())
        assert actual_files == expected_files
        for fname in expected_files:
            fpath = out_dir / fname
            assert fpath.stat().st_size > 0
            assert fpath.read_bytes().startswith(magic)


