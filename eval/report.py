"""Pure data loaders, probe/CKA aggregation, verdict evaluation, and CLI tables.

This module intentionally has zero ``matplotlib`` imports so it remains
lightweight, deterministic, and unit-testable independently of plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from eval.metrics import median_iqr

REPORT_CODECS: tuple[str, ...] = ("B", "C", "D", "A", "B_reverse", "B_bag", "B_shuffle")
E_CONTROLS: tuple[str, ...] = ("B_reverse", "B_bag", "B_shuffle")
ORDERED_CODECS: tuple[str, ...] = ("B", "C", "D", "A", "B_reverse")
MATCH_REGIMES: tuple[str, ...] = ("compute_matched", "info_matched")
EVAL_SPLITS: tuple[str, ...] = ("val", "ood_partial_obs", "intervention", "counterfactual")

PARSE_RATE_THRESHOLD: float = 0.95
EXTREME_MSE_THRESHOLD: float = 1_000.0

PROBE_METRIC_KEYS: tuple[str, ...] = ("hidden_c_r2", "regime_accuracy", "next_b_r2")

_REP_DIR_RE = re.compile(r"^rep(\d+)$")


@dataclass(frozen=True, slots=True)
class MetricStats:
    median: float
    q1: float
    q3: float
    iqr: float
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ComputeCellStats:
    codec: str
    match_regime: str
    n_completed: int
    n_total: int
    tokens_consumed_median: float
    optimizer_steps_median: float
    t_eff: float
    n_episodes: int


@dataclass(frozen=True, slots=True)
class RegimeProbeAggregate:
    match_regime: str
    n_replications: int
    probe_metrics: dict[str, dict[str, MetricStats]]
    cka_median: np.ndarray
    cka_iqr: np.ndarray


@dataclass(frozen=True, slots=True)
class VerdictReport:
    sanity_passed: bool
    completion_ok: bool
    finite_ok: bool
    parse_rate_passing_codecs: dict[str, list[str]]
    extreme_outliers: list[str]
    b_vs_e_outcomes: dict[tuple[str, str], str]
    b_vs_e_notes: dict[tuple[str, str], str]
    baseline_comparisons: dict[str, dict[str, str]]
    regime_rank_shifts: dict[str, Any]
    probe_alignment: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReportBundle:
    eval_summaries: dict[tuple[str, str], dict[str, dict[str, MetricStats]]]
    probe_aggregates: dict[str, RegimeProbeAggregate]
    manifest_stats: dict[tuple[str, str], ComputeCellStats]
    provenance: dict[str, Any]
    verdict: VerdictReport


def _require_git_commit(payload: dict[str, Any], source: Path) -> str:
    commit = payload.get("git_commit")
    if not isinstance(commit, str) or not commit.strip():
        raise ValueError(f"Missing or empty git_commit in {source} (Invariant I13)")
    return commit.strip()


def _to_metric_stats(raw: dict[str, Any]) -> MetricStats:
    return MetricStats(
        median=float(raw["median"]),
        q1=float(raw["q1"]),
        q3=float(raw["q3"]),
        iqr=float(raw["iqr"]),
        values=tuple(float(v) for v in raw["values"]),
    )


def _to_metric_stats_from_values(values: list[float]) -> MetricStats:
    stats = median_iqr(values)
    return MetricStats(
        median=float(stats["median"]),
        q1=float(stats["q1"]),
        q3=float(stats["q3"]),
        iqr=float(stats["iqr"]),
        values=tuple(float(v) for v in stats["values"]),
    )


def load_eval_aggregates(
    aggregates_dir: Path,
) -> dict[tuple[str, str], dict[str, dict[str, MetricStats]]]:
    """Load §5 evaluation aggregate summaries for all regimes and splits."""
    summaries: dict[tuple[str, str], dict[str, dict[str, MetricStats]]] = {}
    for regime in MATCH_REGIMES:
        for split in EVAL_SPLITS:
            summary_path = aggregates_dir / regime / split / "summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing aggregate summary: {summary_path}")
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            _require_git_commit(payload, summary_path)

            metrics_raw = payload.get("metrics", {})
            parsed_metrics: dict[str, dict[str, MetricStats]] = {}
            for metric_name, codec_map in metrics_raw.items():
                parsed_codecs: dict[str, MetricStats] = {}
                for codec, raw_stats in codec_map.items():
                    ms = _to_metric_stats(raw_stats)
                    if len(ms.values) < 3:
                        raise ValueError(
                            f"Expected >= 3 replications for {regime}/{split}/{metric_name}/{codec}, "
                            f"got {len(ms.values)} (Invariant I14)"
                        )
                    parsed_codecs[codec] = ms
                parsed_metrics[metric_name] = parsed_codecs
            summaries[(regime, split)] = parsed_metrics
    return summaries


def load_and_aggregate_probes(
    probes_dir: Path,
    codecs: Sequence[str] = REPORT_CODECS,
) -> dict[str, RegimeProbeAggregate]:
    """Load per-replication probes.json and aggregate with linear percentiles."""
    codec_list = tuple(codecs)
    aggregates: dict[str, RegimeProbeAggregate] = {}

    for regime in MATCH_REGIMES:
        regime_dir = probes_dir / regime
        if not regime_dir.exists():
            raise FileNotFoundError(f"Missing probes directory for {regime}: {regime_dir}")

        rep_entries: list[tuple[int, Path]] = []
        for child in regime_dir.iterdir():
            if not child.is_dir():
                continue
            match = _REP_DIR_RE.match(child.name)
            if match is None:
                continue
            probe_file = child / "probes.json"
            if probe_file.exists():
                rep_entries.append((int(match.group(1)), probe_file))

        rep_entries.sort(key=lambda item: item[0])
        if len(rep_entries) < 3:
            raise ValueError(
                f"Expected >= 3 probe replications for {regime} in {regime_dir}, "
                f"got {len(rep_entries)} (Invariant I14)"
            )

        rep_payloads: list[dict[str, Any]] = []
        for _, probe_file in rep_entries:
            payload = json.loads(probe_file.read_text(encoding="utf-8"))
            _require_git_commit(payload, probe_file)
            rep_payloads.append(payload)

        probe_metrics: dict[str, dict[str, MetricStats]] = {}
        for metric_key in PROBE_METRIC_KEYS:
            per_codec: dict[str, MetricStats] = {}
            for codec in codec_list:
                vals = [float(p["probes"][codec][metric_key]) for p in rep_payloads]
                per_codec[codec] = _to_metric_stats_from_values(vals)
            probe_metrics[metric_key] = per_codec

        n_codecs = len(codec_list)
        cka_stack = np.empty((len(rep_payloads), n_codecs, n_codecs), dtype=np.float64)
        for rep_idx, payload in enumerate(rep_payloads):
            cka_raw = payload["similarity"]["cka"]
            for i, left_codec in enumerate(codec_list):
                for j, right_codec in enumerate(codec_list):
                    cka_stack[rep_idx, i, j] = float(cka_raw[left_codec][right_codec])

        q1, median, q3 = np.percentile(cka_stack, [25.0, 50.0, 75.0], axis=0, method="linear")
        aggregates[regime] = RegimeProbeAggregate(
            match_regime=regime,
            n_replications=len(rep_payloads),
            probe_metrics=probe_metrics,
            cka_median=median,
            cka_iqr=q3 - q1,
        )

    return aggregates


def load_manifest_compute_stats(
    manifest_path: Path,
) -> tuple[dict[tuple[str, str], ComputeCellStats], dict[str, Any]]:
    """Parse manifest.json and summarize compute usage per (match_regime, codec)."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    git_commit = _require_git_commit(payload, manifest_path)

    n_replications = int(payload.get("n_replications", 0))
    if n_replications < 3:
        raise ValueError(
            f"Expected n_replications >= 3 in {manifest_path}, got {n_replications} (Invariant I14)"
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in payload.get("cells", []):
        key = (str(cell["match_regime"]), str(cell["codec"]))
        grouped.setdefault(key, []).append(cell)

    stats: dict[tuple[str, str], ComputeCellStats] = {}
    for (regime, codec), cells in grouped.items():
        completed = [
            c for c in cells if c.get("status") == "completed" and c.get("error") is None
        ]
        tokens = [float(c["tokens_consumed"]) for c in cells]
        steps = [float(c["optimizer_steps"]) for c in cells]
        stats[(regime, codec)] = ComputeCellStats(
            codec=codec,
            match_regime=regime,
            n_completed=len(completed),
            n_total=len(cells),
            tokens_consumed_median=float(np.percentile(tokens, 50.0, method="linear")),
            optimizer_steps_median=float(np.percentile(steps, 50.0, method="linear")),
            t_eff=float(cells[0]["T_eff"]),
            n_episodes=int(cells[0]["n_episodes"]),
        )

    provenance = {
        "budget_profile": payload.get("budget_profile"),
        "git_commit": git_commit,
        "git_dirty": bool(payload.get("git_dirty", False)),
        "n_replications": n_replications,
        "eval_status": payload.get("eval_status"),
    }
    return stats, provenance


def _average_ranks(values: Sequence[float], *, higher_is_better: bool = True) -> np.ndarray:
    """Compute 1-based average ranks (1 = best) without SciPy."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    order_vals = -arr if higher_is_better else arr
    order = np.argsort(order_vals, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        while j < n and order_vals[order[j]] == order_vals[order[i]]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Pure NumPy Spearman rank correlation with average tie-handling."""
    rx = _average_ranks(x, higher_is_better=True)
    ry = _average_ranks(y, higher_is_better=True)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denom = float(np.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(dx * dy) / denom)


def evaluate_verdict(
    eval_summaries: dict[tuple[str, str], dict[str, dict[str, MetricStats]]],
    probe_aggregates: dict[str, RegimeProbeAggregate],
    manifest_stats: dict[tuple[str, str], ComputeCellStats],
    codecs: Sequence[str] = REPORT_CODECS,
) -> VerdictReport:
    """Evaluate the 4-part objective scientific verdict from loaded artifacts."""
    codec_list = tuple(codecs)

    # (a) Sanity gate
    completion_ok = bool(manifest_stats) and all(
        cell.n_total > 0 and cell.n_completed == cell.n_total
        for cell in manifest_stats.values()
    )

    finite_ok = True
    extreme_outliers: list[str] = []

    for (regime, split), metrics_map in eval_summaries.items():
        for metric_name, codec_map in metrics_map.items():
            is_mse = metric_name.endswith("_mse")
            for codec, stats in codec_map.items():
                for rep_idx, val in enumerate(stats.values):
                    if not math.isfinite(val):
                        finite_ok = False
                        if is_mse:
                            extreme_outliers.append(
                                f"{regime}/{split}/{metric_name}/{codec}[rep{rep_idx}]={val}"
                            )
                    elif is_mse and val > EXTREME_MSE_THRESHOLD:
                        extreme_outliers.append(
                            f"{regime}/{split}/{metric_name}/{codec}[rep{rep_idx}]={val:.3g}"
                        )

    for probe_agg in probe_aggregates.values():
        for codec_map in probe_agg.probe_metrics.values():
            for stats in codec_map.values():
                if not all(math.isfinite(v) for v in stats.values):
                    finite_ok = False
        if not (
            np.all(np.isfinite(probe_agg.cka_median))
            and np.all(np.isfinite(probe_agg.cka_iqr))
        ):
            finite_ok = False

    parse_rate_passing_codecs: dict[str, list[str]] = {}
    for regime in MATCH_REGIMES:
        parse_map = eval_summaries[(regime, "val")]["next_state_parse_rate"]
        parse_rate_passing_codecs[regime] = [
            c for c in codec_list if parse_map[c].median > PARSE_RATE_THRESHOLD
        ]

    all_parse_passed = all(
        len(parse_rate_passing_codecs[r]) == len(codec_list) for r in MATCH_REGIMES
    )
    sanity_passed = completion_ok and finite_ok and all_parse_passed and len(extreme_outliers) == 0

    # (b) Causal & structural discrimination (B vs E controls + baselines)
    primary_specs: tuple[tuple[str, str, bool], ...] = (
        ("val", "regime_accuracy", True),
        ("intervention", "intervention_accuracy", True),
        ("counterfactual", "counterfactual_divergence_mse", False),
        ("ood_partial_obs", "regime_accuracy", True),
    )
    b_vs_e_outcomes: dict[tuple[str, str], str] = {}
    b_vs_e_notes: dict[tuple[str, str], str] = {}

    for regime in MATCH_REGIMES:
        for split, metric_name, higher_is_better in primary_specs:
            codec_map = eval_summaries[(regime, split)][metric_name]
            b_stats = codec_map["B"]
            e_map = {c: codec_map[c] for c in E_CONTROLS}
            perm_map = {c: codec_map[c] for c in ("B_bag", "B_shuffle")}

            if higher_is_better:
                clear_all = b_stats.q1 > max(s.q3 for s in e_map.values())
                med_adv = b_stats.median > max(s.median for s in e_map.values())
                clear_perm = b_stats.q1 > max(s.q3 for s in perm_map.values())
            else:
                clear_all = b_stats.q3 < min(s.q1 for s in e_map.values())
                med_adv = b_stats.median < min(s.median for s in e_map.values())
                clear_perm = b_stats.q3 < min(s.q1 for s in perm_map.values())

            key = (regime, f"{split}.{metric_name}")
            if clear_all:
                b_vs_e_outcomes[key] = "CLEAR_IQR_SEPARATION"
                b_vs_e_notes[key] = "B IQR separates cleanly from all E controls"
            elif med_adv:
                b_vs_e_outcomes[key] = "MEDIAN_ADVANTAGE_OVERLAPPING_IQR"
                if clear_perm:
                    b_vs_e_notes[key] = "B IQR separates from {B_bag, B_shuffle}; overlaps B_reverse"
                else:
                    b_vs_e_notes[key] = "B median beats all E controls with overlapping IQRs"
            else:
                b_vs_e_outcomes[key] = "NO_ADVANTAGE"
                if clear_perm:
                    b_vs_e_notes[key] = "B_reverse ties/beats B; B IQR beats {B_bag, B_shuffle}"
                else:
                    b_vs_e_notes[key] = "At least one E control ties or beats B median"

    baseline_comparisons: dict[str, dict[str, str]] = {}
    for regime in MATCH_REGIMES:
        val_reg = eval_summaries[(regime, "val")]["regime_accuracy"]
        do_acc = eval_summaries[(regime, "intervention")]["intervention_accuracy"]

        def _cmp_pair(left: str, right: str) -> str:
            vr_l, vr_r = val_reg[left], val_reg[right]
            do_l, do_r = do_acc[left], do_acc[right]
            val_tag = "IQR>" if vr_l.q1 > vr_r.q3 else ("med>" if vr_l.median > vr_r.median else "med<=")
            do_tag = "IQR>" if do_l.q1 > do_r.q3 else ("med>" if do_l.median > do_r.median else "med<=")
            return (
                f"val:{val_tag} ({vr_l.median:.3f} vs {vr_r.median:.3f}), "
                f"do:{do_tag} ({do_l.median:.3f} vs {do_r.median:.3f})"
            )

        baseline_comparisons[regime] = {
            "B_vs_A": _cmp_pair("B", "A"),
            "B_vs_D": _cmp_pair("B", "D"),
            "C_vs_A": _cmp_pair("C", "A"),
        }

    # (c) Rank shifts (info_matched -> compute_matched)
    regime_rank_shifts: dict[str, Any] = {}
    for split, metric_name in (("val", "regime_accuracy"), ("ood_partial_obs", "regime_accuracy")):
        info_map = eval_summaries[("info_matched", split)][metric_name]
        comp_map = eval_summaries[("compute_matched", split)][metric_name]
        info_meds = [info_map[c].median for c in codec_list]
        comp_meds = [comp_map[c].median for c in codec_list]

        r_info = _average_ranks(info_meds, higher_is_better=True)
        r_comp = _average_ranks(comp_meds, higher_is_better=True)
        rho = _spearman_rho(info_meds, comp_meds)

        ranks_info = {c: float(r_info[i]) for i, c in enumerate(codec_list)}
        ranks_compute = {c: float(r_comp[i]) for i, c in enumerate(codec_list)}
        changed_codecs = {
            c: float(ranks_compute[c] - ranks_info[c])
            for c in codec_list
            if abs(ranks_compute[c] - ranks_info[c]) >= 1.0
        }
        largest_delta_rank = max(
            (abs(ranks_compute[c] - ranks_info[c]) for c in codec_list), default=0.0
        )
        deltas = {c: float(comp_meds[i] - info_meds[i]) for i, c in enumerate(codec_list)}

        regime_rank_shifts[f"{split}.{metric_name}"] = {
            "ranks_info": ranks_info,
            "ranks_compute": ranks_compute,
            "spearman_rho": rho,
            "changed_codecs": changed_codecs,
            "largest_delta_rank": float(largest_delta_rank),
            "compute_minus_info_deltas": deltas,
        }

    # (d) Behavioral-vs-probe alignment & CKA contrast
    codec_idx = {c: i for i, c in enumerate(codec_list)}
    probe_alignment: dict[str, Any] = {}
    for regime in MATCH_REGIMES:
        val_reg = eval_summaries[(regime, "val")]["regime_accuracy"]
        beh_meds = [val_reg[c].median for c in codec_list]
        probe_reg_meds = [
            probe_aggregates[regime].probe_metrics["regime_accuracy"][c].median
            for c in codec_list
        ]
        probe_c_meds = [
            probe_aggregates[regime].probe_metrics["hidden_c_r2"][c].median
            for c in codec_list
        ]
        cka = probe_aggregates[regime].cka_median
        b_i = codec_idx["B"]
        b_vs_struct = [float(cka[b_i, codec_idx[c]]) for c in ("C", "D", "A")]
        b_vs_perm = [float(cka[b_i, codec_idx[c]]) for c in ("B_bag", "B_shuffle")]
        b_vs_rev = float(cka[b_i, codec_idx["B_reverse"]])

        probe_alignment[regime] = {
            "rho_beh_vs_probe_regime": _spearman_rho(beh_meds, probe_reg_meds),
            "rho_beh_vs_hidden_c_r2": _spearman_rho(beh_meds, probe_c_meds),
            "hidden_c_r2_spread": float(max(probe_c_meds) - min(probe_c_meds)),
            "cka_b_vs_structured_median": float(np.median(b_vs_struct)),
            "cka_b_vs_permuted_median": float(np.median(b_vs_perm)),
            "cka_b_vs_reverse": b_vs_rev,
        }

    return VerdictReport(
        sanity_passed=sanity_passed,
        completion_ok=completion_ok,
        finite_ok=finite_ok,
        parse_rate_passing_codecs=parse_rate_passing_codecs,
        extreme_outliers=extreme_outliers,
        b_vs_e_outcomes=b_vs_e_outcomes,
        b_vs_e_notes=b_vs_e_notes,
        baseline_comparisons=baseline_comparisons,
        regime_rank_shifts=regime_rank_shifts,
        probe_alignment=probe_alignment,
    )


def load_report_bundle(
    aggregates_dir: Path,
    probes_dir: Path,
    manifest_path: Path,
) -> ReportBundle:
    """Load all report artifacts, validate invariants, and evaluate verdict."""
    eval_summaries = load_eval_aggregates(aggregates_dir)
    probe_aggregates = load_and_aggregate_probes(probes_dir, codecs=REPORT_CODECS)
    manifest_stats, provenance = load_manifest_compute_stats(manifest_path)
    verdict = evaluate_verdict(
        eval_summaries, probe_aggregates, manifest_stats, codecs=REPORT_CODECS
    )
    return ReportBundle(
        eval_summaries=eval_summaries,
        probe_aggregates=probe_aggregates,
        manifest_stats=manifest_stats,
        provenance=provenance,
        verdict=verdict,
    )


def _fmt_rate(stats: MetricStats) -> str:
    return f"{stats.median:.3f} ({stats.iqr:.3f})"


def _fmt_sci(stats: MetricStats) -> str:
    return f"{stats.median:.2e} ({stats.iqr:.2e})"


def format_summary_tables(bundle: ReportBundle) -> str:
    """Format fixed-width median (IQR) tables, one per match_regime (never pooled)."""
    headers = (
        "Codec",
        "ParseRate",
        "ValRegime↑",
        "DoAcc↑",
        "CFDivMSE↓",
        "OODRegime↑",
        "ProbeC_R2↑",
        "ProbeReg↑",
        "NextB_R2↑",
    )
    sections: list[str] = []

    for regime in MATCH_REGIMES:
        rows: list[tuple[str, ...]] = []
        val_m = bundle.eval_summaries[(regime, "val")]
        do_m = bundle.eval_summaries[(regime, "intervention")]
        cf_m = bundle.eval_summaries[(regime, "counterfactual")]
        ood_m = bundle.eval_summaries[(regime, "ood_partial_obs")]
        pr_m = bundle.probe_aggregates[regime].probe_metrics

        for codec in REPORT_CODECS:
            rows.append(
                (
                    codec,
                    _fmt_rate(val_m["next_state_parse_rate"][codec]),
                    _fmt_rate(val_m["regime_accuracy"][codec]),
                    _fmt_rate(do_m["intervention_accuracy"][codec]),
                    _fmt_sci(cf_m["counterfactual_divergence_mse"][codec]),
                    _fmt_rate(ood_m["regime_accuracy"][codec]),
                    _fmt_rate(pr_m["hidden_c_r2"][codec]),
                    _fmt_rate(pr_m["regime_accuracy"][codec]),
                    _fmt_rate(pr_m["next_b_r2"][codec]),
                )
            )

        widths = [
            max(len(headers[i]), *(len(r[i]) for r in rows))
            for i in range(len(headers))
        ]

        def _fmt_row(cols: tuple[str, ...]) -> str:
            parts = [cols[0].ljust(widths[0])]
            parts.extend(cols[i].rjust(widths[i]) for i in range(1, len(cols)))
            return " | ".join(parts)

        header_line = _fmt_row(headers)
        sep_line = "-+-".join("-" * w for w in widths)
        lines = [f"[{regime}] (median (IQR), n_rep={bundle.provenance.get('n_replications')})"]
        lines.append(header_line)
        lines.append(sep_line)
        for idx, row in enumerate(rows):
            if idx == 4:  # subtle separator before E controls
                lines.append(sep_line)
            lines.append(_fmt_row(row))
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def format_verdict_summary(bundle: ReportBundle) -> str:
    """Format the concise [VERDICT SUMMARY] block."""
    v = bundle.verdict
    lines: list[str] = ["[VERDICT SUMMARY]"]

    # (a) Sanity gate
    parse_counts = ", ".join(
        f"{r}={len(v.parse_rate_passing_codecs[r])}/{len(REPORT_CODECS)}"
        for r in MATCH_REGIMES
    )
    lines.append(
        f"(a) Sanity Gate: {'PASS' if v.sanity_passed else 'FAIL'} | "
        f"completed={v.completion_ok}, finite={v.finite_ok}, "
        f"parse_rate>{PARSE_RATE_THRESHOLD:.2f} [{parse_counts}], "
        f"mse_outliers>{EXTREME_MSE_THRESHOLD:.0f}={len(v.extreme_outliers)}"
    )
    if v.extreme_outliers:
        lines.append(f"    Outliers: {'; '.join(v.extreme_outliers[:4])}")

    # (b) B vs E & baselines
    lines.append("(b) B vs E Controls & Baselines:")
    for regime in MATCH_REGIMES:
        outcomes = [
            f"{m.split('.')[0]}={v.b_vs_e_outcomes[(regime, m)]}"
            for m in (
                "val.regime_accuracy",
                "intervention.intervention_accuracy",
                "counterfactual.counterfactual_divergence_mse",
                "ood_partial_obs.regime_accuracy",
            )
        ]
        lines.append(f"    {regime}: " + "; ".join(outcomes))
        base = v.baseline_comparisons[regime]
        lines.append(
            f"      Baselines: B vs A [{base['B_vs_A']}] | "
            f"B vs D [{base['B_vs_D']}] | C vs A [{base['C_vs_A']}]"
        )

    # (c) Rank shifts
    val_rs = v.regime_rank_shifts["val.regime_accuracy"]
    ood_rs = v.regime_rank_shifts["ood_partial_obs.regime_accuracy"]
    lines.append(
        f"(c) Info -> Compute Rank Shifts: "
        f"val.regime rho={val_rs['spearman_rho']:.3f} (max|dRank|={val_rs['largest_delta_rank']:.1f}, "
        f"changed={val_rs['changed_codecs']}); "
        f"ood.regime rho={ood_rs['spearman_rho']:.3f} (max|dRank|={ood_rs['largest_delta_rank']:.1f})"
    )

    # (d) Behavioral vs probe alignment
    lines.append("(d) Behavioral vs Probe Alignment & CKA:")
    for regime in MATCH_REGIMES:
        pa = v.probe_alignment[regime]
        lines.append(
            f"    {regime}: rho(valReg,probeReg)={pa['rho_beh_vs_probe_regime']:.3f}, "
            f"rho(valReg,hiddenC_R2)={pa['rho_beh_vs_hidden_c_r2']:.3f}, "
            f"hiddenC_R2_spread={pa['hidden_c_r2_spread']:.3f}, "
            f"CKA(B,{{C,D,A}})={pa['cka_b_vs_structured_median']:.2f} vs "
            f"CKA(B,{{B_bag,B_shuffle}})={pa['cka_b_vs_permuted_median']:.2f} "
            f"(B_rev={pa['cka_b_vs_reverse']:.2f})"
        )

    prov = bundle.provenance
    lines.append(
        f"Provenance: budget={prov.get('budget_profile')}, "
        f"git_commit={prov.get('git_commit')}, "
        f"git_dirty={prov.get('git_dirty')}, "
        f"n_replications={prov.get('n_replications')}"
    )
    return "\n".join(lines)


