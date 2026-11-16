"""Pure data loaders, probe/CKA aggregation, verdict evaluation, and CLI tables.

This module intentionally has zero ``matplotlib`` imports so it remains
lightweight, deterministic, and unit-testable independently of plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from eval.metrics import median_iqr

REPORT_CODECS: tuple[str, ...] = ("B", "C", "D", "A", "B_reverse", "B_bag", "B_shuffle")
E_CONTROLS: tuple[str, ...] = ("B_reverse", "B_bag", "B_shuffle")
SYNTACTIC_ORDERED_CODECS: tuple[str, ...] = ("A", "B", "C", "B_reverse")
PERMUTED_CONTROLS: tuple[str, ...] = ("B_bag", "B_shuffle")
MATCH_REGIMES: tuple[str, ...] = ("compute_matched", "info_matched")
EVAL_SPLITS: tuple[str, ...] = ("val", "ood_partial_obs", "intervention", "counterfactual")

ORDERED_PARSE_RATE_MIN: float = 0.50
PERMUTED_PARSE_RATE_MAX: float = 0.25
EXTREME_MSE_THRESHOLD: float = 1_000.0
UNINFORMATIVE_ACCURACY_FLOOR: float = 0.10
INTERVENTION_PARSE_FAIL_INVALID_THRESHOLD: float = 0.50
INTERVENTION_PARSE_FAIL_COMPARE_MAX: float = 0.20
HIDDEN_C_R2_LOW_SPREAD_THRESHOLD: float = 0.05

PROBE_METRIC_KEYS: tuple[str, ...] = ("hidden_c_r2", "regime_accuracy", "next_b_r2")

REQUIRED_SPLIT_METRICS: dict[str, tuple[str, ...]] = {
    "val": ("next_state_parse_rate", "regime_accuracy"),
    "ood_partial_obs": ("regime_accuracy",),
    "intervention": ("intervention_accuracy",),
    "counterfactual": ("counterfactual_divergence_mse",),
}

_STATS_REL_TOL: float = 1e-6
_STATS_ABS_TOL: float = 1e-9

_REP_DIR_RE = re.compile(r"^rep(\d+)$")


@dataclass(frozen=True, slots=True)
class MetricStats:
    median: float
    q1: float
    q3: float
    iqr: float
    values: tuple[float, ...]
    rep_ids: tuple[int | str, ...] | None = None


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
    syntactic_separation_ok: bool
    ordered_parse_passing: dict[str, list[str]]
    permuted_parse_passing: dict[str, list[str]]
    extreme_outliers: list[str]
    b_vs_e_outcomes: dict[tuple[str, str], str]
    b_vs_e_notes: dict[tuple[str, str], str]
    baseline_comparisons: dict[str, dict[str, str]]
    regime_rank_shifts: dict[str, Any]
    probe_alignment: dict[str, Any]
    missing_cells: list[str] = field(default_factory=list)
    b_vs_e_organization: dict[tuple[str, str], str] = field(default_factory=dict)
    b_vs_e_direction: dict[tuple[str, str], str] = field(default_factory=dict)


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


def _extract_git_dirty(payload: dict[str, Any], source: Path) -> bool | None:
    if "git_dirty" not in payload:
        return None
    raw = payload["git_dirty"]
    if not isinstance(raw, bool):
        raise ValueError(f"Expected boolean git_dirty in {source}, got {raw!r}")
    return raw


def _extract_rep_ids(
    raw: dict[str, Any],
    top_level: dict[str, Any],
    n_values: int,
    context: str,
) -> tuple[int | str, ...] | None:
    for container in (raw, top_level):
        for key in ("rep_ids", "replica_ids", "replications"):
            candidate = container.get(key)
            if isinstance(candidate, list):
                if len(candidate) != n_values:
                    raise ValueError(
                        f"Replica ID list length mismatch in {context} ({key}): "
                        f"expected {n_values}, got {len(candidate)}"
                    )
                return tuple(
                    int(item) if isinstance(item, int) and not isinstance(item, bool) else str(item)
                    for item in candidate
                )
    return None


def _to_metric_stats(
    raw: dict[str, Any],
    *,
    rep_ids: tuple[int | str, ...] | None = None,
    context: str = "metric",
) -> MetricStats:
    values = tuple(float(v) for v in raw["values"])
    if not values:
        raise ValueError(f"Empty values list in {context}")

    median = float(raw["median"])
    q1 = float(raw["q1"])
    q3 = float(raw["q3"])
    iqr = float(raw["iqr"])

    if rep_ids is not None and len(rep_ids) != len(values):
        raise ValueError(
            f"Replica ID count ({len(rep_ids)}) != values count ({len(values)}) in {context}"
        )

    if all(math.isfinite(v) for v in values):
        if not all(math.isfinite(x) for x in (median, q1, q3, iqr)):
            raise ValueError(
                f"Non-finite summary stats for finite values in {context}: "
                f"q1={q1}, median={median}, q3={q3}, iqr={iqr}"
            )
        if not math.isclose(iqr, q3 - q1, rel_tol=_STATS_REL_TOL, abs_tol=_STATS_ABS_TOL):
            raise ValueError(
                f"Inconsistent IQR in {context}: iqr={iqr} != q3 - q1 ({q3 - q1})"
            )
        expected = median_iqr(list(values))
        for field_name, actual in (("q1", q1), ("median", median), ("q3", q3)):
            exp_val = float(expected[field_name])
            if not math.isclose(actual, exp_val, rel_tol=_STATS_REL_TOL, abs_tol=_STATS_ABS_TOL):
                raise ValueError(
                    f"Inconsistent {field_name} in {context}: {actual} != expected {exp_val}"
                )

    return MetricStats(
        median=median,
        q1=q1,
        q3=q3,
        iqr=iqr,
        values=values,
        rep_ids=rep_ids,
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


def _load_eval_aggregates_with_provenance(
    aggregates_dir: Path,
) -> tuple[
    dict[tuple[str, str], dict[str, dict[str, MetricStats]]],
    list[tuple[Path, str, bool | None]],
    int,
]:
    summaries: dict[tuple[str, str], dict[str, dict[str, MetricStats]]] = {}
    provenance_entries: list[tuple[Path, str, bool | None]] = []
    expected_n_reps: int | None = None

    for regime in MATCH_REGIMES:
        for split in EVAL_SPLITS:
            summary_path = aggregates_dir / regime / split / "summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing aggregate summary: {summary_path}")
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            commit = _require_git_commit(payload, summary_path)
            dirty = _extract_git_dirty(payload, summary_path)
            provenance_entries.append((summary_path, commit, dirty))

            metrics_raw = payload.get("metrics")
            if not isinstance(metrics_raw, dict):
                raise ValueError(f"Missing or invalid 'metrics' object in {summary_path}")

            for req_metric in REQUIRED_SPLIT_METRICS.get(split, ()):
                if req_metric not in metrics_raw:
                    raise ValueError(
                        f"Missing required metric '{req_metric}' in {regime}/{split} ({summary_path})"
                    )

            parsed_metrics: dict[str, dict[str, MetricStats]] = {}
            for metric_name, codec_map in metrics_raw.items():
                if not isinstance(codec_map, dict):
                    raise ValueError(
                        f"Expected codec map for {regime}/{split}/{metric_name} in {summary_path}"
                    )
                missing_codecs = [c for c in REPORT_CODECS if c not in codec_map]
                if missing_codecs:
                    raise ValueError(
                        f"Missing codecs {missing_codecs} in {regime}/{split}/{metric_name} "
                        f"({summary_path})"
                    )

                parsed_codecs: dict[str, MetricStats] = {}
                for codec, raw_stats in codec_map.items():
                    ctx = f"{regime}/{split}/{metric_name}/{codec} ({summary_path})"
                    raw_values = raw_stats.get("values")
                    if not isinstance(raw_values, list):
                        raise ValueError(f"Missing or invalid 'values' list in {ctx}")
                    rep_ids = _extract_rep_ids(raw_stats, payload, len(raw_values), ctx)
                    ms = _to_metric_stats(raw_stats, rep_ids=rep_ids, context=ctx)
                    if len(ms.values) < 3:
                        raise ValueError(
                            f"Expected >= 3 replications for {regime}/{split}/{metric_name}/{codec}, "
                            f"got {len(ms.values)} (Invariant I14)"
                        )
                    if expected_n_reps is None:
                        expected_n_reps = len(ms.values)
                    elif len(ms.values) != expected_n_reps:
                        raise ValueError(
                            f"Inconsistent replication count in {ctx}: "
                            f"expected {expected_n_reps}, got {len(ms.values)}"
                        )
                    parsed_codecs[codec] = ms
                parsed_metrics[metric_name] = parsed_codecs
            summaries[(regime, split)] = parsed_metrics

    assert expected_n_reps is not None
    return summaries, provenance_entries, expected_n_reps


def load_eval_aggregates(
    aggregates_dir: Path,
) -> dict[tuple[str, str], dict[str, dict[str, MetricStats]]]:
    """Load §5 evaluation aggregate summaries for all regimes and splits."""
    summaries, _, _ = _load_eval_aggregates_with_provenance(aggregates_dir)
    return summaries


def _load_and_aggregate_probes_with_provenance(
    probes_dir: Path,
    codecs: Sequence[str] = REPORT_CODECS,
) -> tuple[dict[str, RegimeProbeAggregate], list[tuple[Path, str, bool | None]]]:
    codec_list = tuple(codecs)
    aggregates: dict[str, RegimeProbeAggregate] = {}
    provenance_entries: list[tuple[Path, str, bool | None]] = []
    expected_n_reps: int | None = None

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

        if expected_n_reps is None:
            expected_n_reps = len(rep_entries)
        elif len(rep_entries) != expected_n_reps:
            raise ValueError(
                f"Probe replication count mismatch for {regime}: "
                f"expected {expected_n_reps}, got {len(rep_entries)}"
            )

        rep_payloads: list[tuple[Path, dict[str, Any]]] = []
        for _, probe_file in rep_entries:
            payload = json.loads(probe_file.read_text(encoding="utf-8"))
            commit = _require_git_commit(payload, probe_file)
            dirty = _extract_git_dirty(payload, probe_file)
            provenance_entries.append((probe_file, commit, dirty))
            rep_payloads.append((probe_file, payload))

        for probe_file, payload in rep_payloads:
            probes_map = payload.get("probes")
            if not isinstance(probes_map, dict):
                raise ValueError(f"Missing or invalid 'probes' object in {probe_file}")
            missing_probe_codecs = [c for c in codec_list if c not in probes_map]
            if missing_probe_codecs:
                raise ValueError(
                    f"Missing probe codecs {missing_probe_codecs} in {probe_file}"
                )
            for codec in codec_list:
                codec_probes = probes_map[codec]
                if not isinstance(codec_probes, dict):
                    raise ValueError(f"Invalid probe map for codec '{codec}' in {probe_file}")
                missing_keys = [k for k in PROBE_METRIC_KEYS if k not in codec_probes]
                if missing_keys:
                    raise ValueError(
                        f"Missing probe metrics {missing_keys} for codec '{codec}' in {probe_file}"
                    )

            cka_raw = payload.get("similarity", {}).get("cka")
            if not isinstance(cka_raw, dict):
                raise ValueError(f"Missing or invalid 'similarity.cka' object in {probe_file}")
            missing_cka_rows = [c for c in codec_list if c not in cka_raw]
            if missing_cka_rows:
                raise ValueError(
                    f"Missing CKA row codecs {missing_cka_rows} in {probe_file}"
                )
            for left_codec in codec_list:
                row = cka_raw[left_codec]
                if not isinstance(row, dict):
                    raise ValueError(f"Invalid CKA row for '{left_codec}' in {probe_file}")
                missing_cka_cols = [c for c in codec_list if c not in row]
                if missing_cka_cols:
                    raise ValueError(
                        f"Missing CKA column codecs {missing_cka_cols} in row '{left_codec}' "
                        f"of {probe_file}"
                    )

        probe_metrics: dict[str, dict[str, MetricStats]] = {}
        for metric_key in PROBE_METRIC_KEYS:
            per_codec: dict[str, MetricStats] = {}
            for codec in codec_list:
                vals = [float(p["probes"][codec][metric_key]) for _, p in rep_payloads]
                per_codec[codec] = _to_metric_stats_from_values(vals)
            probe_metrics[metric_key] = per_codec

        n_codecs = len(codec_list)
        cka_stack = np.empty((len(rep_payloads), n_codecs, n_codecs), dtype=np.float64)
        for rep_idx, (_, payload) in enumerate(rep_payloads):
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

    return aggregates, provenance_entries


def load_and_aggregate_probes(
    probes_dir: Path,
    codecs: Sequence[str] = REPORT_CODECS,
) -> dict[str, RegimeProbeAggregate]:
    """Load per-replication probes.json and aggregate with linear percentiles."""
    aggregates, _ = _load_and_aggregate_probes_with_provenance(probes_dir, codecs=codecs)
    return aggregates


def load_manifest_compute_stats(
    manifest_path: Path,
) -> tuple[dict[tuple[str, str], ComputeCellStats], dict[str, Any]]:
    """Parse manifest.json and summarize compute usage per (match_regime, codec)."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    git_commit = _require_git_commit(payload, manifest_path)
    git_dirty_explicit = _extract_git_dirty(payload, manifest_path)

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
        tokens = [
            float(v) for c in completed if (v := c.get("tokens_consumed")) is not None
        ]
        steps = [
            float(v) for c in completed if (v := c.get("optimizer_steps")) is not None
        ]
        t_eff_vals = [float(v) for c in completed if (v := c.get("T_eff")) is not None]
        n_ep_vals = [int(v) for c in completed if (v := c.get("n_episodes")) is not None]

        if len(set(t_eff_vals)) > 1:
            raise ValueError(
                f"Inconsistent T_eff within ({regime}, {codec}) in {manifest_path}: "
                f"{sorted(set(t_eff_vals))}"
            )
        if len(set(n_ep_vals)) > 1:
            raise ValueError(
                f"Inconsistent n_episodes within ({regime}, {codec}) in {manifest_path}: "
                f"{sorted(set(n_ep_vals))}"
            )

        tokens_median = (
            float(median_iqr(tokens)["median"]) if tokens else float("nan")
        )
        steps_median = (
            float(median_iqr(steps)["median"]) if steps else float("nan")
        )
        t_eff = t_eff_vals[0] if t_eff_vals else float("nan")
        n_episodes = n_ep_vals[0] if n_ep_vals else 0

        stats[(regime, codec)] = ComputeCellStats(
            codec=codec,
            match_regime=regime,
            n_completed=len(completed),
            n_total=len(cells),
            tokens_consumed_median=tokens_median,
            optimizer_steps_median=steps_median,
            t_eff=t_eff,
            n_episodes=n_episodes,
        )

    provenance = {
        "budget_profile": payload.get("budget_profile"),
        "git_commit": git_commit,
        "git_dirty": bool(git_dirty_explicit) if git_dirty_explicit is not None else False,
        "git_dirty_explicit": git_dirty_explicit,
        "n_replications": n_replications,
        "eval_status": payload.get("eval_status"),
        "eval_schema_version": payload.get("eval_schema_version"),
        "decoder_mode": payload.get("decoder_mode"),
    }
    return stats, provenance


def _validate_cross_artifact_provenance(
    manifest_path: Path,
    manifest_provenance: dict[str, Any],
    summary_entries: list[tuple[Path, str, bool | None]],
    probe_entries: list[tuple[Path, str, bool | None]],
) -> None:
    manifest_commit = manifest_provenance["git_commit"]
    manifest_dirty = manifest_provenance["git_dirty"]
    all_entries = summary_entries + probe_entries

    commit_mismatches = [
        f"{path}={commit}" for path, commit, _ in all_entries if commit != manifest_commit
    ]
    if commit_mismatches:
        raise ValueError(
            f"git_commit mismatch against manifest {manifest_path} ({manifest_commit}): "
            f"{'; '.join(commit_mismatches)}"
        )

    dirty_mismatches = [
        f"{path}={dirty}"
        for path, _, dirty in all_entries
        if dirty is not None and dirty != manifest_dirty
    ]
    if dirty_mismatches:
        raise ValueError(
            f"git_dirty mismatch against manifest {manifest_path} ({manifest_dirty}): "
            f"{'; '.join(dirty_mismatches)}"
        )


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
    if len(x) != len(y):
        raise ValueError(f"Spearman correlation inputs must have equal length: {len(x)} vs {len(y)}")
    if len(x) < 2:
        return float("nan")
    rx = _average_ranks(x, higher_is_better=True)
    ry = _average_ranks(y, higher_is_better=True)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denom = float(np.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))
    if denom == 0.0 or not math.isfinite(denom):
        return float("nan")
    return float(np.sum(dx * dy) / denom)


def _is_uninformative_floor(
    codec_map: dict[str, MetricStats],
    codecs: Sequence[str],
    *,
    higher_is_better: bool,
) -> bool:
    medians = [codec_map[c].median for c in codecs]
    iqrs = [codec_map[c].iqr for c in codecs]
    if all(math.isclose(iqr, 0.0, abs_tol=1e-12) for iqr in iqrs) and all(
        math.isclose(m, medians[0], abs_tol=1e-12) for m in medians
    ):
        return True
    if higher_is_better and all(m <= UNINFORMATIVE_ACCURACY_FLOOR for m in medians):
        return True
    return False


def _classify_b_vs_group(
    b_stats: MetricStats,
    group_stats: Sequence[MetricStats],
    *,
    higher_is_better: bool,
) -> str:
    if higher_is_better:
        clear_better = b_stats.q1 > max(s.q3 for s in group_stats)
        clear_worse = b_stats.q3 < max(s.q1 for s in group_stats)
        med_adv = b_stats.median > max(s.median for s in group_stats)
    else:
        clear_better = b_stats.q3 < min(s.q1 for s in group_stats)
        clear_worse = b_stats.q1 > min(s.q3 for s in group_stats)
        med_adv = b_stats.median < min(s.median for s in group_stats)

    if clear_better:
        return "CLEAR_IQR_SEPARATION"
    if clear_worse:
        return "CLEAR_IQR_WORSE"
    if med_adv:
        return "MEDIAN_ADVANTAGE_OVERLAPPING_IQR"
    return "NO_ADVANTAGE"


def _validate_verdict_inputs(
    eval_summaries: dict[tuple[str, str], dict[str, dict[str, MetricStats]]],
    probe_aggregates: dict[str, RegimeProbeAggregate],
    codec_list: tuple[str, ...],
) -> None:
    for regime in MATCH_REGIMES:
        for split, req_metrics in REQUIRED_SPLIT_METRICS.items():
            split_key = (regime, split)
            if split_key not in eval_summaries:
                raise ValueError(f"Missing eval summary for {regime}/{split}")
            split_map = eval_summaries[split_key]
            for metric_name in req_metrics:
                if metric_name not in split_map:
                    raise ValueError(f"Missing metric '{metric_name}' in {regime}/{split}")
                codec_map = split_map[metric_name]
                missing = [c for c in codec_list if c not in codec_map]
                if missing:
                    raise ValueError(
                        f"Missing codecs {missing} in {regime}/{split}/{metric_name}"
                    )

        if regime not in probe_aggregates:
            raise ValueError(f"Missing probe aggregate for regime '{regime}'")
        probe_agg = probe_aggregates[regime]
        for metric_key in PROBE_METRIC_KEYS:
            if metric_key not in probe_agg.probe_metrics:
                raise ValueError(f"Missing probe metric '{metric_key}' for {regime}")
            missing = [c for c in codec_list if c not in probe_agg.probe_metrics[metric_key]]
            if missing:
                raise ValueError(
                    f"Missing codecs {missing} in probe metric '{metric_key}' for {regime}"
                )


def evaluate_verdict(
    eval_summaries: dict[tuple[str, str], dict[str, dict[str, MetricStats]]],
    probe_aggregates: dict[str, RegimeProbeAggregate],
    manifest_stats: dict[tuple[str, str], ComputeCellStats],
    codecs: Sequence[str] = REPORT_CODECS,
    *,
    n_replications: int | None = None,
) -> VerdictReport:
    """Evaluate the 4-part objective scientific verdict from loaded artifacts."""
    codec_list = tuple(codecs)
    _validate_verdict_inputs(eval_summaries, probe_aggregates, codec_list)

    expected_n_reps = n_replications
    if expected_n_reps is None and probe_aggregates:
        expected_n_reps = next(iter(probe_aggregates.values())).n_replications

    # (a) Sanity gate
    missing_cells: list[str] = []
    expected_keys = {(regime, codec) for regime in MATCH_REGIMES for codec in codec_list}
    completion_ok = bool(manifest_stats) and expected_n_reps is not None and expected_n_reps > 0
    if expected_n_reps is not None and expected_n_reps > 0:
        for regime in MATCH_REGIMES:
            for codec in codec_list:
                cell = manifest_stats.get((regime, codec))
                if cell is None:
                    missing_cells.append(f"{regime}/{codec} (0/{expected_n_reps} cells)")
                    completion_ok = False
                else:
                    if cell.n_total < expected_n_reps:
                        missing_cells.append(
                            f"{regime}/{codec} ({cell.n_total}/{expected_n_reps} cells)"
                        )
                        completion_ok = False
                    elif cell.n_total != expected_n_reps:
                        completion_ok = False
                    if cell.n_completed != expected_n_reps:
                        completion_ok = False
        if set(manifest_stats.keys()) - expected_keys:
            completion_ok = False
    else:
        completion_ok = False

    finite_ok = True
    extreme_outliers: list[str] = []

    for (regime, split), metrics_map in eval_summaries.items():
        for metric_name, codec_map in metrics_map.items():
            is_mse = metric_name.endswith("_mse")
            for codec, stats in codec_map.items():
                for rep_idx, val in enumerate(stats.values):
                    if stats.rep_ids is not None and rep_idx < len(stats.rep_ids):
                        rid = stats.rep_ids[rep_idx]
                        rep_label = str(rid) if str(rid).startswith("rep") else f"rep{rid}"
                    else:
                        rep_label = f"rep_id=missing(idx={rep_idx})"

                    if not math.isfinite(val):
                        finite_ok = False
                        if is_mse:
                            extreme_outliers.append(
                                f"{regime}/{split}/{metric_name}/{codec}[{rep_label}]={val}"
                            )
                    elif is_mse and val > EXTREME_MSE_THRESHOLD:
                        extreme_outliers.append(
                            f"{regime}/{split}/{metric_name}/{codec}[{rep_label}]={val:.3g}"
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

    ordered_parse_passing: dict[str, list[str]] = {}
    permuted_parse_passing: dict[str, list[str]] = {}
    for regime in MATCH_REGIMES:
        parse_map = eval_summaries[(regime, "val")]["next_state_parse_rate"]
        ordered_parse_passing[regime] = [
            c
            for c in SYNTACTIC_ORDERED_CODECS
            if c in parse_map and parse_map[c].median > ORDERED_PARSE_RATE_MIN
        ]
        permuted_parse_passing[regime] = [
            c
            for c in PERMUTED_CONTROLS
            if c in parse_map and parse_map[c].median < PERMUTED_PARSE_RATE_MAX
        ]

    syntactic_separation_ok = all(
        len(ordered_parse_passing[r]) == len(SYNTACTIC_ORDERED_CODECS)
        and len(permuted_parse_passing[r]) == len(PERMUTED_CONTROLS)
        for r in MATCH_REGIMES
    )
    sanity_passed = completion_ok and finite_ok and syntactic_separation_ok

    # (b) Causal & structural discrimination (B vs E controls + baselines)
    primary_specs: tuple[tuple[str, str, bool], ...] = (
        ("val", "regime_accuracy", True),
        ("intervention", "intervention_accuracy", True),
        ("counterfactual", "counterfactual_divergence_mse", False),
        ("ood_partial_obs", "regime_accuracy", True),
    )
    b_vs_e_outcomes: dict[tuple[str, str], str] = {}
    b_vs_e_organization: dict[tuple[str, str], str] = {}
    b_vs_e_direction: dict[tuple[str, str], str] = {}
    b_vs_e_notes: dict[tuple[str, str], str] = {}

    for regime in MATCH_REGIMES:
        for split, metric_name, higher_is_better in primary_specs:
            codec_map = eval_summaries[(regime, split)][metric_name]
            b_stats = codec_map["B"]
            e_group = [codec_map[c] for c in E_CONTROLS]
            org_group = [codec_map[c] for c in PERMUTED_CONTROLS]
            dir_group = [codec_map["B_reverse"]]
            key = (regime, f"{split}.{metric_name}")

            if split == "intervention" and metric_name == "intervention_accuracy":
                pf_map = eval_summaries[(regime, "intervention")].get("parse_fail_rate")
                if pf_map is not None:
                    invalid_codecs = [
                        c
                        for c in codec_list
                        if c in pf_map and pf_map[c].median > INTERVENTION_PARSE_FAIL_INVALID_THRESHOLD
                    ]
                    if len(invalid_codecs) > len(codec_list) / 2:
                        b_vs_e_outcomes[key] = "MEASUREMENT_INVALID"
                        b_vs_e_organization[key] = "MEASUREMENT_INVALID"
                        b_vs_e_direction[key] = "MEASUREMENT_INVALID"
                        b_vs_e_notes[key] = (
                            f"Measurement invalid: parse_fail_rate > "
                            f"{INTERVENTION_PARSE_FAIL_INVALID_THRESHOLD:.2f} for "
                            f"{len(invalid_codecs)}/{len(codec_list)} codecs "
                            f"({', '.join(invalid_codecs)})"
                        )
                        continue

            if _is_uninformative_floor(codec_map, codec_list, higher_is_better=higher_is_better):
                b_vs_e_outcomes[key] = "UNINFORMATIVE_FLOOR"
                b_vs_e_organization[key] = "UNINFORMATIVE_FLOOR"
                b_vs_e_direction[key] = "UNINFORMATIVE_FLOOR"
                b_vs_e_notes[key] = (
                    "Uninformative floor: all codec medians <= floor or identical with zero IQR"
                )
                continue

            overall = _classify_b_vs_group(b_stats, e_group, higher_is_better=higher_is_better)
            org_outcome = _classify_b_vs_group(
                b_stats, org_group, higher_is_better=higher_is_better
            )
            dir_outcome = _classify_b_vs_group(
                b_stats, dir_group, higher_is_better=higher_is_better
            )
            b_vs_e_outcomes[key] = overall
            b_vs_e_organization[key] = org_outcome
            b_vs_e_direction[key] = dir_outcome
            b_vs_e_notes[key] = (
                f"org(B vs {{B_bag,B_shuffle}})={org_outcome}; "
                f"dir(B vs B_reverse)={dir_outcome}"
            )

    baseline_comparisons: dict[str, dict[str, str]] = {}
    for regime in MATCH_REGIMES:
        val_reg = eval_summaries[(regime, "val")]["regime_accuracy"]
        do_acc = eval_summaries[(regime, "intervention")]["intervention_accuracy"]
        do_pf = eval_summaries[(regime, "intervention")].get("parse_fail_rate")

        def _cmp_tag(left_stats: MetricStats, right_stats: MetricStats) -> str:
            if left_stats.q1 > right_stats.q3:
                return "IQR>"
            if left_stats.q3 < right_stats.q1:
                return "IQR<"
            if left_stats.median > right_stats.median:
                return "med>"
            return "med<="

        def _cmp_pair(left: str, right: str) -> str:
            vr_l, vr_r = val_reg[left], val_reg[right]
            do_l, do_r = do_acc[left], do_acc[right]
            val_tag = _cmp_tag(vr_l, vr_r)
            if do_pf is not None and left in do_pf and right in do_pf:
                pf_l, pf_r = do_pf[left].median, do_pf[right].median
                if (
                    pf_l >= INTERVENTION_PARSE_FAIL_COMPARE_MAX
                    or pf_r >= INTERVENTION_PARSE_FAIL_COMPARE_MAX
                ):
                    return (
                        f"val:{val_tag} ({vr_l.median:.3f} vs {vr_r.median:.3f}), "
                        f"do:DEFERRED(pf={pf_l:.3f}/{pf_r:.3f}>={INTERVENTION_PARSE_FAIL_COMPARE_MAX:.1f}; "
                        f"{do_l.median:.3f} vs {do_r.median:.3f})"
                    )
            do_tag = _cmp_tag(do_l, do_r)
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
        hidden_c_r2_spread = float(max(probe_c_meds) - min(probe_c_meds))
        cka = probe_aggregates[regime].cka_median
        b_i = codec_idx["B"]
        b_vs_struct = [float(cka[b_i, codec_idx[c]]) for c in ("C", "D", "A")]
        b_vs_perm = [float(cka[b_i, codec_idx[c]]) for c in ("B_bag", "B_shuffle")]
        b_vs_rev = float(cka[b_i, codec_idx["B_reverse"]])

        probe_alignment[regime] = {
            "rho_beh_vs_probe_regime": _spearman_rho(beh_meds, probe_reg_meds),
            "rho_beh_vs_hidden_c_r2": _spearman_rho(beh_meds, probe_c_meds),
            "hidden_c_r2_spread": hidden_c_r2_spread,
            "hidden_c_r2_low_spread": hidden_c_r2_spread < HIDDEN_C_R2_LOW_SPREAD_THRESHOLD,
            "cka_b_vs_structured_median": float(np.median(b_vs_struct)),
            "cka_b_vs_permuted_median": float(np.median(b_vs_perm)),
            "cka_b_vs_reverse": b_vs_rev,
        }

    return VerdictReport(
        sanity_passed=sanity_passed,
        completion_ok=completion_ok,
        missing_cells=missing_cells,
        finite_ok=finite_ok,
        syntactic_separation_ok=syntactic_separation_ok,
        ordered_parse_passing=ordered_parse_passing,
        permuted_parse_passing=permuted_parse_passing,
        extreme_outliers=extreme_outliers,
        b_vs_e_outcomes=b_vs_e_outcomes,
        b_vs_e_organization=b_vs_e_organization,
        b_vs_e_direction=b_vs_e_direction,
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
    eval_summaries, summary_prov, eval_n_reps = _load_eval_aggregates_with_provenance(
        aggregates_dir
    )
    probe_aggregates, probe_prov = _load_and_aggregate_probes_with_provenance(
        probes_dir, codecs=REPORT_CODECS
    )
    manifest_stats, provenance = load_manifest_compute_stats(manifest_path)

    _validate_cross_artifact_provenance(manifest_path, provenance, summary_prov, probe_prov)

    probe_n_reps = next(iter(probe_aggregates.values())).n_replications
    manifest_n_reps = int(provenance["n_replications"])
    if not (eval_n_reps == probe_n_reps == manifest_n_reps):
        raise ValueError(
            f"Replication count mismatch across artifacts: "
            f"eval_summaries={eval_n_reps}, probes={probe_n_reps}, manifest={manifest_n_reps}"
        )

    verdict = evaluate_verdict(
        eval_summaries,
        probe_aggregates,
        manifest_stats,
        codecs=REPORT_CODECS,
        n_replications=manifest_n_reps,
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


def _fmt_rho(val: float) -> str:
    return f"{val:.3f}" if math.isfinite(val) else "n/a"


def format_summary_tables(bundle: ReportBundle) -> str:
    """Format fixed-width median (IQR) tables, one per match_regime (never pooled)."""
    has_do_pf = all(
        "parse_fail_rate" in bundle.eval_summaries[(regime, "intervention")]
        and all(
            c in bundle.eval_summaries[(regime, "intervention")]["parse_fail_rate"]
            for c in REPORT_CODECS
        )
        for regime in MATCH_REGIMES
    )
    if has_do_pf:
        headers: tuple[str, ...] = (
            "Codec",
            "ParseRate",
            "ValRegime↑",
            "DoAcc↑",
            "DoParseFail↓",
            "CFDivMSE↓",
            "OODRegime↑",
            "ProbeC_R2↑",
            "ProbeReg↑",
            "NextB_R2↑",
        )
    else:
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
    e_control_set = set(E_CONTROLS)

    for regime in MATCH_REGIMES:
        rows: list[tuple[str, ...]] = []
        val_m = bundle.eval_summaries[(regime, "val")]
        do_m = bundle.eval_summaries[(regime, "intervention")]
        cf_m = bundle.eval_summaries[(regime, "counterfactual")]
        ood_m = bundle.eval_summaries[(regime, "ood_partial_obs")]
        pr_m = bundle.probe_aggregates[regime].probe_metrics

        for codec in REPORT_CODECS:
            if has_do_pf:
                rows.append(
                    (
                        codec,
                        _fmt_rate(val_m["next_state_parse_rate"][codec]),
                        _fmt_rate(val_m["regime_accuracy"][codec]),
                        _fmt_rate(do_m["intervention_accuracy"][codec]),
                        _fmt_rate(do_m["parse_fail_rate"][codec]),
                        _fmt_sci(cf_m["counterfactual_divergence_mse"][codec]),
                        _fmt_rate(ood_m["regime_accuracy"][codec]),
                        _fmt_rate(pr_m["hidden_c_r2"][codec]),
                        _fmt_rate(pr_m["regime_accuracy"][codec]),
                        _fmt_rate(pr_m["next_b_r2"][codec]),
                    )
                )
            else:
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

        actual_n_rep = len(val_m["regime_accuracy"][REPORT_CODECS[0]].values)
        header_line = _fmt_row(headers)
        sep_line = "-+-".join("-" * w for w in widths)
        lines = [f"[{regime}] (median (IQR), n_rep={actual_n_rep})"]
        lines.append(header_line)
        lines.append(sep_line)
        in_e_controls = False
        for codec, row in zip(REPORT_CODECS, rows):
            if not in_e_controls and codec in e_control_set:
                lines.append(sep_line)
                in_e_controls = True
            lines.append(_fmt_row(row))
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def format_verdict_summary(bundle: ReportBundle) -> str:
    """Format the concise [VERDICT SUMMARY] block."""
    v = bundle.verdict
    lines: list[str] = ["[VERDICT SUMMARY]"]

    # (a) Sanity gate
    ord_counts = ", ".join(
        f"{r}={len(v.ordered_parse_passing[r])}/{len(SYNTACTIC_ORDERED_CODECS)}"
        for r in MATCH_REGIMES
    )
    perm_counts = ", ".join(
        f"{r}={len(v.permuted_parse_passing[r])}/{len(PERMUTED_CONTROLS)}"
        for r in MATCH_REGIMES
    )
    lines.append(
        f"(a) Sanity Gate: {'PASS' if v.sanity_passed else 'FAIL'} | "
        f"completed={v.completion_ok}, finite={v.finite_ok}, "
        f"syntactic_sep={v.syntactic_separation_ok} "
        f"(ordered>{ORDERED_PARSE_RATE_MIN:.2f} [{ord_counts}], "
        f"permuted<{PERMUTED_PARSE_RATE_MAX:.2f} [{perm_counts}])"
    )
    if v.missing_cells:
        lines.append(
            f"    Missing/incomplete manifest cells (n={len(v.missing_cells)}): "
            f"{'; '.join(v.missing_cells)}"
        )
    if v.extreme_outliers:
        lines.append(
            f"    [WARN] MSE>{EXTREME_MSE_THRESHOLD:.0f} outliers (non-blocking, n={len(v.extreme_outliers)}):"
        )
        for outlier in v.extreme_outliers:
            lines.append(f"      - {outlier}")

    # (b) B vs E & baselines
    lines.append("(b) B vs E Controls & Baselines:")
    for regime in MATCH_REGIMES:
        lines.append(f"    {regime}:")
        for m in (
            "val.regime_accuracy",
            "intervention.intervention_accuracy",
            "counterfactual.counterfactual_divergence_mse",
            "ood_partial_obs.regime_accuracy",
        ):
            key = (regime, m)
            org_state = v.b_vs_e_organization.get(key, "N/A")
            dir_state = v.b_vs_e_direction.get(key, "N/A")
            note = v.b_vs_e_notes.get(key, "")
            lines.append(
                f"      {m.split('.')[0]}={v.b_vs_e_outcomes[key]} "
                f"[org={org_state}, dir={dir_state}] ({note})"
            )
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
        f"val.regime rho={_fmt_rho(val_rs['spearman_rho'])} "
        f"(max|dRank|={val_rs['largest_delta_rank']:.1f}, "
        f"changed={val_rs['changed_codecs']}); "
        f"ood.regime rho={_fmt_rho(ood_rs['spearman_rho'])} "
        f"(max|dRank|={ood_rs['largest_delta_rank']:.1f})"
    )

    # (d) Behavioral vs probe alignment
    lines.append("(d) Behavioral vs Probe Alignment & CKA:")
    for regime in MATCH_REGIMES:
        pa = v.probe_alignment[regime]
        rho_reg_str = _fmt_rho(pa["rho_beh_vs_probe_regime"])
        rho_c_str = _fmt_rho(pa["rho_beh_vs_hidden_c_r2"])
        low_spread_flag = " [low-spread; not a signal]" if pa.get("hidden_c_r2_low_spread") else ""
        lines.append(
            f"    {regime}: rho(valReg,probeReg)={rho_reg_str}, "
            f"rho(valReg,hiddenC_R2)={rho_c_str}{low_spread_flag}, "
            f"hiddenC_R2_spread={pa['hidden_c_r2_spread']:.3f}, "
            f"CKA(B,{{C,D,A}})={pa['cka_b_vs_structured_median']:.2f} vs "
            f"CKA(B,{{B_bag,B_shuffle}})={pa['cka_b_vs_permuted_median']:.2f} "
            f"(B_rev={pa['cka_b_vs_reverse']:.2f})"
        )

    prov = bundle.provenance
    extra_prov = ""
    if prov.get("eval_schema_version") is not None:
        extra_prov += f", eval_schema_version={prov.get('eval_schema_version')}"
    if prov.get("decoder_mode") is not None:
        extra_prov += f", decoder_mode={prov.get('decoder_mode')}"
    lines.append(
        f"Provenance: budget={prov.get('budget_profile')}, "
        f"git_commit={prov.get('git_commit')}, "
        f"git_dirty={prov.get('git_dirty')}, "
        f"n_replications={prov.get('n_replications')}, "
        f"eval_status={prov.get('eval_status')}{extra_prov}"
    )
    return "\n".join(lines)