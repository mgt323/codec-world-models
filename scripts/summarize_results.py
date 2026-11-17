"""CLI and headless figure renderer for full_v0 results & scientific verdict."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.report import (
    MATCH_REGIMES,
    REPORT_CODECS,
    ReportBundle,
    format_summary_tables,
    format_verdict_summary,
    load_report_bundle,
)

# Okabe-Ito colorblind-safe palette
_REGIME_COLORS: dict[str, str] = {
    "compute_matched": "#0072B2",
    "info_matched": "#D55E00",
}
_REGIME_LABELS: dict[str, str] = {
    "compute_matched": "compute_matched",
    "info_matched": "info_matched",
}
_CODEC_COLORS: dict[str, str] = {
    "B": "#0072B2",
    "C": "#009E73",
    "D": "#E69F00",
    "A": "#CC79A7",
    "B_reverse": "#56B4E9",
    "B_bag": "#D55E00",
    "B_shuffle": "#000000",
}


def _style_axes(ax: object) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35, linewidth=0.8)


def _plot_grouped_whiskers(
    ax: object,
    stats_by_regime: dict[str, dict[str, object]],
    codecs: Sequence[str],
    *,
    show_legend: bool = False,
) -> None:
    xs = np.arange(len(codecs), dtype=np.float64)
    offsets = {"compute_matched": -0.14, "info_matched": 0.14}
    markers = {"compute_matched": "o", "info_matched": "s"}

    for regime in MATCH_REGIMES:
        x_pos = xs + offsets[regime]
        medians = np.array([stats_by_regime[regime][c].median for c in codecs], dtype=np.float64)
        err_lo = np.array(
            [max(0.0, stats_by_regime[regime][c].median - stats_by_regime[regime][c].q1) for c in codecs],
            dtype=np.float64,
        )
        err_hi = np.array(
            [max(0.0, stats_by_regime[regime][c].q3 - stats_by_regime[regime][c].median) for c in codecs],
            dtype=np.float64,
        )
        ax.errorbar(
            x_pos,
            medians,
            yerr=np.vstack([err_lo, err_hi]),
            fmt=markers[regime],
            color=_REGIME_COLORS[regime],
            ecolor=_REGIME_COLORS[regime],
            elinewidth=1.6,
            capsize=3.0,
            markersize=5.5,
            label=_REGIME_LABELS[regime] if show_legend else None,
            zorder=3,
        )

    ax.axvline(3.5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(codecs), rotation=25, ha="right", fontsize=8.5)
    _style_axes(ax)
    if show_legend:
        ax.legend(frameon=False, fontsize=8.5, loc="best")


def _save_figure(fig: Figure, out_path: Path) -> None:
    FigureCanvasAgg(fig)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def render_fig1_causal_metrics(bundle: ReportBundle, out_path: Path) -> None:
    fig = Figure(figsize=(11.0, 7.0), constrained_layout=True)
    axes = fig.subplots(2, 2)

    panels = [
        (axes[0, 0], "val", "regime_accuracy", "A. Regime Accuracy (val, ↑)", 0.25),
        (axes[0, 1], "intervention", "intervention_accuracy", "B. Intervention Accuracy (do(), ↑)", None),
        (axes[1, 0], "counterfactual", "counterfactual_divergence_mse", "C. Counterfactual Divergence MSE (↓ lower is better)", None),
        (axes[1, 1], "ood_partial_obs", "regime_accuracy", "D. OOD Partial-Obs Regime Accuracy (↑)", 0.25),
    ]

    for idx, (ax, split, metric_name, title, chance_line) in enumerate(panels):
        stats_by_regime = {
            regime: bundle.eval_summaries[(regime, split)][metric_name]
            for regime in MATCH_REGIMES
        }
        _plot_grouped_whiskers(
            ax,
            stats_by_regime,
            REPORT_CODECS,
            show_legend=(idx == 0),
        )
        if chance_line is not None:
            ax.axhline(chance_line, color="#666666", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")

    fig.suptitle(
        f"Behavioral & Causal Metrics by Codec ({bundle.provenance.get('budget_profile', 'full_v0')})",
        fontsize=12.0,
        fontweight="bold",
    )
    _save_figure(fig, out_path)


def render_fig2_compute_pareto(
    bundle: ReportBundle,
    out_path: Path,
    *,
    compute_axis: str = "tokens_consumed",
) -> None:
    fig = Figure(figsize=(7.5, 5.2), constrained_layout=True)
    ax = fig.subplots(1, 1)

    scale = 1e6 if compute_axis == "tokens_consumed" else 1.0
    xlabel = (
        "Tokens Consumed (millions)"
        if compute_axis == "tokens_consumed"
        else "Optimizer Steps"
    )

    for codec in REPORT_CODECS:
        color = _CODEC_COLORS[codec]
        info_cell = bundle.manifest_stats[("info_matched", codec)]
        comp_cell = bundle.manifest_stats[("compute_matched", codec)]
        info_stat = bundle.eval_summaries[("info_matched", "val")]["regime_accuracy"][codec]
        comp_stat = bundle.eval_summaries[("compute_matched", "val")]["regime_accuracy"][codec]

        x_info = (
            info_cell.tokens_consumed_median
            if compute_axis == "tokens_consumed"
            else info_cell.optimizer_steps_median
        ) / scale
        x_comp = (
            comp_cell.tokens_consumed_median
            if compute_axis == "tokens_consumed"
            else comp_cell.optimizer_steps_median
        ) / scale

        if abs(x_comp - x_info) > 1e-6 or abs(comp_stat.median - info_stat.median) > 1e-6:
            ax.annotate(
                "",
                xy=(x_comp, comp_stat.median),
                xytext=(x_info, info_stat.median),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.3, alpha=0.75),
                zorder=2,
            )

        # Open marker for info_matched
        ax.errorbar(
            x_info,
            info_stat.median,
            yerr=[[max(0.0, info_stat.median - info_stat.q1)], [max(0.0, info_stat.q3 - info_stat.median)]],
            fmt="o",
            mfc="white",
            mec=color,
            ecolor=color,
            mew=1.6,
            capsize=2.5,
            markersize=6.5,
            alpha=0.9,
            zorder=3,
        )
        # Filled marker for compute_matched
        ax.errorbar(
            x_comp,
            comp_stat.median,
            yerr=[[max(0.0, comp_stat.median - comp_stat.q1)], [max(0.0, comp_stat.q3 - comp_stat.median)]],
            fmt="o",
            mfc=color,
            mec=color,
            ecolor=color,
            capsize=2.5,
            markersize=6.5,
            label=codec,
            zorder=4,
        )

    ax.axhline(0.25, color="#666666", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_xlabel(xlabel, fontsize=10.0)
    ax.set_ylabel("Primary Causal Accuracy — Val Regime Accuracy (↑)", fontsize=10.0)
    ax.set_title(
        "Compute-Efficiency Scaling: info_matched (open) → compute_matched (filled)",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
    )
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="best")
    _save_figure(fig, out_path)


def render_fig3_probes_and_rsa(
    bundle: ReportBundle,
    out_path: Path,
    *,
    cka_regime: str = "compute_matched",
) -> None:
    fig = Figure(figsize=(11.0, 4.8), constrained_layout=True)
    ax_cka, ax_probe = fig.subplots(1, 2)

    cka = bundle.probe_aggregates[cka_regime].cka_median
    im = ax_cka.imshow(cka, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    n = len(REPORT_CODECS)
    ax_cka.set_xticks(np.arange(n))
    ax_cka.set_yticks(np.arange(n))
    ax_cka.set_xticklabels(list(REPORT_CODECS), rotation=30, ha="right", fontsize=8.5)
    ax_cka.set_yticklabels(list(REPORT_CODECS), fontsize=8.5)
    for i in range(n):
        for j in range(n):
            val = float(cka[i, j])
            txt_color = "black" if val > 0.6 else "white"
            ax_cka.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=7.5)
    ax_cka.set_title(f"A. Median Pairwise CKA ({cka_regime})", fontsize=10.5, fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax_cka, fraction=0.046, pad=0.04)

    stats_by_regime = {
        regime: bundle.probe_aggregates[regime].probe_metrics["hidden_c_r2"]
        for regime in MATCH_REGIMES
    }
    _plot_grouped_whiskers(ax_probe, stats_by_regime, REPORT_CODECS, show_legend=True)
    ax_probe.set_ylabel("Frozen Linear Probe R² (hidden_c)", fontsize=9.5)
    ax_probe.set_title("B. Frozen Linear Probe: hidden_c R² (↑)", fontsize=10.5, fontweight="bold", loc="left")

    fig.suptitle(
        f"Representational Similarity & Latent Probing ({bundle.provenance.get('budget_profile', 'full_v0')})",
        fontsize=12.0,
        fontweight="bold",
    )
    _save_figure(fig, out_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize full_v0 evaluation/probe results, evaluate scientific verdict, and render figures."
    )
    parser.add_argument(
        "--aggregates-dir",
        type=Path,
        default=_ROOT / "artifacts" / "runs" / "aggregates",
    )
    parser.add_argument(
        "--probes-dir",
        type=Path,
        default=_ROOT / "artifacts" / "probes" / "full_v0",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=_ROOT / "artifacts" / "runs" / "full_v0" / "manifest.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "artifacts" / "reports" / "full_v0",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf"),
        default="png",
    )
    parser.add_argument(
        "--compute-axis",
        choices=("tokens_consumed", "optimizer_steps"),
        default="tokens_consumed",
    )
    parser.add_argument(
        "--cka-regime",
        choices=("compute_matched", "info_matched"),
        default="compute_matched",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = load_report_bundle(
        aggregates_dir=args.aggregates_dir,
        probes_dir=args.probes_dir,
        manifest_path=args.manifest_path,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.format
    render_fig1_causal_metrics(bundle, args.out_dir / f"fig1_causal_metrics.{ext}")
    render_fig2_compute_pareto(
        bundle,
        args.out_dir / f"fig2_compute_pareto.{ext}",
        compute_axis=args.compute_axis,
    )
    render_fig3_probes_and_rsa(
        bundle,
        args.out_dir / f"fig3_probes_and_rsa.{ext}",
        cka_regime=args.cka_regime,
    )

    print(format_summary_tables(bundle))
    print()
    print(format_verdict_summary(bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


