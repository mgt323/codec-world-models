"""Size the cross-segment-target residual before deciding whether to mask it.

Two effects are separated:

1. **Reporting bias** — with the locked ``all_non_pad`` policy the mean loss
   includes trivially-predictable ``<eos> -> <bos>`` targets whose density is
   codec-dependent. Measured exactly from a single run, since every step logs
   both ``loss_all_targets`` and ``loss_within_segment`` on identical weights.
2. **Optimization effect** — training under ``within_segment`` yields different
   weights. Measured by training each codec under both policies with identical
   seeds and pool, then comparing the within-segment loss curve (the common
   yardstick both policies can be scored on).

The comparison that matters is the **A-vs-B differential**: a bias identical
across codecs cancels in cross-codec claims; one that differs does not.

Writes ``artifacts/diagnostics/loss_mask/summary.json``. Nothing here changes
the locked training objective.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from train.config import build_run_config  # noqa: E402
from train.data import episodes_from_parquet, load_codec_vocab  # noqa: E402
from train.loop import run_training  # noqa: E402

PROFILE = "tier0"
REGIME = "compute_matched"
REPLICATION = 0
CODECS = ("A", "B")
OUT_ROOT = pathlib.Path("artifacts/diagnostics/loss_mask")
TAIL = 10  # steps averaged for the "converged" comparison


def read_metrics(run_dir: pathlib.Path) -> list[dict]:
    path = run_dir / "metrics_train.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").strip().splitlines()
    ]


def main() -> int:
    profile_pool: dict[str, list[list[int]]] = {}
    for codec in CODECS:
        vocab = load_codec_vocab(codec, "artifacts/vocabs")
        profile_pool[codec] = episodes_from_parquet(
            f"artifacts/encoded/train_{codec}.parquet", vocab, max_episodes=200
        )

    results: dict[str, dict] = {}
    for codec in CODECS:
        config = build_run_config(
            codec=codec,
            match_regime=REGIME,
            replication_id=REPLICATION,
            budget_profile=PROFILE,
        )
        for policy, restrict in (("all_non_pad", False), ("within_segment", True)):
            print(f"training {codec} / {policy} ...", flush=True)
            result = run_training(
                config,
                output_root=OUT_ROOT / policy,
                episode_pool=profile_pool[codec],
                runs_subdir=PROFILE,
                restrict_loss_to_segment=restrict,
            )
            rows = read_metrics(result.run_dir)
            results[f"{codec}/{policy}"] = {
                "codec": codec,
                "policy": policy,
                "steps": len(rows),
                "rows": rows,
            }

    summary: dict[str, dict] = {}
    print()
    print("=" * 78)
    print("1. REPORTING BIAS (same weights, both maskings scored per step)")
    print("=" * 78)
    print(
        f"{'codec':<8}{'loss_all':>11}{'loss_within':>13}"
        f"{'loss_cross':>12}{'bias':>11}{'cross %':>10}"
    )
    for codec in CODECS:
        rows = results[f"{codec}/all_non_pad"]["rows"][-TAIL:]
        all_t = statistics.fmean(r["loss_all_targets"] for r in rows)
        within = statistics.fmean(r["loss_within_segment"] for r in rows)
        cross = statistics.fmean(r["loss_cross_segment"] for r in rows)
        n_w = sum(r["n_within"] for r in rows)
        n_c = sum(r["n_cross"] for r in rows)
        pct = 100.0 * n_c / (n_w + n_c)
        summary.setdefault(codec, {})["reporting"] = {
            "loss_all_targets": all_t,
            "loss_within_segment": within,
            "loss_cross_segment": cross,
            "bias_all_minus_within": all_t - within,
            "cross_target_pct": pct,
        }
        print(
            f"{codec:<8}{all_t:>11.6f}{within:>13.6f}{cross:>12.6f}"
            f"{all_t - within:>11.6f}{pct:>9.4f}%"
        )

    bias_a = summary["A"]["reporting"]["bias_all_minus_within"]
    bias_b = summary["B"]["reporting"]["bias_all_minus_within"]
    print()
    print(f"per-codec bias      A {bias_a:+.6f}   B {bias_b:+.6f}")
    print(f"A-B differential    {bias_a - bias_b:+.6f} nats")

    print()
    print("=" * 78)
    print("2. OPTIMIZATION EFFECT (trained under each policy, scored within-segment)")
    print("=" * 78)
    print(
        f"{'codec':<8}{'trained_all':>14}{'trained_within':>16}"
        f"{'delta':>11}{'rel %':>10}"
    )
    for codec in CODECS:
        a_rows = results[f"{codec}/all_non_pad"]["rows"][-TAIL:]
        w_rows = results[f"{codec}/within_segment"]["rows"][-TAIL:]
        scored_a = statistics.fmean(r["loss_within_segment"] for r in a_rows)
        scored_w = statistics.fmean(r["loss_within_segment"] for r in w_rows)
        delta = scored_w - scored_a
        summary[codec]["optimization"] = {
            "within_loss_trained_all_non_pad": scored_a,
            "within_loss_trained_within_segment": scored_w,
            "delta": delta,
        }
        print(
            f"{codec:<8}{scored_a:>14.6f}{scored_w:>16.6f}"
            f"{delta:>11.6f}{100.0 * delta / scored_a:>9.3f}%"
        )

    d_a = summary["A"]["optimization"]["delta"]
    d_b = summary["B"]["optimization"]["delta"]
    print()
    print(f"A-B differential    {d_a - d_b:+.6f} nats")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(
            {
                "profile": PROFILE,
                "match_regime": REGIME,
                "replication_id": REPLICATION,
                "tail_steps_averaged": TAIL,
                "codecs": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print()
    print(f"wrote {OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
