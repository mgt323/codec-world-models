"""Generate raw State episode datasets (HDF5 + seed checksum manifest).

Locked split parameters: ``configs.dataset_v0`` (PROGRAM_SPEC.md Dataset
splits). Emits active splits only — ``ood_noise`` remains deferred until an
OOD ``noise_scale`` is locked (do not invent one here).

Outputs (under ``--out-dir``):
- ``{split}.h5`` — flat State columns, one file per active split
- ``seed_checksums.json`` — per-split base_seed, sizes, difficulty, sha256

No codec encoding or tokenization.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.dataset_v0 import (  # noqa: E402
    N_EPISODES_DRY_RUN,
    N_STEPS_V0,
    SPLIT_SPECS,
    SplitSpec,
    active_split_specs,
    assert_episode_seed_ranges_disjoint,
    n_episodes_by_split_for_scale,
)
from data.generate_episodes import generate_split  # noqa: E402
from data.hdf5_episodes import write_episodes_hdf5_streaming  # noqa: E402
from world.schema import CausalRegime, SensorSource, State  # noqa: E402


def _difficulty_dict(difficulty) -> dict[str, float]:
    return dict(asdict(difficulty))


def _tee_stats(
    episodes: Iterator[tuple[int, list[State]]],
    *,
    regime_counts: Counter[str],
    coverage_counts: Counter[str],
) -> Iterator[tuple[int, list[State]]]:
    for seed, states in episodes:
        if states:
            regime_counts[states[0].latent.regime.value] += 1
        for state in states:
            if state.source is SensorSource.SENSOR_BOTH:
                coverage_counts["both"] += 1
            elif state.source is SensorSource.SENSOR_A:
                coverage_counts["a_only"] += 1
            elif state.source is SensorSource.SENSOR_B:
                coverage_counts["b_only"] += 1
            else:
                coverage_counts["neither"] += 1
        yield seed, states


def _regime_fractions(counts: Counter[str], n_episodes: int) -> dict[str, float]:
    denom = n_episodes or 1
    return {r.value: counts.get(r.value, 0) / denom for r in CausalRegime}


def _coverage_fractions(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {
        k: counts.get(k, 0) / total
        for k in ("both", "a_only", "b_only", "neither")
    }


def _report_split(
    spec: SplitSpec,
    *,
    n_episodes: int,
    checksum: str,
    regime_counts: Counter[str],
    coverage_counts: Counter[str],
) -> None:
    assert spec.difficulty is not None
    fractions = _regime_fractions(regime_counts, n_episodes)
    coverage = _coverage_fractions(coverage_counts)
    print(f"=== {spec.name} ===")
    print(f"  episodes: {n_episodes}")
    print(f"  base_seed: {spec.base_seed}")
    print(f"  difficulty: {_difficulty_dict(spec.difficulty)}")
    print(f"  sha256: {checksum}")
    print("  regime distribution (fraction / count):")
    for regime in CausalRegime:
        key = regime.value
        print(f"    {key}: {fractions[key]:.4f} ({regime_counts.get(key, 0)})")
    print("  partial-obs coverage (timestep fractions):")
    for key in ("both", "a_only", "b_only", "neither"):
        print(f"    {key}: {coverage[key]:.4f}")
    print()


def generate_dataset(
    out_dir: Path,
    *,
    n_episodes_by_split: dict[str, int],
    n_steps: int = N_STEPS_V0,
) -> dict[str, object]:
    """Generate active splits; write HDF5 + manifest; return manifest dict."""
    active = active_split_specs()
    assert_episode_seed_ranges_disjoint(active, n_episodes_by_split)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "n_episodes_by_split": dict(n_episodes_by_split),
        "n_steps": n_steps,
        "splits": {},
        "deferred_splits": {
            s.name: {"base_seed_reserved": s.base_seed, "note": s.note}
            for s in SPLIT_SPECS
            if s.status == "deferred"
        },
        "base_seed_policy": (
            "episode_seed = base_seed + episode_index; "
            "base seeds spaced by 1_000_000 (see configs.dataset_v0); "
            "dry_run is a strict prefix of v0 under the same base seeds"
        ),
    }

    print("Deferred splits (not generated):")
    for spec in SPLIT_SPECS:
        if spec.status == "deferred":
            print(f"  - {spec.name}: {spec.note}")
    print()

    splits_out: dict[str, object] = {}
    for spec in active:
        assert spec.difficulty is not None
        n_episodes = n_episodes_by_split[spec.name]
        regime_counts: Counter[str] = Counter()
        coverage_counts: Counter[str] = Counter(
            {"both": 0, "a_only": 0, "b_only": 0, "neither": 0}
        )
        h5_path = out_dir / f"{spec.name}.h5"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stream = _tee_stats(
                generate_split(
                    spec.name,
                    n_episodes,
                    spec.difficulty,
                    spec.base_seed,
                    n_steps,
                    intervention_fraction=0.0,
                ),
                regime_counts=regime_counts,
                coverage_counts=coverage_counts,
            )
            checksum = write_episodes_hdf5_streaming(
                h5_path,
                stream,
                n_episodes=n_episodes,
                n_steps=n_steps,
                split_name=spec.name,
            )
            for w in caught:
                print(f"WARNING [{spec.name}]: {w.message}")

        entry = {
            "base_seed": spec.base_seed,
            "n_episodes": n_episodes,
            "n_steps": n_steps,
            "difficulty": _difficulty_dict(spec.difficulty),
            "sha256": checksum,
            "hdf5": h5_path.name,
            "regime_distribution": _regime_fractions(regime_counts, n_episodes),
            "partial_obs_coverage": _coverage_fractions(coverage_counts),
        }
        splits_out[spec.name] = entry
        _report_split(
            spec,
            n_episodes=n_episodes,
            checksum=checksum,
            regime_counts=regime_counts,
            coverage_counts=coverage_counts,
        )

    manifest["splits"] = splits_out
    manifest_path = out_dir / "seed_checksums.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote manifest: {manifest_path}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate raw State HDF5 datasets + seed checksum manifest."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "artifacts" / "dataset_v0_dry_run",
        help="Output directory for HDF5 files and seed_checksums.json",
    )
    parser.add_argument(
        "--scale",
        choices=("dry_run", "v0"),
        default="dry_run",
        help="Episode counts: dry_run=2000/split; v0=50k train/val, 5k ood_partial_obs",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="Override: uniform episodes for every active split",
    )
    parser.add_argument(
        "--n-episodes-train",
        type=int,
        default=None,
        help="Override train episode count",
    )
    parser.add_argument(
        "--n-episodes-val",
        type=int,
        default=None,
        help="Override val episode count",
    )
    parser.add_argument(
        "--n-episodes-ood-partial-obs",
        type=int,
        default=None,
        help="Override ood_partial_obs episode count",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=N_STEPS_V0,
        help=f"Timesteps per episode (default {N_STEPS_V0})",
    )
    args = parser.parse_args(argv)

    counts = n_episodes_by_split_for_scale(args.scale)
    if args.n_episodes is not None:
        counts = {name: args.n_episodes for name in counts}
    if args.n_episodes_train is not None:
        counts["train"] = args.n_episodes_train
    if args.n_episodes_val is not None:
        counts["val"] = args.n_episodes_val
    if args.n_episodes_ood_partial_obs is not None:
        counts["ood_partial_obs"] = args.n_episodes_ood_partial_obs

    generate_dataset(
        args.out_dir,
        n_episodes_by_split=counts,
        n_steps=args.n_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
