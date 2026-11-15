"""Write intervention and counterfactual evaluation HDF5 splits.

These suites are not train/val/OOD partitions. Seeds start at 5_000_000 and
6_000_000 so they stay disjoint from the active dataset base seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.dataset_v0 import N_STEPS_V0  # noqa: E402
from data.hdf5_episodes import write_episodes_hdf5_streaming  # noqa: E402
from eval.score import (  # noqa: E402
    COUNTERFACTUAL_BASE_SEED,
    INTERVENTION_BASE_SEED,
    make_counterfactual_pairs,
    make_intervention_episodes,
)

DEFAULT_N = 256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=_ROOT / "artifacts" / "eval_splits")
    parser.add_argument("--n-episodes", type=int, default=DEFAULT_N)
    parser.add_argument("--n-steps", type=int, default=N_STEPS_V0)
    args = parser.parse_args(argv)
    if args.n_episodes < 1:
        raise SystemExit("n-episodes must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    interventions = make_intervention_episodes(
        args.n_episodes,
        base_seed=INTERVENTION_BASE_SEED,
        n_steps=args.n_steps,
    )
    intervention_checksum = write_episodes_hdf5_streaming(
        args.out_dir / "intervention.h5",
        iter(interventions),
        n_episodes=len(interventions),
        n_steps=args.n_steps,
        split_name="intervention",
    )
    pairs = make_counterfactual_pairs(
        args.n_episodes,
        base_seed=COUNTERFACTUAL_BASE_SEED,
        n_steps=args.n_steps,
    )
    observational = [pair[0] for pair in pairs]
    intervened = [pair[1] for pair in pairs]
    obs_checksum = write_episodes_hdf5_streaming(
        args.out_dir / "counterfactual_observational.h5",
        iter(observational),
        n_episodes=len(observational),
        n_steps=args.n_steps,
        split_name="counterfactual_observational",
    )
    int_checksum = write_episodes_hdf5_streaming(
        args.out_dir / "counterfactual_intervened.h5",
        iter(intervened),
        n_episodes=len(intervened),
        n_steps=args.n_steps,
        split_name="counterfactual_intervened",
    )
    manifest = {
        "n_episodes": args.n_episodes,
        "n_steps": args.n_steps,
        "intervention_base_seed": INTERVENTION_BASE_SEED,
        "counterfactual_base_seed": COUNTERFACTUAL_BASE_SEED,
        "intervention_sha256": intervention_checksum,
        "counterfactual_observational_sha256": obs_checksum,
        "counterfactual_intervened_sha256": int_checksum,
    }
    (args.out_dir / "eval_splits.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
