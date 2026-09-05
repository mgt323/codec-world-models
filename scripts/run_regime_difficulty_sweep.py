"""Run difficulty_sweep and print full accuracy + confusion tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.regime_baseline import difficulty_sweep, format_sweep_row


def main() -> int:
    rows = difficulty_sweep(
        noise_scale_values=[0.05, 0.1, 0.15, 0.2, 0.3],
        confounding_strength_values=[0.0],
        n_episodes_per_cell=100,
        n_steps=30,
        start_seed=0,
    )
    for row in rows:
        print(format_sweep_row(row))
        print()

    out = _ROOT / "eval" / "difficulty_sweep_noise_c0.json"
    serializable = []
    for row in rows:
        serializable.append(dict(row))
    out.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
