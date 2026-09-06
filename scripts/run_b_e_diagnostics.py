"""Compare Codec B vs E-transforms surface diagnostics on one shared sample."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from obs_codecs.diagnostics import compute_codec_diagnostics
from obs_codecs.encode_b import encode_B
from obs_codecs.transforms_e import bag_b, reverse_b, shuffle_b
from world.parity_fixtures import sample_diverse_parity_states
from world.schema import observe

OUT_PATH = _ROOT / "obs_codecs" / "diagnostics_comparison_B_E.json"
_SEED = 0
_KEYS = ("vocab_size", "avg_tokens_per_observation", "avg_string_length_chars")


def main() -> int:
    observations = [observe(s) for s in sample_diverse_parity_states(target_n=500)]
    reports = {
        "B": compute_codec_diagnostics(observations, encode=encode_B),
        "B_shuffle": compute_codec_diagnostics(
            observations, encode=lambda o: shuffle_b(encode_B(o), _SEED)
        ),
        "B_reverse": compute_codec_diagnostics(
            observations, encode=lambda o: reverse_b(encode_B(o))
        ),
        "B_bag": compute_codec_diagnostics(
            observations, encode=lambda o: bag_b(encode_B(o), _SEED)
        ),
    }

    width_m = max(len("metric"), max(len(k) for k in _KEYS))
    width_v = 12
    header = f"{'metric':<{width_m}}" + "".join(f"  {name:>{width_v}}" for name in reports)
    print(header)
    print("-" * len(header))
    for key in _KEYS:
        cells = []
        for report in reports.values():
            value = getattr(report, key)
            cells.append(
                f"  {value:>{width_v}.4f}"
                if isinstance(value, float)
                else f"  {value:>{width_v}}"
            )
        print(f"{key:<{width_m}}" + "".join(cells))

    payload = {
        "n_observations": len(observations),
        "seed": _SEED,
        "tokenization_scheme": reports["B"].tokenization_scheme,
        "comparison": {
            key: {name: getattr(report, key) for name, report in reports.items()}
            for key in _KEYS
        },
        **{
            name: {
                "vocab_size": report.vocab_size,
                "avg_tokens_per_observation": report.avg_tokens_per_observation,
                "avg_string_length_chars": report.avg_string_length_chars,
                "token_histogram": report.token_histogram,
            }
            for name, report in reports.items()
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
