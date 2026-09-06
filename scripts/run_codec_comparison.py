"""Compare Codec A / B / C / D surface diagnostics on one shared observation sample.

Reuses ``sample_diverse_parity_states`` once; runs ``compute_codec_diagnostics``
per codec with the same Observations. Does not modify ``diagnostics.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from obs_codecs.diagnostics import CodecDiagnostics, EncodeFn, compute_codec_diagnostics
from obs_codecs.encode_a import encode_A
from obs_codecs.encode_b import encode_B
from obs_codecs.encode_c import encode_C
from obs_codecs.encode_d import encode_D
from world.parity_fixtures import sample_diverse_parity_states
from world.schema import observe

OUT_PATH = _ROOT / "obs_codecs" / "diagnostics_comparison_ABCD.json"

_ENCODERS: dict[str, EncodeFn] = {
    "A": encode_A,
    "B": encode_B,
    "C": encode_C,
    "D": encode_D,
}

_COMPARE_KEYS = (
    "vocab_size",
    "avg_tokens_per_observation",
    "avg_string_length_chars",
)


def _metric_row(report: CodecDiagnostics) -> dict[str, float | int]:
    return {
        "vocab_size": report.vocab_size,
        "avg_tokens_per_observation": report.avg_tokens_per_observation,
        "avg_string_length_chars": report.avg_string_length_chars,
    }


def _print_side_by_side(reports: dict[str, CodecDiagnostics]) -> None:
    col_metric = "metric"
    width_m = max(len(col_metric), max(len(k) for k in _COMPARE_KEYS))
    width_v = 12

    header = f"{col_metric:<{width_m}}" + "".join(
        f"  {codec:>{width_v}}" for codec in reports
    )
    print(header)
    print("-" * len(header))
    for key in _COMPARE_KEYS:
        cells = []
        for report in reports.values():
            value = getattr(report, key)
            cells.append(
                f"  {value:>{width_v}.4f}" if isinstance(value, float)
                else f"  {value:>{width_v}}"
            )
        print(f"{key:<{width_m}}" + "".join(cells))


def build_comparison(
    *, target_n: int = 500
) -> tuple[dict[str, object], dict[str, CodecDiagnostics]]:
    """One sample → observe once → diagnostics per codec on identical Observations."""
    states = sample_diverse_parity_states(target_n=target_n)
    observations = [observe(s) for s in states]

    reports = {
        codec: compute_codec_diagnostics(observations, encode=encode)
        for codec, encode in _ENCODERS.items()
    }

    payload: dict[str, object] = {
        "n_observations": len(observations),
        "target_n": target_n,
        "same_sample": True,
        "tokenization_scheme": next(iter(reports.values())).tokenization_scheme,
        "comparison": {
            key: {codec: getattr(report, key) for codec, report in reports.items()}
            for key in _COMPARE_KEYS
        },
    }
    for codec, report in reports.items():
        payload[codec] = {**_metric_row(report), "full": report.to_dict()}
    return payload, reports


def main() -> int:
    payload, reports = build_comparison(target_n=500)
    _print_side_by_side(reports)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
