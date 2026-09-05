"""Compare Codec A vs B surface diagnostics on one shared observation sample.

Reuses ``sample_diverse_parity_states`` once; runs ``compute_codec_diagnostics``
twice with the same Observations (encode_A / encode_B injected). Does not
modify ``diagnostics.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from obs_codecs.diagnostics import CodecDiagnostics, compute_codec_diagnostics
from obs_codecs.encode_a import encode_A
from obs_codecs.encode_b import encode_B
from world.parity_fixtures import sample_diverse_parity_states
from world.schema import observe

OUT_PATH = _ROOT / "obs_codecs" / "diagnostics_comparison_AB.json"

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


def _print_side_by_side(a: CodecDiagnostics, b: CodecDiagnostics) -> None:
    col_metric = "metric"
    col_a = "A"
    col_b = "B"
    width_m = max(len(col_metric), max(len(k) for k in _COMPARE_KEYS))
    width_a = max(len(col_a), 12)
    width_b = max(len(col_b), 12)

    header = f"{col_metric:<{width_m}}  {col_a:>{width_a}}  {col_b:>{width_b}}"
    print(header)
    print("-" * len(header))
    for key in _COMPARE_KEYS:
        va = getattr(a, key)
        vb = getattr(b, key)
        if isinstance(va, float):
            print(f"{key:<{width_m}}  {va:>{width_a}.4f}  {vb:>{width_b}.4f}")
        else:
            print(f"{key:<{width_m}}  {va:>{width_a}}  {vb:>{width_b}}")


def build_comparison(
    *, target_n: int = 500
) -> tuple[dict[str, object], CodecDiagnostics, CodecDiagnostics]:
    """One sample → observe once → diagnostics for A and B on identical Observations."""
    states = sample_diverse_parity_states(target_n=target_n)
    observations = [observe(s) for s in states]

    report_a = compute_codec_diagnostics(observations, encode=encode_A)
    report_b = compute_codec_diagnostics(observations, encode=encode_B)

    payload: dict[str, object] = {
        "n_observations": len(observations),
        "target_n": target_n,
        "same_sample": True,
        "tokenization_scheme": report_a.tokenization_scheme,
        "A": {
            **_metric_row(report_a),
            "full": report_a.to_dict(),
        },
        "B": {
            **_metric_row(report_b),
            "full": report_b.to_dict(),
        },
        "comparison": {
            key: {"A": getattr(report_a, key), "B": getattr(report_b, key)}
            for key in _COMPARE_KEYS
        },
    }
    return payload, report_a, report_b


def main() -> int:
    payload, report_a, report_b = build_comparison(target_n=500)
    _print_side_by_side(report_a, report_b)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
