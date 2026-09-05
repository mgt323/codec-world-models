"""Run information-parity audit for a chosen codec on a diverse ~500-state sample.

Reuses ``obs_codecs.parity_audit.run_parity_audit`` — inject encode/parse only.
State fixtures come from ``world.parity_fixtures`` (not obs_codecs).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from obs_codecs.encode_a import encode_A, parse_A
from obs_codecs.encode_b import encode_B, parse_B
from obs_codecs.parity_audit import EncodeFn, ParseFactsFn, run_parity_audit
from world.parity_fixtures import sample_diverse_parity_states
from world.schema import observe

_CODECS: dict[str, tuple[EncodeFn, ParseFactsFn]] = {
    "A": (encode_A, parse_A),
    "B": (encode_B, parse_B),
}


def _coverage(states: list) -> dict[str, object]:
    return {
        "n_states": len(states),
        "regimes": sorted({s.latent.regime.value for s in states}),
        "noise_bins": sorted({s.noise_bin.value for s in states}),
        "partial_obs": dict(
            Counter(
                (
                    "both"
                    if s.a_observed and s.b_observed
                    else "a_only"
                    if s.a_observed and not s.b_observed
                    else "b_only"
                    if (not s.a_observed) and s.b_observed
                    else "neither"
                )
                for s in states
            )
        ),
        "n_samples_values": sorted({s.n_samples for s in states}),
        "n_samples_eq_1": sum(1 for s in states if s.n_samples == 1),
        "intervention_none": sum(1 for s in states if s.active_intervention is None),
        "intervention_set": sum(1 for s in states if s.active_intervention is not None),
    }


def run_for_codec(codec: str, *, target_n: int = 500) -> tuple[dict[str, object], bool]:
    """Sample states, observe once, run shared parity auditor on Observations."""
    if codec not in _CODECS:
        raise ValueError(f"Unknown codec {codec!r}; choose from {sorted(_CODECS)}")
    encode, parse = _CODECS[codec]
    states = sample_diverse_parity_states(target_n=target_n)
    observations = [observe(s) for s in states]
    report = run_parity_audit(observations, encode=encode, parse=parse)
    payload = {
        "codec": codec,
        "coverage": _coverage(states),
        "n_total": report.n_total,
        "n_passed": report.n_passed,
        "n_failed": report.n_total - report.n_passed,
        "pass_rate": report.pass_rate,
        "ok": report.ok,
        "mismatches": [m.to_detail_dict() for m in report.mismatches],
        "summary": report.summary(),
    }
    return payload, report.ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codec",
        choices=sorted(_CODECS),
        required=True,
        help="Codec whose encode/parse are injected into the shared runner",
    )
    parser.add_argument("--target-n", type=int, default=500)
    args = parser.parse_args(argv)

    payload, ok = run_for_codec(args.codec, target_n=args.target_n)
    summary = payload.pop("summary")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(summary, file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
