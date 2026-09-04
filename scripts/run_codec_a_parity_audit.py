"""Run Codec A parity audit on a diverse ~500-state sample; print full results."""

from __future__ import annotations

import json
import sys
from collections import Counter

from obs_codecs.encode_a import decode_A_facts, encode_A
from obs_codecs.parity_audit import run_parity_audit, sample_diverse_parity_states


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


def main() -> int:
    states = sample_diverse_parity_states(target_n=500)
    report = run_parity_audit(states, encode=encode_A, decode_facts=decode_A_facts)

    payload = {
        "coverage": _coverage(states),
        "n_total": report.n_total,
        "n_passed": report.n_passed,
        "n_failed": report.n_total - report.n_passed,
        "pass_rate": report.pass_rate,
        "ok": report.ok,
        "mismatches": [m.to_detail_dict() for m in report.mismatches],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(report.summary(), file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
