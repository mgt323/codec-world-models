"""Compute Codec A surface diagnostics and write obs_codecs/diagnostics_A.json."""

from __future__ import annotations

from pathlib import Path

from obs_codecs.diagnostics import compute_codec_diagnostics, write_diagnostics_json
from obs_codecs.encode_a import encode_A
from obs_codecs.parity_audit import sample_diverse_parity_states
from world.schema import observe

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "obs_codecs" / "diagnostics_A.json"


def main() -> None:
    states = sample_diverse_parity_states(target_n=500)
    observations = [observe(s) for s in states]
    report = compute_codec_diagnostics(observations, encode=encode_A)
    path = write_diagnostics_json(report, OUT_PATH)
    print(f"wrote {path}")
    print(
        f"n={report.n_observations} vocab={report.vocab_size} "
        f"avg_tokens={report.avg_tokens_per_observation:.2f} "
        f"avg_chars={report.avg_string_length_chars:.2f}"
    )


if __name__ == "__main__":
    main()
