"""Compute full-train regex_v4 token rates for compute-matched budgeting.

Writes ``obs_codecs/token_rates_train_v4.json`` from encoded train parquet
(``artifacts/encoded/train_{A,B,C,D}.parquet`` by default). This file is the
authoritative rate source for ``COMPUTE_MATCH_BUDGETS.md``; do not use
``diagnostics_comparison_ABCD.json`` (parity sample only).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from obs_codecs.diagnostics import (  # noqa: E402
    TOKENIZATION_SCHEME,
    TOKENIZATION_SCHEME_VERSION,
    tokenize,
)

OUT_DEFAULT = _ROOT / "obs_codecs" / "token_rates_train_v4.json"
STEMS = ("A", "B", "C", "D")
B_FAMILY = ("B", "B_shuffle", "B_reverse", "B_bag")


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def avg_tokens_for_parquet(path: Path) -> tuple[int, int, float]:
    frame = pd.read_parquet(path, columns=["encoded_string"])
    n = len(frame)
    if n == 0:
        raise ValueError(f"Empty parquet: {path}")
    total = sum(len(tokenize(text)) for text in frame["encoded_string"].astype(str))
    return n, total, total / n


def build_payload(*, encoded_dir: Path, n_steps: int) -> dict[str, object]:
    rates: dict[str, dict[str, float | int]] = {}
    for stem in STEMS:
        path = encoded_dir / f"train_{stem}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        n_obs, total_tok, avg = avg_tokens_for_parquet(path)
        rates[stem] = {
            "n_observations": n_obs,
            "total_tokens": total_tok,
            "avg_tokens_per_observation": avg,
            "tokens_per_episode_at_n_steps": avg * n_steps,
            "source_parquet": str(path.relative_to(_ROOT)).replace("\\", "/"),
        }

    b_avg = float(rates["B"]["avg_tokens_per_observation"])
    inherited = {
        stem: {
            "avg_tokens_per_observation": b_avg,
            "tokens_per_episode_at_n_steps": b_avg * n_steps,
            "inherited_from": "B",
            "note": (
                "B-family shares B rates under regex_v4 "
                "(COMPUTE_MATCH_BUDGETS.md §4)."
            ),
        }
        for stem in B_FAMILY
        if stem != "B"
    }

    return {
        "role": "authoritative_token_rates_for_compute_matched_budgets",
        "split": "train",
        "n_steps_assumed": n_steps,
        "tokenization_scheme": TOKENIZATION_SCHEME,
        "tokenizer_scheme_version": TOKENIZATION_SCHEME_VERSION,
        "build_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "encoded_dir": str(encoded_dir.relative_to(_ROOT)).replace("\\", "/"),
        "avg_tokens_per_observation": {
            stem: rates[stem]["avg_tokens_per_observation"] for stem in STEMS
        },
        "avg_tokens_per_observation_rounded_3dp": {
            stem: round(float(rates[stem]["avg_tokens_per_observation"]), 3)
            for stem in STEMS
        },
        "codecs": rates,
        "b_family_inherited": inherited,
        "notes": [
            "Rates measured by re-tokenizing encoded train parquet under "
            f"{TOKENIZATION_SCHEME_VERSION} (strings unchanged).",
            "Use this file for COMPUTE_MATCH_BUDGETS.md / train configs.",
            "obs_codecs/diagnostics_comparison_ABCD.json is parity-sample "
            "diagnostics only — not a rate source for budgeting.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoded-dir",
        type=Path,
        default=_ROOT / "artifacts" / "encoded",
    )
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--n-steps", type=int, default=30)
    args = parser.parse_args(argv)

    payload = build_payload(encoded_dir=args.encoded_dir, n_steps=args.n_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {args.out}")
    rounded = payload["avg_tokens_per_observation_rounded_3dp"]
    assert isinstance(rounded, dict)
    for stem, avg in rounded.items():
        print(f"  {stem}: {avg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
