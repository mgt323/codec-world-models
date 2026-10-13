"""Build train-only vocabs from encoded parquet and report val/OOD OOV rates.

Example::

    python scripts/build_vocab.py \\
        --encoded-dir artifacts/encoded \\
        --out-dir artifacts/vocabs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, _ROOT)

from obs_codecs.vocab import (  # noqa: E402
    B_FAMILY_VARIANTS,
    VOCAB_GROUP_A,
    VOCAB_GROUP_B,
    VOCAB_GROUP_C,
    VOCAB_GROUP_D,
    build_and_save_vocabs,
)


def _print_report(report) -> None:
    scheme = report.tokenization_scheme.encode("ascii", errors="replace").decode("ascii")
    print("tokenization:", scheme)
    print("shared_b_family:", report.shared_b_family)
    print()
    print(f"{'group':<12} {'vocab':>10} {'numeric':>10} {'other':>8}")
    for key in (VOCAB_GROUP_A, VOCAB_GROUP_B, VOCAB_GROUP_C, VOCAB_GROUP_D):
        if key not in report.vocab_sizes:
            continue
        print(
            f"{key:<12} {report.vocab_sizes[key]:>10} "
            f"{report.numeric_token_counts[key]:>10} "
            f"{report.non_numeric_token_counts[key]:>8}"
        )
    if report.shared_b_family:
        print(f"  (B family aliases: {', '.join(B_FAMILY_VARIANTS)})")
    print()
    print(
        f"{'split':<18} {'variant':<12} {'tok_oov':>10} {'type_oov':>10} "
        f"{'n_tok':>10}"
    )
    for row in report.oov:
        print(
            f"{row.split:<18} {row.variant:<12} "
            f"{row.oov_token_rate:>10.4f} {row.oov_type_rate:>10.4f} "
            f"{row.n_tokens:>10}"
        )
    print()
    for note in report.notes:
        safe = note.encode("ascii", errors="replace").decode("ascii")
        print(f"NOTE: {safe}")


def main(argv: list[str] | None = None) -> int:
    # Windows consoles often default to a legacy code page; vocab notes / scheme
    # strings include Unicode punctuation from the locked tokenizer scheme.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=(
            "Build train-only regex_v4 vocabs per codec (shared B-family) "
            "and report val / ood_partial_obs OOV rates."
        )
    )
    parser.add_argument(
        "--encoded-dir",
        type=Path,
        default=_ROOT / "artifacts" / "encoded",
        help="Directory with {split}_{variant}.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "artifacts" / "vocabs",
        help="Output directory for vocab_*.json and vocab_report.json",
    )
    parser.add_argument(
        "--no-share-b-family",
        action="store_true",
        help="Build independent vocabs for B / B_shuffle / B_reverse / B_bag",
    )
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["val", "ood_partial_obs"],
        help="Splits used for OOV reporting (default: val ood_partial_obs)",
    )
    args = parser.parse_args(argv)

    print(f"encoded-dir: {args.encoded_dir}")
    print(f"out-dir: {args.out_dir}")
    print(f"share_b_family: {not args.no_share_b_family}")
    print(f"eval_splits: {args.eval_splits}")
    print()

    report = build_and_save_vocabs(
        args.encoded_dir,
        args.out_dir,
        eval_splits=args.eval_splits,
        share_b_family=not args.no_share_b_family,
    )
    _print_report(report)
    print(f"\nWrote vocabs + vocab_report.json under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
