"""Encode raw State HDF5 datasets into per-codec parquet string files.

Reads splits from ``--raw-dir`` (``{split}.h5``), writes
``{split}_{A|B|C|D|B_shuffle|B_reverse|B_bag}.parquet`` under ``--out-dir``.

Each record: episode_seed, timestep, encoded_string (no tokenization).
Validates a deterministic random sample per file (100% required) before
continuing. See ``data.encode_episodes`` for format rationale (parquet).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.dataset_v0 import active_split_specs  # noqa: E402
from data.encode_episodes import (  # noqa: E402
    DEFAULT_VALIDATION_SAMPLE_SIZE,
    DEFAULT_VALIDATION_SEED,
    EncodingValidationError,
    encode_dataset,
)


def _print_report_table(reports) -> None:
    print(
        f"{'split':<18} {'variant':<12} {'n':>8} {'val_pass':>10} "
        f"{'avg_tok':>8} {'bytes':>12}"
    )
    for r in reports:
        print(
            f"{r.split:<18} {r.variant:<12} {r.record_count:>8} "
            f"{r.validation_passed}/{r.validation_sample_size:<6} "
            f"{r.avg_tokens_per_record:>8.2f} {r.storage_bytes:>12}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Encode HDF5 State splits through codecs A–D and B E-transforms "
            "into parquet string datasets (no tokenization)."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_ROOT / "artifacts" / "dataset_v0_dry_run",
        help="Directory containing {split}.h5 and seed_checksums.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "artifacts" / "encoded",
        help="Output directory for parquet files + encoding_report.json",
    )
    parser.add_argument(
        "--validation-sample-size",
        type=int,
        default=DEFAULT_VALIDATION_SAMPLE_SIZE,
        help="Random records audited per split/variant (default 1000)",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=DEFAULT_VALIDATION_SEED,
        help="RNG seed for validation subsample (reproducible)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Subset of splits (default: all active splits with HDF5 present)",
    )
    args = parser.parse_args(argv)

    if args.splits is not None:
        split_names = list(args.splits)
    else:
        split_names = [s.name for s in active_split_specs()]

    print(
        "Storage: parquet (columnar compression + faster loads than JSONL "
        "at string-sequence scale; raw encoded strings only, no tokens)."
    )
    print(f"raw-dir: {args.raw_dir}")
    print(f"out-dir: {args.out_dir}")
    print(f"splits: {split_names}")
    print(
        f"validation: sample_size={args.validation_sample_size} "
        f"seed={args.validation_seed}"
    )
    print()

    try:
        reports = encode_dataset(
            args.raw_dir,
            args.out_dir,
            split_names=split_names,
            sample_size=args.validation_sample_size,
            validation_seed=args.validation_seed,
        )
    except EncodingValidationError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        failed = exc.report
        print(
            f"FLAG: validation below 100% for {failed.split}/{failed.variant}",
            file=sys.stderr,
        )
        return 1

    _print_report_table(reports)
    print()
    print(f"Wrote report: {args.out_dir / 'encoding_report.json'}")
    if any(not r.ok for r in reports):
        print("FLAG: one or more variants failed validation", file=sys.stderr)
        return 1
    print("All split/variant sample validations: 100%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
