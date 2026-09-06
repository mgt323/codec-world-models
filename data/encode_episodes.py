"""Encode raw State HDF5 splits into per-codec parquet string datasets.

Stores raw codec strings only (no tokenization). E-variants of Codec B use
``derive_transform_seed(episode_seed, timestep, variant_name)``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from obs_codecs.diagnostics import tokenize
from obs_codecs.encode_a import encode_A, parse_A
from obs_codecs.encode_b import decode_B_facts_unordered, encode_B, parse_B
from obs_codecs.encode_c import encode_C, parse_C
from obs_codecs.encode_d import encode_D, parse_D
from obs_codecs.transforms_e import (
    bag_b,
    derive_transform_seed,
    reverse_b,
    shuffle_b,
)
from world.schema import FactRecord, Observation, State, facts_from_observation, observe

ParseFactsFn = Callable[[str], FactRecord]
EncodeRowFn = Callable[[Observation, int, int], str]

# Fixed RNG for validation subsample (reproducible audits).
DEFAULT_VALIDATION_SEED = 20260306
DEFAULT_VALIDATION_SAMPLE_SIZE = 1000

# Output stem suffixes (files: ``{split}_{stem}.parquet``).
VARIANT_A = "A"
VARIANT_B = "B"
VARIANT_C = "C"
VARIANT_D = "D"
VARIANT_B_SHUFFLE = "B_shuffle"
VARIANT_B_REVERSE = "B_reverse"
VARIANT_B_BAG = "B_bag"

ALL_VARIANT_STEMS: tuple[str, ...] = (
    VARIANT_A,
    VARIANT_B,
    VARIANT_C,
    VARIANT_D,
    VARIANT_B_SHUFFLE,
    VARIANT_B_REVERSE,
    VARIANT_B_BAG,
)


@dataclass(frozen=True, slots=True)
class CodecVariant:
    """One encode target written as ``{split}_{stem}.parquet``."""

    stem: str
    encode_row: EncodeRowFn
    parse_facts: ParseFactsFn


@dataclass(frozen=True, slots=True)
class ValidationFailure:
    index: int
    episode_seed: int
    timestep: int
    encoded: str
    expected: FactRecord
    recovered: FactRecord | None
    error: str


@dataclass(frozen=True, slots=True)
class VariantEncodeReport:
    split: str
    variant: str
    path: str
    record_count: int
    validation_sample_size: int
    validation_passed: int
    validation_pass_rate: float
    avg_tokens_per_record: float
    storage_bytes: int
    failures: tuple[ValidationFailure, ...]

    @property
    def ok(self) -> bool:
        return (
            self.validation_sample_size == self.validation_passed
            and len(self.failures) == 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "variant": self.variant,
            "path": self.path,
            "record_count": self.record_count,
            "validation_sample_size": self.validation_sample_size,
            "validation_passed": self.validation_passed,
            "validation_pass_rate": self.validation_pass_rate,
            "avg_tokens_per_record": self.avg_tokens_per_record,
            "storage_bytes": self.storage_bytes,
            "failures": [
                {
                    "index": f.index,
                    "episode_seed": f.episode_seed,
                    "timestep": f.timestep,
                    "encoded": f.encoded,
                    "expected": repr(f.expected),
                    "recovered": None if f.recovered is None else repr(f.recovered),
                    "error": f.error,
                }
                for f in self.failures
            ],
        }


def _encode_a(obs: Observation, _episode_seed: int, _timestep: int) -> str:
    return encode_A(obs)


def _encode_b(obs: Observation, _episode_seed: int, _timestep: int) -> str:
    return encode_B(obs)


def _encode_c(obs: Observation, _episode_seed: int, _timestep: int) -> str:
    return encode_C(obs)


def _encode_d(obs: Observation, _episode_seed: int, _timestep: int) -> str:
    return encode_D(obs)


def _encode_b_shuffle(obs: Observation, episode_seed: int, timestep: int) -> str:
    seed = derive_transform_seed(episode_seed, timestep, "shuffle_b")
    return shuffle_b(encode_B(obs), seed)


def _encode_b_reverse(obs: Observation, _episode_seed: int, _timestep: int) -> str:
    return reverse_b(encode_B(obs))


def _encode_b_bag(obs: Observation, episode_seed: int, timestep: int) -> str:
    seed = derive_transform_seed(episode_seed, timestep, "bag_b")
    return bag_b(encode_B(obs), seed)


def codec_variants() -> tuple[CodecVariant, ...]:
    """Locked A–D codecs plus B E-transforms (independent shuffle/bag seeds)."""
    return (
        CodecVariant(VARIANT_A, _encode_a, parse_A),
        CodecVariant(VARIANT_B, _encode_b, parse_B),
        CodecVariant(VARIANT_C, _encode_c, parse_C),
        CodecVariant(VARIANT_D, _encode_d, parse_D),
        CodecVariant(VARIANT_B_SHUFFLE, _encode_b_shuffle, decode_B_facts_unordered),
        CodecVariant(VARIANT_B_REVERSE, _encode_b_reverse, decode_B_facts_unordered),
        CodecVariant(VARIANT_B_BAG, _encode_b_bag, decode_B_facts_unordered),
    )


def parquet_path(out_dir: Path, split_name: str, variant_stem: str) -> Path:
    return out_dir / f"{split_name}_{variant_stem}.parquet"


def avg_tokens_per_string(texts: Sequence[str]) -> float:
    """Mean provisional token count (``obs_codecs.diagnostics.tokenize``)."""
    if not texts:
        return 0.0
    total = sum(len(tokenize(t)) for t in texts)
    return total / len(texts)


def build_encoded_frame(
    episodes: Sequence[tuple[int, list[State]]],
    variant: CodecVariant,
) -> pd.DataFrame:
    """Encode all timesteps for one variant into a three-column frame."""
    episode_seeds: list[int] = []
    timesteps: list[int] = []
    encoded: list[str] = []
    for episode_seed, states in episodes:
        for timestep, state in enumerate(states):
            obs = observe(state)
            episode_seeds.append(int(episode_seed))
            timesteps.append(int(timestep))
            encoded.append(variant.encode_row(obs, int(episode_seed), int(timestep)))
    return pd.DataFrame(
        {
            "episode_seed": episode_seeds,
            "timestep": timesteps,
            "encoded_string": encoded,
        }
    )


def write_encoded_parquet(frame: pd.DataFrame, path: Path) -> int:
    """Write parquet; return file size in bytes.

    Parquet (not JSONL): columnar compression and vectorized loads scale better
    for multi-million string-sequence rows than line-oriented JSONL, while still
    keeping a simple ``(episode_seed, timestep, encoded_string)`` schema.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow")
    return path.stat().st_size


def _obs_lookup(
    episodes: Sequence[tuple[int, list[State]]],
) -> dict[tuple[int, int], Observation]:
    out: dict[tuple[int, int], Observation] = {}
    for episode_seed, states in episodes:
        for timestep, state in enumerate(states):
            out[(int(episode_seed), int(timestep))] = observe(state)
    return out


def validate_encoded_sample(
    frame: pd.DataFrame,
    *,
    episodes: Sequence[tuple[int, list[State]]],
    parse_facts: ParseFactsFn,
    sample_size: int = DEFAULT_VALIDATION_SAMPLE_SIZE,
    validation_seed: int = DEFAULT_VALIDATION_SEED,
) -> tuple[int, int, tuple[ValidationFailure, ...]]:
    """Parity-style fact recovery on a deterministic random subsample.

    Returns ``(n_sample, n_passed, failures)``. Pass rate must be 100%.
    """
    n = len(frame)
    if n == 0:
        return 0, 0, ()
    k = min(sample_size, n)
    rng = random.Random(validation_seed)
    indices = sorted(rng.sample(range(n), k=k))
    lookup = _obs_lookup(episodes)

    failures: list[ValidationFailure] = []
    n_passed = 0
    for index in indices:
        row = frame.iloc[index]
        episode_seed = int(row["episode_seed"])
        timestep = int(row["timestep"])
        encoded = str(row["encoded_string"])
        obs = lookup[(episode_seed, timestep)]
        expected = facts_from_observation(obs)
        recovered: FactRecord | None = None
        error: str | None = None
        try:
            recovered = parse_facts(encoded)
            if recovered != expected:
                error = "fact_mismatch"
        except Exception as exc:  # noqa: BLE001 — audit must capture any parse failure
            error = f"{type(exc).__name__}: {exc}"

        if error is None:
            n_passed += 1
        else:
            failures.append(
                ValidationFailure(
                    index=index,
                    episode_seed=episode_seed,
                    timestep=timestep,
                    encoded=encoded,
                    expected=expected,
                    recovered=recovered,
                    error=error,
                )
            )
    return k, n_passed, tuple(failures)


class EncodingValidationError(RuntimeError):
    """Raised when a split/variant sample audit is below 100%."""

    def __init__(self, report: VariantEncodeReport) -> None:
        self.report = report
        detail = "\n".join(
            f"  [{f.index}] ep={f.episode_seed} t={f.timestep} "
            f"error={f.error!r} expected={f.expected!r} recovered={f.recovered!r} "
            f"encoded={f.encoded!r}"
            for f in report.failures
        )
        super().__init__(
            f"Encoding validation failed for {report.split}/{report.variant}: "
            f"{report.validation_passed}/{report.validation_sample_size} passed "
            f"({100.0 * report.validation_pass_rate:.1f}%). Failures:\n{detail}"
        )


def encode_split_variant(
    episodes: Sequence[tuple[int, list[State]]],
    *,
    split_name: str,
    variant: CodecVariant,
    out_dir: Path,
    sample_size: int = DEFAULT_VALIDATION_SAMPLE_SIZE,
    validation_seed: int = DEFAULT_VALIDATION_SEED,
) -> VariantEncodeReport:
    """Encode one variant, write parquet, validate sample (must be 100%)."""
    frame = build_encoded_frame(episodes, variant)
    path = parquet_path(out_dir, split_name, variant.stem)
    storage_bytes = write_encoded_parquet(frame, path)
    n_sample, n_passed, failures = validate_encoded_sample(
        frame,
        episodes=episodes,
        parse_facts=variant.parse_facts,
        sample_size=sample_size,
        validation_seed=validation_seed,
    )
    pass_rate = 1.0 if n_sample == 0 else n_passed / n_sample
    report = VariantEncodeReport(
        split=split_name,
        variant=variant.stem,
        path=str(path),
        record_count=len(frame),
        validation_sample_size=n_sample,
        validation_passed=n_passed,
        validation_pass_rate=pass_rate,
        avg_tokens_per_record=avg_tokens_per_string(frame["encoded_string"].tolist()),
        storage_bytes=storage_bytes,
        failures=failures,
    )
    if not report.ok:
        raise EncodingValidationError(report)
    return report


def encode_dataset(
    raw_dir: Path,
    out_dir: Path,
    *,
    split_names: Sequence[str],
    sample_size: int = DEFAULT_VALIDATION_SAMPLE_SIZE,
    validation_seed: int = DEFAULT_VALIDATION_SEED,
    variants: Sequence[CodecVariant] | None = None,
) -> list[VariantEncodeReport]:
    """Encode every listed split × variant; stop on first validation failure."""
    from data.hdf5_episodes import read_episodes_hdf5

    chosen = tuple(variants) if variants is not None else codec_variants()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: list[VariantEncodeReport] = []

    for split_name in split_names:
        h5_path = raw_dir / f"{split_name}.h5"
        if not h5_path.is_file():
            raise FileNotFoundError(f"Missing raw split HDF5: {h5_path}")
        episodes = read_episodes_hdf5(h5_path)
        for variant in chosen:
            report = encode_split_variant(
                episodes,
                split_name=split_name,
                variant=variant,
                out_dir=out_dir,
                sample_size=sample_size,
                validation_seed=validation_seed,
            )
            reports.append(report)

    manifest = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "validation_sample_size": sample_size,
        "validation_seed": validation_seed,
        "storage_format": "parquet",
        "storage_rationale": (
            "Parquet chosen over JSONL for columnar compression and faster "
            "downstream loads at multi-million string-sequence scale; schema is "
            "episode_seed, timestep, encoded_string (raw strings only; no tokens)."
        ),
        "e_transform_seed_policy": (
            "derive_transform_seed(episode_seed, timestep, variant_name) with "
            "variant_name in {shuffle_b, bag_b}; reverse_b needs no seed"
        ),
        "reports": [r.to_dict() for r in reports],
    }
    manifest_path = out_dir / "encoding_report.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return reports
