"""Train-split vocabulary construction for provisional ``regex_v4`` tokenization.

Builds token→id maps from encoded parquet corpora (``encoded_string`` column).
B E-transform variants share one B-family vocab (token-preserving permutations;
``bag_b`` swaps ``→`` for ``|``, so the shared alphabet includes both glyphs).

Under ``regex_v4``, number spans are split into digit / ``-`` / ``.`` character
tokens, so vocab size stays bounded and train-only vocabs typically cover
val/OOD with ~0% OOV (tokenizer-only; encoded strings unchanged).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from obs_codecs.diagnostics import (
    TOKENIZATION_SCHEME,
    TOKENIZATION_SCHEME_VERSION,
    tokenize,
)

TokenizerFn = Callable[[str], list[str]]

# Reserved at the front of every vocab. Indices are stable across codecs.
SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")
PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = SPECIAL_TOKENS
PAD_ID, BOS_ID, EOS_ID, UNK_ID = range(len(SPECIAL_TOKENS))

# Encode-stem names (must match ``data.encode_episodes`` / parquet filenames).
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

# Shuffle/reverse preserve B's token multiset; bag swaps → for |. One shared
# content alphabet avoids four drifted copies of the same float set.
B_FAMILY_VARIANTS: tuple[str, ...] = (
    VARIANT_B,
    VARIANT_B_SHUFFLE,
    VARIANT_B_REVERSE,
    VARIANT_B_BAG,
)

# Logical vocab keys written under artifacts (B-family shares one content set).
VOCAB_GROUP_A = "A"
VOCAB_GROUP_B = "B"
VOCAB_GROUP_C = "C"
VOCAB_GROUP_D = "D"

# regex_v4 numeric pieces (and overlapping punct "." from non-number matches).
_NUMERIC_PIECE_CHARS = frozenset("0123456789.-")
_ENCODED_COLUMN = "encoded_string"


@dataclass(frozen=True, slots=True)
class VocabArtifact:
    """Persisted vocab: forward map, reverse map, and build metadata."""

    token_to_id: dict[str, int]
    id_to_token: list[str]
    metadata: dict[str, Any]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VocabArtifact:
        token_to_id = {str(k): int(v) for k, v in payload["token_to_id"].items()}
        id_to_token = [str(t) for t in payload["id_to_token"]]
        metadata = dict(payload.get("metadata", {}))
        if len(token_to_id) != len(id_to_token):
            raise ValueError("token_to_id and id_to_token length mismatch")
        for idx, tok in enumerate(id_to_token):
            if token_to_id.get(tok) != idx:
                raise ValueError(f"Inconsistent maps at id={idx} token={tok!r}")
        return cls(token_to_id=token_to_id, id_to_token=id_to_token, metadata=metadata)


@dataclass(frozen=True, slots=True)
class OovStats:
    """Out-of-vocabulary rates for one eval split against a train vocab."""

    split: str
    variant: str
    n_tokens: int
    n_oov_tokens: int
    oov_token_rate: float
    n_types: int
    n_oov_types: int
    oov_type_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VocabReport:
    """Per-group sizes plus OOV sanity checks on val / ood_partial_obs."""

    vocab_sizes: dict[str, int]
    numeric_token_counts: dict[str, int]
    non_numeric_token_counts: dict[str, int]
    oov: tuple[OovStats, ...]
    shared_b_family: bool
    tokenization_scheme: str
    special_tokens: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_sizes": self.vocab_sizes,
            "numeric_token_counts": self.numeric_token_counts,
            "non_numeric_token_counts": self.non_numeric_token_counts,
            "oov": [s.to_dict() for s in self.oov],
            "shared_b_family": self.shared_b_family,
            "tokenization_scheme": self.tokenization_scheme,
            "special_tokens": list(self.special_tokens),
            "notes": list(self.notes),
        }


def is_number_token(token: str) -> bool:
    """True iff ``token`` is a regex_v4 numeric character piece (digit, '-', '.')."""
    return len(token) == 1 and token in _NUMERIC_PIECE_CHARS


def build_vocab(
    parquet_paths: list[Path],
    tokenizer: TokenizerFn = tokenize,
) -> dict[str, int]:
    """Build token→id from TRAIN parquet paths only (specials reserved first).

    Content tokens are sorted for determinism. Unknown eval tokens map via
    ``<unk>`` at lookup time (not inserted here).
    """
    if not parquet_paths:
        raise ValueError("parquet_paths must be non-empty")

    content: set[str] = set()
    for path in parquet_paths:
        content.update(_iter_tokens_from_parquet(path, tokenizer=tokenizer))

    special_set = set(SPECIAL_TOKENS)
    content -= special_set
    ordered = list(SPECIAL_TOKENS) + sorted(content)
    return {tok: idx for idx, tok in enumerate(ordered)}


def lookup_id(token: str, token_to_id: Mapping[str, int], *, unk_id: int = UNK_ID) -> int:
    """Map a surface token to an id, falling back to ``<unk>``."""
    return token_to_id.get(token, unk_id)


def encode_string_ids(
    text: str,
    token_to_id: Mapping[str, int],
    *,
    tokenizer: TokenizerFn = tokenize,
    unk_id: int = UNK_ID,
) -> list[int]:
    """Tokenize ``text`` and map each token through ``token_to_id`` / ``<unk>``."""
    return [lookup_id(tok, token_to_id, unk_id=unk_id) for tok in tokenizer(text)]


def artifact_from_token_to_id(
    token_to_id: Mapping[str, int],
    *,
    metadata: Mapping[str, Any],
) -> VocabArtifact:
    """Validate and wrap a token→id map with reverse list + metadata."""
    n = len(token_to_id)
    id_to_token = [""] * n
    for tok, idx in token_to_id.items():
        if not isinstance(idx, int) or idx < 0 or idx >= n:
            raise ValueError(f"Invalid id {idx!r} for token {tok!r}")
        if id_to_token[idx]:
            raise ValueError(f"Duplicate id {idx}")
        id_to_token[idx] = tok
    if any(t == "" for t in id_to_token):
        raise ValueError("token_to_id ids are not a contiguous range")
    for i, name in enumerate(SPECIAL_TOKENS):
        if id_to_token[i] != name:
            raise ValueError(
                f"Special token slot {i} must be {name!r}, got {id_to_token[i]!r}"
            )
    return VocabArtifact(
        token_to_id=dict(token_to_id),
        id_to_token=id_to_token,
        metadata=dict(metadata),
    )


def save_vocab(artifact: VocabArtifact, path: Path | str) -> Path:
    """Write vocab JSON (token→id, reverse list, metadata)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(artifact.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def load_vocab(path: Path | str) -> VocabArtifact:
    """Load a vocab JSON written by ``save_vocab``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return VocabArtifact.from_dict(payload)


def compute_oov_stats(
    parquet_path: Path,
    token_to_id: Mapping[str, int],
    *,
    split: str,
    variant: str,
    tokenizer: TokenizerFn = tokenize,
) -> OovStats:
    """Fraction of tokens/types in ``parquet_path`` absent from train vocab keys.

    Specials are ignored as content; comparison uses content keys only
    (``token_to_id`` minus reserved specials). Occurrence OOV counts tokens
    that would map to ``<unk>``.
    """
    vocab_content = set(token_to_id) - set(SPECIAL_TOKENS)
    n_tokens = 0
    n_oov_tokens = 0
    types: set[str] = set()
    oov_types: set[str] = set()

    for tok in _iter_tokens_from_parquet(parquet_path, tokenizer=tokenizer):
        n_tokens += 1
        types.add(tok)
        if tok not in vocab_content and tok not in SPECIAL_TOKENS:
            n_oov_tokens += 1
            oov_types.add(tok)

    n_types = len(types)
    n_oov_types = len(oov_types)
    return OovStats(
        split=split,
        variant=variant,
        n_tokens=n_tokens,
        n_oov_tokens=n_oov_tokens,
        oov_token_rate=(n_oov_tokens / n_tokens) if n_tokens else 0.0,
        n_types=n_types,
        n_oov_types=n_oov_types,
        oov_type_rate=(n_oov_types / n_types) if n_types else 0.0,
    )


def build_and_save_vocabs(
    encoded_dir: Path,
    out_dir: Path,
    *,
    train_split: str = "train",
    eval_splits: Sequence[str] = ("val", "ood_partial_obs"),
    tokenizer: TokenizerFn = tokenize,
    share_b_family: bool = True,
) -> VocabReport:
    """Build train-only vocabs, save JSON per variant stem, return size/OOV report.

    When ``share_b_family`` is True (recommended), content is collected from
    ``train_B`` and ``train_B_bag`` (to include both ``→`` and ``|``), then the
    same map is written for B / B_shuffle / B_reverse / B_bag.
    """
    encoded_dir = Path(encoded_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_commit = _try_git_commit(encoded_dir)

    notes = _special_token_notes() + _numeric_handling_notes()
    if share_b_family:
        notes = notes + (
            "B_shuffle / B_reverse / B_bag share B-family vocab "
            "(built from train_B union train_B_bag so both arrow and pipe glyphs "
            "are present).",
        )

    group_specs = _vocab_group_specs(share_b_family=share_b_family)
    artifacts_by_group: dict[str, VocabArtifact] = {}
    sizes: dict[str, int] = {}
    numeric_counts: dict[str, int] = {}
    non_numeric_counts: dict[str, int] = {}

    for group_name, source_stems, write_stems in group_specs:
        paths = [
            _parquet_path(encoded_dir, train_split, stem) for stem in source_stems
        ]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing train parquet for vocab: {path}")
        token_to_id = build_vocab(paths, tokenizer=tokenizer)
        n_numeric = sum(1 for t in token_to_id if is_number_token(t))
        n_non_numeric = len(token_to_id) - n_numeric
        meta_base = {
            "vocab_group": group_name,
            "source_split": train_split,
            "source_variants": list(source_stems),
            "shared_for_variants": list(write_stems),
            "tokenization_scheme": TOKENIZATION_SCHEME,
            "tokenizer_scheme_version": TOKENIZATION_SCHEME_VERSION,
            "special_tokens": list(SPECIAL_TOKENS),
            "special_token_ids": {
                PAD_TOKEN: PAD_ID,
                BOS_TOKEN: BOS_ID,
                EOS_TOKEN: EOS_ID,
                UNK_TOKEN: UNK_ID,
            },
            "vocab_size": len(token_to_id),
            "numeric_token_count": n_numeric,
            "non_numeric_token_count": n_non_numeric,
            "build_utc": build_utc,
            "git_commit": git_commit,
            "share_b_family": share_b_family and group_name == VOCAB_GROUP_B,
        }
        artifact = artifact_from_token_to_id(token_to_id, metadata=meta_base)
        artifacts_by_group[group_name] = artifact
        sizes[group_name] = artifact.vocab_size
        numeric_counts[group_name] = n_numeric
        non_numeric_counts[group_name] = n_non_numeric

        for stem in write_stems:
            stem_meta = dict(meta_base)
            stem_meta["variant_stem"] = stem
            stem_meta["shared_from_group"] = group_name
            stem_artifact = VocabArtifact(
                token_to_id=artifact.token_to_id,
                id_to_token=artifact.id_to_token,
                metadata=stem_meta,
            )
            save_vocab(stem_artifact, out_dir / f"vocab_{stem}.json")

    # Also key sizes by variant for the B aliases in the report.
    for stem in B_FAMILY_VARIANTS:
        if share_b_family and VOCAB_GROUP_B in sizes:
            sizes.setdefault(stem, sizes[VOCAB_GROUP_B])
            numeric_counts.setdefault(stem, numeric_counts[VOCAB_GROUP_B])
            non_numeric_counts.setdefault(stem, non_numeric_counts[VOCAB_GROUP_B])

    oov_rows: list[OovStats] = []
    for split in eval_splits:
        for stem in ALL_VARIANT_STEMS:
            path = _parquet_path(encoded_dir, split, stem)
            if not path.is_file():
                continue
            group = _group_for_stem(stem, share_b_family=share_b_family)
            artifact = artifacts_by_group[group]
            oov_rows.append(
                compute_oov_stats(
                    path,
                    artifact.token_to_id,
                    split=split,
                    variant=stem,
                    tokenizer=tokenizer,
                )
            )

    report = VocabReport(
        vocab_sizes=sizes,
        numeric_token_counts=numeric_counts,
        non_numeric_token_counts=non_numeric_counts,
        oov=tuple(oov_rows),
        shared_b_family=share_b_family,
        tokenization_scheme=TOKENIZATION_SCHEME,
        special_tokens=SPECIAL_TOKENS,
        notes=notes,
    )
    report_path = out_dir / "vocab_report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def _vocab_group_specs(
    *,
    share_b_family: bool,
) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """Return (group_name, source_stems, write_stems) triples."""
    specs: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        (VOCAB_GROUP_A, (VARIANT_A,), (VARIANT_A,)),
    ]
    if share_b_family:
        specs.append(
            (
                VOCAB_GROUP_B,
                (VARIANT_B, VARIANT_B_BAG),
                B_FAMILY_VARIANTS,
            )
        )
    else:
        for stem in B_FAMILY_VARIANTS:
            specs.append((stem, (stem,), (stem,)))
    specs.extend(
        [
            (VOCAB_GROUP_C, (VARIANT_C,), (VARIANT_C,)),
            (VOCAB_GROUP_D, (VARIANT_D,), (VARIANT_D,)),
        ]
    )
    return specs


def _group_for_stem(stem: str, *, share_b_family: bool) -> str:
    if share_b_family and stem in B_FAMILY_VARIANTS:
        return VOCAB_GROUP_B
    if stem == VARIANT_A:
        return VOCAB_GROUP_A
    if stem == VARIANT_C:
        return VOCAB_GROUP_C
    if stem == VARIANT_D:
        return VOCAB_GROUP_D
    if stem in B_FAMILY_VARIANTS:
        return stem
    raise ValueError(f"Unknown variant stem: {stem!r}")


def _parquet_path(encoded_dir: Path, split_name: str, variant_stem: str) -> Path:
    return encoded_dir / f"{split_name}_{variant_stem}.parquet"


def _iter_tokens_from_parquet(
    path: Path,
    *,
    tokenizer: TokenizerFn,
) -> Iterable[str]:
    frame = pd.read_parquet(path, columns=[_ENCODED_COLUMN])
    if _ENCODED_COLUMN not in frame.columns:
        raise ValueError(f"{path} missing column {_ENCODED_COLUMN!r}")
    for text in frame[_ENCODED_COLUMN].astype(str):
        yield from tokenizer(text)


def _try_git_commit(cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _special_token_notes() -> tuple[str, ...]:
    return (
        "<unk> is required: val/OOD may emit float strings absent from train.",
        "<pad> is required for batching: Codec B token length varies per timestep.",
        "<bos>/<eos> are reserved for episode/sequence packing; per-timestep "
        "fixed-length codecs do not need them inside a single observation string.",
    )


def _numeric_handling_notes() -> tuple[str, ...]:
    return (
        "regex_v4 splits each matched number span into digit/sign/dot character "
        "tokens (tokenizer-only; encode_* strings unchanged). Rejected: rounding "
        "_format_value (FactRecord bin-flip); bin-index surface (drops continuous "
        "values needed for eval task #1 MSE).",
    )
