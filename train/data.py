"""Episode token loading for stage-1 training."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from obs_codecs.vocab import VocabArtifact, encode_string_ids, load_vocab


def file_sha256(path: Path | str, *, chunk_bytes: int = 1 << 20) -> str | None:
    """SHA-256 of a dataset file, or ``None`` if it does not exist."""
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pool_fingerprint(pool: Sequence[Sequence[int]]) -> str:
    """Order-sensitive digest of the tokenized pool actually trained on.

    Complements the source-file checksum: it also covers the vocab mapping and
    any pool truncation, so a run injected with a cached pool is still pinned
    to the exact token content it consumed. Token ids are serialized as
    explicit little-endian int32 so the digest is machine-independent.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(len(pool).to_bytes(8, "little"))
    for episode in pool:
        # Length prefix keeps episode boundaries in the digest, so two
        # different segmentations of the same flat stream cannot collide.
        digest.update(len(episode).to_bytes(8, "little"))
        digest.update(np.asarray(episode, dtype="<i4").tobytes())
    return digest.hexdigest()


def vocab_path_for_codec(codec: str, vocab_dir: Path | str = "artifacts/vocabs") -> Path:
    """B-family variants share ``vocab_B.json``."""
    group = "B" if codec.startswith("B") else codec
    return Path(vocab_dir) / f"vocab_{group}.json"


def load_codec_vocab(codec: str, vocab_dir: Path | str = "artifacts/vocabs") -> VocabArtifact:
    return load_vocab(vocab_path_for_codec(codec, vocab_dir))


def episodes_from_parquet(
    parquet_path: Path | str,
    vocab: VocabArtifact,
    *,
    max_episodes: int | None = None,
) -> list[list[int]]:
    """Group timestep rows into episode token-id sequences (no bos/eos)."""
    df = pd.read_parquet(parquet_path)
    required = {"episode_seed", "timestep", "encoded_string"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"parquet missing columns: {sorted(missing)}")

    episodes: list[list[int]] = []
    for _seed, group in df.sort_values(["episode_seed", "timestep"]).groupby(
        "episode_seed", sort=True
    ):
        ids: list[int] = []
        for text in group["encoded_string"].tolist():
            ids.extend(encode_string_ids(str(text), vocab.token_to_id))
        episodes.append(ids)
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    return episodes


def synthetic_episodes(
    *,
    n_episodes: int,
    tokens_per_episode: int,
    vocab_size_native: int,
    seed: int = 0,
) -> list[list[int]]:
    """Tiny deterministic episodes for smoke tests (ids in content range)."""
    import random

    rng = random.Random(seed)
    # Content ids start after specials 0..3.
    lo, hi = 4, max(5, vocab_size_native)
    out: list[list[int]] = []
    for _ in range(n_episodes):
        out.append([rng.randrange(lo, hi) for _ in range(tokens_per_episode)])
    return out


def present_episode_order(
    n_pool: int,
    n_presentations: int,
    shuffle_seed: int,
) -> list[int]:
    """Reshuffle-per-epoch indices covering exactly ``n_presentations``.

    Each full pass over the pool uses ``shuffle_seed + epoch`` so order is
    reproducible and differs across epochs.
    """
    import random

    if n_pool < 1:
        raise ValueError("n_pool must be positive")
    if n_presentations < 1:
        raise ValueError("n_presentations must be positive")

    order: list[int] = []
    epoch = 0
    while len(order) < n_presentations:
        rng = random.Random(shuffle_seed + epoch)
        perm = list(range(n_pool))
        rng.shuffle(perm)
        order.extend(perm)
        epoch += 1
    return order[:n_presentations]


def select_episodes(
    pool: Sequence[Sequence[int]],
    presentation_indices: Sequence[int],
) -> list[list[int]]:
    return [list(pool[i]) for i in presentation_indices]
