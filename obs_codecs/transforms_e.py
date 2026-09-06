"""E-variants: structure-destroyed transforms of Codec B strings.

These are NOT Observation→string codecs. Per EXPERIMENT_PLAN.md §3
(Structure-destroyed controls on B), each transform takes ``encode_B(obs)``
output and destroys event *organization* while preserving recoverable
facts under ``observation_from_B_unordered``.

- ``shuffle_b`` — permute arrow-separated events (seeded RNG; no global random)
- ``reverse_b`` — reverse arrow-separated event order
- ``bag_b`` — drop arrows; rejoin events with ``BAG_SEP`` (``" | "``) in a
  seeded shuffle order so no directional / sequential cue remains

``BAG_SEP`` is defined on ``encode_b`` (same dialect the unordered parser
accepts). Optional ``C-shuffle`` is out of scope here.

Offline / dataset encoding must seed ``shuffle_b`` / ``bag_b`` via
``derive_transform_seed(episode_seed, timestep, variant_name)`` so the
Fisher–Yates pattern varies per timestep and per variant (never a fixed
per-episode or global seed). ``reverse_b`` needs no seed.
"""

from __future__ import annotations

import hashlib
import random

from obs_codecs.encode_b import ARROW_SEP, BAG_SEP, split_b_events

# Matches ``random.Random`` / uint64 PCG64 seed width (8-byte little-endian).
_TRANSFORM_SEED_MASK = (1 << 64) - 1


def derive_transform_seed(episode_seed: int, timestep: int, variant_name: str) -> int:
    """Stable, process-independent seed for E-transforms.

    Never uses Python's builtin ``hash()`` (unstable across runs/interpreters).

    Payload is the UTF-8 encoding of
    ``f"{episode_seed}:{timestep}:{variant_name}"``. Digest is blake2b
    truncated to 8 bytes, interpreted little-endian and masked to
    ``[0, 2**64)`` so it seeds ``random.Random`` (used by ``shuffle_b`` /
    ``bag_b``) and would also be valid for ``numpy.random.Generator`` /
    PCG64 if that family is adopted later.

    ``variant_name`` is part of the hash input so ``"shuffle_b"`` and
    ``"bag_b"`` get independent permutations at the same
    ``(episode_seed, timestep)``.
    """
    payload = f"{episode_seed}:{timestep}:{variant_name}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & _TRANSFORM_SEED_MASK


def shuffle_b(text: str, seed: int) -> str:
    """Deterministically permute Codec B events; rejoin with ``ARROW_SEP``."""
    events = split_b_events(text)
    if ARROW_SEP not in text and BAG_SEP in text:
        raise ValueError(
            "shuffle_b expects an arrow-separated Codec B string; "
            "got bag-separated input"
        )
    shuffled = list(events)
    random.Random(seed).shuffle(shuffled)
    return ARROW_SEP.join(shuffled)


def reverse_b(text: str) -> str:
    """Reverse Codec B event order; rejoin with ``ARROW_SEP``."""
    events = split_b_events(text)
    if ARROW_SEP not in text and BAG_SEP in text:
        raise ValueError(
            "reverse_b expects an arrow-separated Codec B string; "
            "got bag-separated input"
        )
    return ARROW_SEP.join(reversed(events))


def bag_b(text: str, seed: int) -> str:
    """Destroy order markers: shuffle events and join with neutral ``BAG_SEP``.

    Separator is ``" | "`` (see ``BAG_SEP``) — not a space, because meta
    events contain spaces and must remain re-splitable. Order of events is
    a seeded permutation (same RNG discipline as ``shuffle_b``).
    """
    events = split_b_events(text)
    shuffled = list(events)
    random.Random(seed).shuffle(shuffled)
    return BAG_SEP.join(shuffled)
