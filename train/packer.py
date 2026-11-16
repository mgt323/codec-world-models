"""Document-masked episode packer (Decision 1 / COMPUTE_MATCH_BUDGETS §4).

Concatenates episode token streams with ``<bos>``/``<eos>`` at boundaries,
emits parallel ``segment_ids``, and chunks to exactly ``L`` (pad the tail).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch

from model.architecture import CONTEXT_LENGTH_L, PAD_SEGMENT_ID
from obs_codecs.vocab import BOS_ID, EOS_ID, PAD_ID


@dataclass(frozen=True, slots=True)
class PackedWindow:
    """One fixed-length training window."""

    input_ids: torch.Tensor  # (L,) long
    segment_ids: torch.Tensor  # (L,) long; PAD_SEGMENT_ID on pads
    n_content_tokens: int


def iter_episode_stream(
    episodes: Sequence[Sequence[int]],
    *,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
) -> tuple[list[int], list[int]]:
    """Flatten episodes into a token stream + parallel segment ids.

    Each episode becomes ``[bos] + tokens + [eos]`` under one segment id;
    segment id increments at every episode boundary.
    """
    tokens: list[int] = []
    segments: list[int] = []
    for seg_id, episode in enumerate(episodes):
        piece = [bos_id, *list(episode), eos_id]
        tokens.extend(piece)
        segments.extend([seg_id] * len(piece))
    return tokens, segments


def pack_stream_to_windows(
    tokens: Sequence[int],
    segment_ids: Sequence[int],
    *,
    context_length: int = CONTEXT_LENGTH_L,
    pad_id: int = PAD_ID,
    pad_segment_id: int = PAD_SEGMENT_ID,
) -> list[PackedWindow]:
    """Chunk a flat stream into windows of exactly ``context_length``."""
    if len(tokens) != len(segment_ids):
        raise ValueError("tokens and segment_ids length mismatch")
    if context_length < 1:
        raise ValueError("context_length must be positive")

    windows: list[PackedWindow] = []
    n = len(tokens)
    if n == 0:
        return windows

    offset = 0
    while offset < n:
        end = min(offset + context_length, n)
        chunk_tok = list(tokens[offset:end])
        chunk_seg = list(segment_ids[offset:end])
        n_content = len(chunk_tok)
        if n_content < context_length:
            pad_n = context_length - n_content
            chunk_tok.extend([pad_id] * pad_n)
            chunk_seg.extend([pad_segment_id] * pad_n)
        windows.append(
            PackedWindow(
                input_ids=torch.tensor(chunk_tok, dtype=torch.long),
                segment_ids=torch.tensor(chunk_seg, dtype=torch.long),
                n_content_tokens=n_content,
            )
        )
        offset = end
    return windows


def pack_episodes(
    episodes: Sequence[Sequence[int]],
    *,
    context_length: int = CONTEXT_LENGTH_L,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    pad_id: int = PAD_ID,
) -> list[PackedWindow]:
    """Pack a list of episode token-id sequences into ``L``-windows."""
    tokens, segments = iter_episode_stream(
        episodes, bos_id=bos_id, eos_id=eos_id
    )
    return pack_stream_to_windows(
        tokens,
        segments,
        context_length=context_length,
        pad_id=pad_id,
    )


def iter_packed_batches(
    windows: Sequence[PackedWindow],
    *,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(input_ids, segment_ids)`` batches of shape ``(B, L)``."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for i in range(0, len(windows), batch_size):
        chunk = windows[i : i + batch_size]
        ids = torch.stack([w.input_ids for w in chunk], dim=0)
        segs = torch.stack([w.segment_ids for w in chunk], dim=0)
        yield ids, segs


def count_stream_tokens(episodes: Iterable[Sequence[int]]) -> int:
    """Content tokens plus one ``<bos>`` and one ``<eos>`` per episode."""
    return sum(len(episode) + 2 for episode in episodes)


def iter_packed_windows(
    episodes: Iterable[Sequence[int]],
    *,
    context_length: int = CONTEXT_LENGTH_L,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    pad_id: int = PAD_ID,
    pad_segment_id: int = PAD_SEGMENT_ID,
    pin_memory: bool = False,
) -> Iterator[PackedWindow]:
    """Yield ``L``-windows without materializing the packed stream.

    Segment ids, boundary markers, and the padded tail match ``pack_episodes``.
    The carry buffer holds at most one incomplete window plus the current episode.
    """
    if context_length < 1:
        raise ValueError("context_length must be positive")

    tokens: list[int] = []
    segments: list[int] = []
    for seg_id, episode in enumerate(episodes):
        tokens.append(bos_id)
        segments.append(seg_id)
        tokens.extend(int(tok) for tok in episode)
        segments.extend([seg_id] * len(episode))
        tokens.append(eos_id)
        segments.append(seg_id)
        while len(tokens) >= context_length:
            yield _make_window(
                tokens[:context_length],
                segments[:context_length],
                n_content=context_length,
                context_length=context_length,
                pad_id=pad_id,
                pad_segment_id=pad_segment_id,
                pin_memory=pin_memory,
            )
            del tokens[:context_length]
            del segments[:context_length]
    if tokens:
        n_content = len(tokens)
        yield _make_window(
            tokens,
            segments,
            n_content=n_content,
            context_length=context_length,
            pad_id=pad_id,
            pad_segment_id=pad_segment_id,
            pin_memory=pin_memory,
        )


def iter_packed_batches_streaming(
    episodes: Iterable[Sequence[int]],
    *,
    batch_size: int,
    context_length: int = CONTEXT_LENGTH_L,
    pin_memory: bool = False,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(B, L)`` batches from :func:`iter_packed_windows`.

    The final batch may be shorter than ``batch_size`` (no ``drop_last``).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ids_buf: list[torch.Tensor] = []
    seg_buf: list[torch.Tensor] = []
    for window in iter_packed_windows(
        episodes,
        context_length=context_length,
        pin_memory=pin_memory,
    ):
        ids_buf.append(window.input_ids)
        seg_buf.append(window.segment_ids)
        if len(ids_buf) == batch_size:
            yield _stack_batch(ids_buf, seg_buf, pin_memory=pin_memory)
            ids_buf = []
            seg_buf = []
    if ids_buf:
        yield _stack_batch(ids_buf, seg_buf, pin_memory=pin_memory)


def _make_window(
    tokens: Sequence[int],
    segments: Sequence[int],
    *,
    n_content: int,
    context_length: int,
    pad_id: int,
    pad_segment_id: int,
    pin_memory: bool,
) -> PackedWindow:
    chunk_tok = list(tokens)
    chunk_seg = list(segments)
    if n_content < context_length:
        pad_n = context_length - n_content
        chunk_tok.extend([pad_id] * pad_n)
        chunk_seg.extend([pad_segment_id] * pad_n)
    return PackedWindow(
        input_ids=_long_tensor(chunk_tok, pin_memory=pin_memory),
        segment_ids=_long_tensor(chunk_seg, pin_memory=pin_memory),
        n_content_tokens=n_content,
    )


def _long_tensor(values: Sequence[int], *, pin_memory: bool) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.long)
    if pin_memory:
        return tensor.pin_memory()
    return tensor


def _stack_batch(
    ids_buf: list[torch.Tensor],
    seg_buf: list[torch.Tensor],
    *,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.stack(ids_buf, dim=0)
    segs = torch.stack(seg_buf, dim=0)
    if pin_memory and not ids.is_pinned():
        ids = ids.pin_memory()
        segs = segs.pin_memory()
    return ids, segs
