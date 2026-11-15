"""Load a scoring checkpoint. Non-final files are rejected (contract §2.6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model.architecture import DecoderOnlyLM, build_model


def load_scoring_checkpoint(
    path: Path | str,
    device: torch.device,
) -> tuple[DecoderOnlyLM, dict[str, Any]]:
    """Load ``checkpoints/final.pt`` and freeze it.

    ``scoring`` must be true and the filename must be ``final.pt``.
    """
    ckpt = Path(path)
    if ckpt.name != "final.pt":
        raise ValueError(
            f"refusing non-final checkpoint {ckpt.name}; scoring reads final.pt only"
        )
    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"{ckpt} is not a training checkpoint payload")
    if payload.get("scoring") is not True:
        raise ValueError(
            f"{ckpt} is not a scoring checkpoint (scoring={payload.get('scoring')!r})"
        )
    model = build_model()
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.float()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload
