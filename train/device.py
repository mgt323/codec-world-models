"""Device selection for stage-1 training.

ROCm PyTorch exposes the GPU through the CUDA API (``torch.device("cuda")``,
``torch.version.hip``). Weight initialization and episode shuffling stay on
CPU generators; this module only chooses where the module and batches run.

Precision is locked to fp32 for every codec. Mixed precision would be a
methodological change (invariant I6) and is not selected here.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import torch

DeviceSpec = Literal["auto", "rocm", "cuda", "cpu"]
PRECISION = "fp32"

_GPU_SPECS = frozenset({"rocm", "cuda"})


def gpu_available() -> bool:
    """True when the HIP/CUDA runtime can see a device."""
    return bool(torch.cuda.is_available())


def resolve_device(spec: DeviceSpec, *, require_gpu: bool = False) -> torch.device:
    """Resolve a CLI device spec to a ``torch.device``.

    ``rocm`` and ``cuda`` both mean the PyTorch accelerator device. On this
    project that device is ROCm/HIP. ``auto`` prefers the accelerator and
    otherwise stays on CPU. ``require_gpu`` rejects a CPU result.
    """
    if spec not in ("auto", "rocm", "cuda", "cpu"):
        raise ValueError(
            f"Unknown device spec {spec!r}; expected auto, rocm, cuda, or cpu"
        )
    if spec in _GPU_SPECS and not gpu_available():
        hip = getattr(torch.version, "hip", None)
        raise RuntimeError(
            f"device spec {spec!r} requires a ROCm/HIP GPU, but "
            f"torch.cuda.is_available() is false (torch={torch.__version__}, "
            f"hip={hip!r}). Install the ROCm wheel and confirm /dev/kfd is "
            "readable. Set HSA_OVERRIDE_GFX_VERSION=12.0.1 only when the "
            "wheel's arch list does not include gfx1201."
        )
    if spec == "cpu":
        device = torch.device("cpu")
    elif spec in _GPU_SPECS:
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if gpu_available() else "cpu")
    if require_gpu and device.type != "cuda":
        raise RuntimeError(
            "require_gpu is set but the resolved device is cpu "
            f"(spec={spec!r})"
        )
    return device


def _gcn_arch_name(index: int) -> str | None:
    try:
        props = torch.cuda.get_device_properties(index)
    except Exception:
        return None
    for attr in ("gcnArchName", "gcn_arch_name"):
        value = getattr(props, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def hardware_metadata(device: torch.device) -> dict[str, Any]:
    """Provenance recorded in reproducibility.json. Not part of RunConfig."""
    hip = getattr(torch.version, "hip", None)
    info: dict[str, Any] = {
        "device_type": device.type,
        "device_name": None,
        "gcn_arch_name": None,
        "torch_version": torch.__version__,
        "hip_version": hip,
        "precision": PRECISION,
        "hsa_override_gfx_version": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
    }
    if device.type == "cuda" and gpu_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(index)
        info["gcn_arch_name"] = _gcn_arch_name(index)
    return info
