"""Deterministic weight initialization (Decision 3).

Trunk (non-embedding, non-LM-head) shares ``model_init_seed`` across codecs.
Token embedding + untied LM head use
``blake2b_u64(f"{model_init_seed}:{codec_name}")`` — same recipe as
``derive_transform_seed``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.architecture import DecoderOnlyLM
from obs_codecs.transforms_e import blake2b_u64


def derive_codec_init_seed(model_init_seed: int, codec_name: str) -> int:
    """Codec-tagged seed for embedding table and output head."""
    return blake2b_u64(f"{model_init_seed}:{codec_name}")


def _seeded_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    # torch Generator seeds are int64; mask blake2b u64 into signed-safe range.
    gen.manual_seed(int(seed % (2**63 - 1)))
    return gen


def _init_linear_or_embedding(
    module: nn.Module,
    generator: torch.Generator,
) -> None:
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=generator)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
        return
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=generator)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def initialize_model(
    model: DecoderOnlyLM,
    *,
    model_init_seed: int,
    codec_name: str,
) -> DecoderOnlyLM:
    """Initialize ``model`` in-place per Decision 3; returns the same module.

    Call before moving the module to CUDA so RNG stays on CPU.
    """
    trunk_gen = _seeded_generator(model_init_seed)
    emb_seed = derive_codec_init_seed(model_init_seed, codec_name)
    emb_gen = _seeded_generator(emb_seed)

    for name, module in model.named_modules():
        if name == "tok_emb" or name.startswith("tok_emb."):
            continue
        if name == "lm_head" or name.startswith("lm_head."):
            continue
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            _init_linear_or_embedding(module, trunk_gen)

    _init_linear_or_embedding(model.tok_emb, emb_gen)
    _init_linear_or_embedding(model.lm_head, emb_gen)
    return model


def trunk_state_dict(model: DecoderOnlyLM) -> dict[str, torch.Tensor]:
    """CPU clones of trunk tensors for equality tests."""
    skip_prefixes = ("tok_emb.", "lm_head.")
    out: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if any(name.startswith(p) for p in skip_prefixes):
            continue
        out[name] = tensor.detach().cpu().clone()
    return out
