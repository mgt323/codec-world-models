"""Model init determinism and trunk/embedding separation (Decision 3)."""

from __future__ import annotations

import torch

from model.architecture import TransformerConfig, build_model
from model.init import (
    derive_codec_init_seed,
    initialize_model,
    trunk_state_dict,
)
from obs_codecs.transforms_e import blake2b_u64


def test_derive_codec_init_seed_uses_blake2b_recipe() -> None:
    seed = 1_000_000
    assert derive_codec_init_seed(seed, "A") == blake2b_u64(f"{seed}:A")
    assert derive_codec_init_seed(seed, "A") == derive_codec_init_seed(seed, "A")
    assert derive_codec_init_seed(seed, "A") != derive_codec_init_seed(seed, "B")


def test_same_seed_codec_identical_embedding_init() -> None:
    cfg = TransformerConfig()
    a1 = build_model(cfg)
    a2 = build_model(cfg)
    initialize_model(a1, model_init_seed=1_000_000, codec_name="A")
    initialize_model(a2, model_init_seed=1_000_000, codec_name="A")
    assert torch.equal(a1.tok_emb.weight, a2.tok_emb.weight)
    assert torch.equal(a1.lm_head.weight, a2.lm_head.weight)


def test_same_seed_different_codec_shared_trunk_different_emb() -> None:
    cfg = TransformerConfig()
    model_a = build_model(cfg)
    model_b = build_model(cfg)
    seed = 1_000_000
    initialize_model(model_a, model_init_seed=seed, codec_name="A")
    initialize_model(model_b, model_init_seed=seed, codec_name="B")

    trunk_a = trunk_state_dict(model_a)
    trunk_b = trunk_state_dict(model_b)
    assert trunk_a.keys() == trunk_b.keys()
    for key in trunk_a:
        assert torch.equal(trunk_a[key], trunk_b[key]), f"trunk mismatch at {key}"

    assert not torch.equal(model_a.tok_emb.weight, model_b.tok_emb.weight)
    assert not torch.equal(model_a.lm_head.weight, model_b.lm_head.weight)


def test_param_count_in_pilot_band() -> None:
    model = build_model(TransformerConfig())
    n = model.param_count()
    assert 5_000_000 <= n <= 20_000_000, n


# Locked Decision 2 shared count for dec_only_v0_L1024_d384_l6_h6_ff1536_vpad48.
_LOCKED_PARAM_COUNT = 11_056_896
_PARAM_PARITY_CODECS = ("A", "B", "C", "D")  # E-variants share B's table size


def test_param_count_parity_across_codecs() -> None:
    """Empirical Decision 2 guard: every codec instantiation has identical numel.

    Instantiates A/B/C/D with codec-tagged embedding init and counts
    ``sum(p.numel() for p in model.parameters())`` independently — not merely
    asserting shared ``TransformerConfig.vocab_size``.
    """
    seed = 1_000_000
    counts: dict[str, int] = {}
    for codec in _PARAM_PARITY_CODECS:
        model = build_model(TransformerConfig())
        initialize_model(model, model_init_seed=seed, codec_name=codec)
        counts[codec] = sum(p.numel() for p in model.parameters())

    assert counts["A"] == _LOCKED_PARAM_COUNT, counts
    assert len(set(counts.values())) == 1, counts
    for codec in _PARAM_PARITY_CODECS:
        assert counts[codec] == _LOCKED_PARAM_COUNT, (codec, counts)
