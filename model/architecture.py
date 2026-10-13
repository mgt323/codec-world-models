"""Stage-1 decoder-only transformer (next-token prediction).

Architecture is shared across codecs. Embedding / LM-head tables use the locked
shared vocab pad size (Decision 2). Forward takes ``segment_ids`` and builds the
causal + document attention mask internally (Decision 1 / runner contract
``mask_source=segment_ids``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from obs_codecs.vocab import PAD_ID

# Locked knobs (model/README.md).
ARCHITECTURE_ID = "dec_only_v0_L1024_d384_l6_h6_ff1536_vpad48"
CONTEXT_LENGTH_L = 1024
SHARED_VOCAB_SIZE = 48
D_MODEL = 384
N_LAYERS = 6
N_HEADS = 6
D_FF = 1536
# Segment id written on pad positions; never equals a real episode segment.
PAD_SEGMENT_ID = -1


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Hyperparameters for the shared v0 decoder-only trunk."""

    vocab_size: int = SHARED_VOCAB_SIZE
    context_length: int = CONTEXT_LENGTH_L
    n_layers: int = N_LAYERS
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    d_ff: int = D_FF
    dropout: float = 0.0
    pad_id: int = PAD_ID
    architecture_id: str = ARCHITECTURE_ID

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.context_length < 1:
            raise ValueError("context_length must be positive")


def attention_allow_mask(
    segment_ids: torch.Tensor,
    *,
    pad_segment_id: int = PAD_SEGMENT_ID,
) -> torch.Tensor:
    """Boolean mask ``(B, L, L)``: True where query may attend to key.

    Combines causal lower-triangular, same ``segment_id``, and non-pad keys.
    """
    if segment_ids.ndim != 2:
        raise ValueError(f"segment_ids must be (B, L), got {tuple(segment_ids.shape)}")
    _batch, length = segment_ids.shape
    device = segment_ids.device
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    key_is_content = segment_ids != pad_segment_id
    key_ok = key_is_content[:, None, :]
    allow = causal.unsqueeze(0) & same_segment & key_ok
    # Pad (and any fully-blocked) rows would be all -inf under softmax; keep
    # the diagonal so numerics stay finite. Loss already ignores pad targets.
    eye = torch.eye(length, dtype=torch.bool, device=device)
    return allow | eye.unsqueeze(0)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def _project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def attention_probs(
        self, x: torch.Tensor, allow: torch.Tensor
    ) -> torch.Tensor:
        """Return softmax attention weights ``(B, H, L, L)`` under ``allow``."""
        q, k, _v = self._project_qkv(x)
        scale = self.head_dim**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        additive = torch.zeros(
            scores.shape[0],
            1,
            scores.shape[-1],
            scores.shape[-1],
            device=scores.device,
            dtype=scores.dtype,
        )
        additive = additive.masked_fill(~allow.unsqueeze(1), float("-inf"))
        scores = scores + additive
        return torch.softmax(scores, dim=-1)

    def forward(self, x: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self._project_qkv(x)
        attn_mask = allow.unsqueeze(1)
        additive = torch.zeros(
            batch, 1, length, length, device=x.device, dtype=x.dtype
        )
        additive = additive.masked_fill(~attn_mask, float("-inf"))
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), allow)
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderOnlyLM(nn.Module):
    """Decoder-only LM. Does not accept a caller-supplied dense attention mask."""

    def __init__(self, cfg: TransformerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TransformerConfig()
        c = self.cfg
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model, padding_idx=c.pad_id)
        self.pos_emb = nn.Embedding(c.context_length, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        # Set on every forward; integration tests read the mask the model used.
        self.last_allow_mask: torch.Tensor | None = None

    def derive_allow_mask(self, segment_ids: torch.Tensor) -> torch.Tensor:
        """Document + causal allow mask — sole mask source for forward."""
        return attention_allow_mask(segment_ids)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        _batch, length = input_ids.shape
        if length > self.cfg.context_length:
            raise ValueError(
                f"sequence length {length} exceeds context_length "
                f"{self.cfg.context_length}"
            )
        positions = torch.arange(length, device=input_ids.device)
        return self.drop(self.tok_emb(input_ids) + self.pos_emb(positions))

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits ``(B, L, vocab_size)``.

        ``segment_ids`` must be parallel to ``input_ids``; pad positions use
        ``PAD_SEGMENT_ID``. Dense masks from the caller are rejected by API design.
        """
        if input_ids.shape != segment_ids.shape:
            raise ValueError(
                f"input_ids shape {tuple(input_ids.shape)} != "
                f"segment_ids shape {tuple(segment_ids.shape)}"
            )
        x = self._embed(input_ids)
        allow = self.derive_allow_mask(segment_ids)
        self.last_allow_mask = allow
        for block in self.blocks:
            x = block(x, allow)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def attention_probs_at_layer(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        *,
        layer_index: int = 0,
    ) -> torch.Tensor:
        """Softmax attention weights ``(B, H, L, L)`` at ``layer_index``.

        Uses the same ``derive_allow_mask(segment_ids)`` path as ``forward``.
        """
        if not (0 <= layer_index < len(self.blocks)):
            raise ValueError(f"layer_index out of range: {layer_index}")
        if input_ids.shape != segment_ids.shape:
            raise ValueError("input_ids / segment_ids shape mismatch")
        x = self._embed(input_ids)
        allow = self.derive_allow_mask(segment_ids)
        self.last_allow_mask = allow
        for i, block in enumerate(self.blocks):
            if i == layer_index:
                h = block.ln1(x)
                return block.attn.attention_probs(h, allow)
            x = block(x, allow)
        raise RuntimeError("unreachable")

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trunk_parameters(self) -> list[nn.Parameter]:
        """All parameters except token embedding and LM head."""
        emb_ids = {id(p) for p in self.tok_emb.parameters()}
        head_ids = {id(p) for p in self.lm_head.parameters()}
        skip = emb_ids | head_ids
        return [p for p in self.parameters() if id(p) not in skip]


def estimate_flops_proxy_token_updates(
    tokens_consumed: int,
    *,
    context_length: int = CONTEXT_LENGTH_L,
) -> dict[str, float | int]:
    """Log FLOPs estimate under fixed-L (token-updates proxy).

    Per COMPUTE_MATCH_BUDGETS.md, matched token-updates approx matched FLOPs
    when architecture and L are identical; we log the proxy explicitly.
    """
    return {
        "token_updates": tokens_consumed,
        "context_length_L": context_length,
        "flops_proxy": "token_updates_under_fixed_L",
        # Rough transformer FLOPs ~ 6 * N_params * tokens (Kaplan-style);
        # filled by caller with actual N when available.
        "note": "primary matching axis is token_updates; 6ND is secondary",
    }


def build_model(cfg: TransformerConfig | None = None) -> DecoderOnlyLM:
    return DecoderOnlyLM(cfg or TransformerConfig())
