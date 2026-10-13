# Model — locked cross-codec fairness decisions

Stage-1 decoder-only transformer for next-token prediction (`EXPERIMENT_PLAN.md` §6).
Comparable runs share architecture, \(L\), tokenizer (`regex_v4`), and optimizer family;
cells differ only by `codec`, `match_regime`, and replication seeds
(`TRAINING_RUNNER_CONTRACT.md`).

## Decision 1 — Context length \(L = 1024\) and packer

**Choice:** fixed context window \(L = 1024\) for every codec; pack episode token
streams (reshuffle-per-epoch order) into exact-\(L\) chunks with `<bos>`/`<eos>` at
every episode boundary; emit parallel `segment_ids` (mask source of truth);
derive causal + document attention masks from `segment_ids` and pad positions;
pad a short final chunk with `<pad>` and exclude pads from attention and loss.

**Rationale:** `COMPUTE_MATCH_BUDGETS.md` already locks fixed-\(L\) packing so
token-updates ≈ FLOPs across codecs whose mean episode lengths differ by ~1.6×,
and locks document masking so packed windows cannot leak regime / `hidden_c` /
intervention structure across episode boundaries. Among \(\{512, 1024, 2048\}\),
**1024** is the pilot-scale sweet spot: attention cost scales \(\propto L^2\), so
2048 is ~4× more expensive per sequence and sits outside the ~5–20M /
tens-of-minutes single-seed envelope assumed for \(T_{\mathrm{target}}\); 512 is
the same “all-split” regime as 1024 (A≈2169, B≈1377 tok/episode both exceed
\(L\)) but fragments episodes more aggressively. Multi-pack density is **not**
required for fairness under locked masking — density differences affect only
padding efficiency. Packer: `train/packer.py`; mask derivation: `model/architecture.py`.

## Decision 2 — Vocab-size parity via shared padded table size 48

**Choice:** option **(a)** — every codec’s embedding table and untied LM head use
`shared_vocab_size = 48` (rounded up from max train vocab 42 under `regex_v4`
with headroom). Codec-native ids occupy \([0, V_{\mathrm{codec}})\;\); unused
rows are never indexed.

**Rationale:** native sizes are A=42, B-family=34, C=37, D=30. Letting table size
track \(V_{\mathrm{codec}}\) would make “identical architecture” literally false
(different param counts) and would force every result write-up to defend a
codec-specific model-size confound. Padding six unused rows past A’s 42 costs
\(6 \times d_{\mathrm{model}}\) parameters on emb + head (~4.6k weights at
\(d=384\), ≪0.05% of the ~11M trunk) and buys exact `model_param_count` parity
logged in `reproducibility.json`. Rejected (b): documenting a small delta is
possible but keeps a perpetual reporting caveat for no scientific gain.

## Decision 3 — Trunk-shared `model_init_seed` + codec-tagged emb/head

**Choice:** all non-embedding, non-output-head parameters (trunk) are initialized
from `model_init_seed` identically across codecs within replication \(r\).
Token embedding and untied output projection are initialized from
`derive_codec_init_seed(model_init_seed, codec) =
blake2b_u64(f"{model_init_seed}:{codec}")` — the same blake2b-8 little-endian
recipe as `derive_transform_seed` (never builtin `hash()`).

**Rationale:** Decision 2 makes trunk tensors shape-identical, so a shared trunk
init is scientifically meaningful and keeps “replication \(r\)” aligned across
codecs. Embedding rows do **not** share semantics across codecs (id \(k\) is a
different surface token under A vs B), so a fully shared emb/head init would
pair unrelated symbols to the same random vectors — a silent cross-codec
coupling. Codec-tagged emb/head init avoids that while preserving paired trunk
noise. Implementation: `model/init.py`.

## Locked architecture / optimizer ids

| Knob | Value |
|------|--------|
| `architecture_id` | `dec_only_v0_L1024_d384_l6_h6_ff1536_vpad48` |
| `d_model` / layers / heads / `d_ff` | 384 / 6 / 6 / 1536 |
| `context_length_L` | 1024 |
| `shared_vocab_size` | 48 |
| `optimizer_id` | `adamw_v0_lr3e4_wd0.1_bs8` |
| Stage-1 objective | next-token NLL only (no shared state head) |

Approximate param count at vocab pad 48: **11 056 896** (~11.06M), inside the
5–20M pilot band; exact count is also written to `reproducibility.json`.
