# Training matrix runner contract (design draft)

**Status:** design only — entrypoint spec for future `scripts/run_training_matrix.py`  
(to be implemented alongside `model/` / `train/`).  
**Consumes:** `COMPUTE_MATCH_BUDGETS.md` (locked \(T\), floor, resampling, masking).  
**Plan:** `EXPERIMENT_PLAN.md` §5 — 3–5 replications; report **median + IQR** (bootstrap optional).  
**Config discipline:** comparable cells differ only by `codec`, `match_regime`, and replication seeds (shared locked difficulty, architecture, \(L\), tokenizer).

No training code in this document.

---

## 1. Units

| Unit | Meaning |
|------|---------|
| **Replication** | One of \(R \in \{3,4,5\}\) independent runs of the *same* `(codec, match_regime)` cell, differing in model init + data-order seeds. |
| **Cell** | `(codec, match_regime)` — the scientific comparison atom. |
| **Run** | One training job: one cell × one `replication_id`. |
| **Matrix** | Cartesian product of codecs × match regimes × replications, plus shared locked knobs. |

Default for full v0 reporting: **\(R = 5\)**. Pilots may use \(R = 3\); never \(R < 3\) for a claim that cites §5.

---

## 2. Single-run config schema

Each run is fully specified by a JSON-serializable config (field names stable for logging / resume).

### 2.1 Identity (must differ across comparable runs only as allowed)

| Field | Type | Notes |
|-------|------|--------|
| `codec` | `str` | One of `A`, `B`, `C`, `D`, `B_shuffle`, `B_reverse`, `B_bag`. |
| `match_regime` | `str` | `compute_matched` \| `info_matched`. |
| `replication_id` | `int` | `0 .. R-1`. |
| `model_init_seed` | `int` | Weights / dropout RNG seed for this replication. |
| `data_shuffle_seed` | `int` | Reshuffle-per-epoch RNG seed for this replication (**paired** with `model_init_seed`; see §3). |

### 2.2 Budget (from `COMPUTE_MATCH_BUDGETS.md`, not free knobs)

| Field | Type | Compute-matched | Info-matched |
|-------|------|-----------------|--------------|
| `n_episodes` | `int` | Floored table (§1): A `50000`, B(+E) `78770`, C `73330`, D `65372` | Always `50000` |
| `T_eff` | `float` | Floored table: A `108438000`, B(+E) `108437932.8`, C `108437470.8`, D `108436458.72` | Codec’s one-pass total (§2): A `108438000`, B(+E) `68832000`, C `73938000`, D `82938000` |
| `T_target` | `float` | `108438000` (nominal; logged for audit) | `null` / omitted |
| `budget_profile` | `str` | `full_v0` (scientific) \| `tier1` \| `tier0` (pilots) | same |
| `avg_tokens_per_observation` | `float` | From `token_rates_train_v4.json` / §0 (E-variants = B) | same |
| `n_steps` | `int` | `30` | `30` |
| `token_rate_artifact` | `str` | `obs_codecs/token_rates_train_v4.json` | same |

Runner **must not** recompute floor ad hoc in a codec-specific way; it loads the locked table (or regenerates via the documented formula with §0 exact decimals). Stopping criterion: consume exactly `n_episodes` episode presentations (hence ≈ `T_eff` token-updates under the locked rates).

#### 2.2.1 Budget profiles (single formula, three scales)

`train/config.py` derives every budget from the `COMPUTE_MATCH_BUDGETS.md` §0 formula using `Decimal` arithmetic; no tier hardcodes its own literals. A `BudgetProfile` is fully described by `pool_episodes`, which is simultaneously the `info_matched` episode count and the anchor for `T_target = pool_episodes × n_steps × avg_A`:

| Profile | `pool_episodes` | `T_target` | Interpretation |
|---------|-----------------|------------|----------------|
| `tier0` | `200` | `433752` | `plumbing_only` — pipeline validation only |
| `tier1` | `2000` | `4337520` | `plumbing_only` — dry-run scale, matches current encoded pool |
| `full_v0` | `50000` | `108438000` | `scientific` — the locked v0 table |

At `pool_episodes = 50000` the derivation reproduces the §1/§2 tables exactly; `tests/test_matrix_runner.py::test_full_v0_derivation_matches_locked_doc_table` is the drift guard. Pilot-tier metrics are **not** interpretable in any direction (see `COMPUTE_MATCH_BUDGETS.md` §4 locked item 9); the manifest records this as `interpretation: "plumbing_only"`.

### 2.3 Locked shared knobs (identical across the matrix)

| Field | Type | Status |
|-------|------|--------|
| `tokenization_scheme_version` | `str` | `regex_v4` |
| `vocab_policy` | `str` | train-only vocab; B-family shared (`COMPUTE_MATCH_BUDGETS` / vocab builder) |
| `context_length_L` | `int` | **1024** (locked; `model/README.md` Decision 1) |
| `document_masking` | `bool` | `true` (locked) |
| `boundary_markers` | `bool` | `true` — insert `<bos>`/`<eos>` at episode boundaries |
| `mask_source` | `str` | `segment_ids` (explicit array; not token-scan) |
| `loss_target_policy` | `str` | `all_non_pad` (locked) — cross-segment boundary targets are scored; measured residual in `COMPUTE_MATCH_BUDGETS.md` §4 item 4. Logged on every run. |
| `reshuffle_policy` | `str` | `reshuffle_per_epoch` (not with-replacement) |
| `difficulty` | object / id | locked `DEFAULT_DIFFICULTY_V0` (train/val); OOD eval uses its own split |
| `dataset_split_train` | `str` | path / id for train HDF5 + encoded parquet |
| `architecture_id` | `str` | shared model family / depth / width hash |
| `optimizer_id` | `str` | shared optimizer + schedule family |

### 2.4 Reproducibility log (required on every run)

Per `.cursor/rules/02-reproducibility` and `PROGRAM_SPEC` §5:

- `git_commit`
- full config dump plus `config_sha256` (SHA-256 over the sorted config dump; wall-clock is not a config field, so it is excluded by construction) — also the resume identity key (§4.5)
- `budget_profile`
- `harness_step_cap` — `null` for a full-budget run; an integer marks a step-capped smoke run whose `tokens_consumed` is below `T_eff`
- `dataset_checksum` — SHA-256 of the encoded train parquet — plus `episode_pool_source`, `episode_pool_size`, and `episode_pool_fingerprint` (digest of the tokenized pool actually consumed). The fingerprint is required because the matrix runner injects a cached, possibly truncated pool; the file checksum alone would not pin what the run trained on.
- `model_param_count`
- `tokens_consumed` / `optimizer_steps` / FLOPs estimate (token-updates proxy under fixed \(L\))
- `codec`, `match_regime`, `replication_id`, both seeds
- wall-clock (engineering only; not a matching axis)

### 2.5 Example (compute-matched, Codec B, replication 2)

```json
{
  "codec": "B",
  "match_regime": "compute_matched",
  "replication_id": 2,
  "model_init_seed": 1000002,
  "data_shuffle_seed": 2000002,
  "n_episodes": 78770,
  "T_eff": 108437932.8,
  "T_target": 108438000,
  "budget_profile": "full_v0",
  "avg_tokens_per_observation": 45.888,
  "n_steps": 30,
  "token_rate_artifact": "obs_codecs/token_rates_train_v4.json",
  "tokenization_scheme_version": "regex_v4",
  "document_masking": true,
  "mask_source": "segment_ids",
  "reshuffle_policy": "reshuffle_per_epoch",
  "context_length_L": 1024,
  "architecture_id": "dec_only_v0_L1024_d384_l6_h6_ff1536_vpad48",
  "optimizer_id": "adamw_v0_lr3e4_wd0.1_bs8"
}
```

### 2.6 Checkpoint policy (locked)

**Evaluate only the final checkpoint** — the weights at the end of the run’s scheduled `n_episodes` consumption (i.e. after the budgeted episode presentations / ≈ `T_eff` token-updates).  

**Forbidden for scoring:** “best-of-training” selection via val-loss monitoring, early stopping, or any other mid-run metric that picks a checkpoint other than the final step.

**Rationale:** under compute-matched, codecs take different numbers of passes over the same 50k pool (A ≈ 1.00×, D ≈ 1.31×, B ≈ 1.58×). Best-checkpoint selection would give longer runs more opportunities to land on a favorable snapshot by chance — a silent “more tries” advantage that the compute-matched regime is designed to neutralize. Info-matched cells share episode count but must use the **same** final-only rule so aggregation (§5) stays uniform across regimes.

**Intermediate checkpoints:** optional for diagnostics / learning curves. If saved, mark them explicitly as **`non_scoring`** (e.g. path or metadata flag). Only the **final** checkpoint’s eval numbers may enter `metrics_eval.json` used by §5 aggregation. Runner must refuse to aggregate a cell if any replication’s scoring metrics were produced from a non-final checkpoint.

Config / log fields (recommended):

| Field | Value |
|-------|--------|
| `scoring_checkpoint` | `final` |
| `early_stopping` | `false` |
| `best_checkpoint_selection` | `false` |

---

## 3. Seed pairing (locked decoupling)

For replication \(r \in \{0,\ldots,R-1\}\):

```text
model_init_seed   = MODEL_SEED_BASE   + r
data_shuffle_seed = SHUFFLE_SEED_BASE + r
```

with **distinct bases** (recommended defaults: `MODEL_SEED_BASE = 1_000_000`, `SHUFFLE_SEED_BASE = 2_000_000`) so the two streams never collide and remain independently offset.

**Within one replication \(r\):** every codec × both match regimes share the **same** `(model_init_seed, data_shuffle_seed)` pair only where scientifically required:

- **`data_shuffle_seed`:** identical across **all codecs** in replication \(r\) (locked: shuffle methodology / order stream not codec-specific).
- **`model_init_seed`:** **Locked (Decision 3):** shared **trunk** init from `model_init_seed` across codecs in replication \(r\); **embedding table + untied output head** init from `blake2b_u64(f"{model_init_seed}:{codec}")` (same recipe as `derive_transform_seed`). Embedding tables are size-matched via vocab padding to 48 (`model/README.md` Decision 2), so trunk parameter tensors are shape-identical and bit-identical across codecs for a given `model_init_seed`.

**Across replications:** both seeds change with \(r\) (never reuse one shuffle for all model inits).

E-transform episode content seeds (`derive_transform_seed`) remain as already locked in the encode pipeline; they are **not** the training shuffle seed.

---

## 4. Matrix expansion (`run_training_matrix.py`)

### 4.1 Inputs

- `--tier {0,1}` / `--budget-profile {tier0,tier1,full_v0}`: selects the `BudgetProfile` (§2.2.1). Exactly one is required — there is no default scale, so no run can silently land at the wrong budget.
- `codecs`: default `["A","B","C","D","B_shuffle","B_reverse","B_bag"]` (or a subset for pilots)
- `match_regimes`: `["compute_matched","info_matched"]`
- `n_replications`: `3`–`5` (default `5`; pilots may use fewer)
- shared locked knobs (§2.3), including `context_length_L`
- paths: dataset, vocabs, output root
- `--smoke-max-steps`: harness-only optimizer-step cap. Recorded in both `reproducibility.json` and the manifest; a capped run is a plumbing artifact, never a budgeted one.

### 4.2 Expansion

```text
for match_regime in match_regimes:
  for codec in codecs:
    for r in range(n_replications):
      emit RunConfig(codec, match_regime, r, seeds(r), budget(codec, match_regime), shared...)
      launch or enqueue train(run_config)
```

### 4.3 Outputs (per run)

Directory convention (illustrative):

```text
artifacts/runs/{budget_profile}/{match_regime}/{codec}/rep{r}/
  config.json
  metrics_train.jsonl
  metrics_eval.json          # scoring: final checkpoint only (§2.6); schema §4.3.1
  checkpoints/
    final.pt                 # scoring
    step_*.pt                # optional; non_scoring diagnostics only
  reproducibility.json
```

The `{budget_profile}` segment (`tier0` / `tier1` / `full_v0`) keeps pilot artifacts physically separate from scientific ones, so a pilot run can never be swept into aggregation by a path glob. `metrics_eval.json` is **absent** until `eval/` gains a producer — it is never stubbed with placeholder values.

#### 4.3.1 `metrics_eval.json` schema (keyed by eval split)

Scoring metrics are **nested by eval split**, never flat at the top level. Active v0 splits at minimum: `val`, `ood_partial_obs` (add further `ood_*` keys when those splits unlock).

```json
{
  "val": {
    "next_state_mse": 0.0,
    "regime_accuracy": 0.0,
    "intervention_accuracy": 0.0,
    "calibration_ece": 0.0
  },
  "ood_partial_obs": {
    "next_state_mse": 0.0,
    "regime_accuracy": 0.0,
    "intervention_accuracy": 0.0,
    "calibration_ece": 0.0
  }
}
```

Metric key set follows `EXPERIMENT_PLAN.md` eval tasks (names may grow; every key present under one split should be present under all reported splits for that run, using `null` only if a task is explicitly inapplicable). All values are from the **final** checkpoint (§2.6).

### 4.4 Entrypoint responsibilities

All gates run **before any cell launches**, so a misconfigured matrix fails in seconds rather than mid-sweep:

1. Validate every config against the profile's freshly derived budget table (reject wrong `n_episodes` / `T_eff` / `T_target` / rate for `(codec, match_regime)`).
2. Enforce shared `data_shuffle_seed` across codecs for the same `replication_id`, distinct across replications, and both seeds paired to `replication_id` per §3.
3. Refuse to start if `document_masking` is false, `reshuffle_policy != reshuffle_per_epoch`, or `scoring_checkpoint != final` / early-stopping enabled.
4. **Pool-size guard:** for every codec in the matrix, load the encoded train split and assert the pool holds exactly `profile.pool_episodes` episodes. A stale or short corpus aborts the matrix instead of silently shrinking one codec's budget.
5. Write aggregation inputs (§5) after all reps for a cell complete — using **only** final-checkpoint eval metrics (§2.6), preserving the eval-split key. *(Deferred: no `metrics_eval.json` producer exists yet, so §5 is unimplemented.)*

The episode-pool cache is **process-scoped**: one tokenized pool per codec, held in memory for the duration of one invocation and never persisted. Every `--resume` re-invocation rebuilds it from the parquet and re-runs the guard.

### 4.5 Resume and manifest

`artifacts/runs/{budget_profile}/manifest.json` is the resume state; the runner never re-derives progress by scanning history.

- **Cell status:** `pending` → `running` → `completed` \| `failed` \| `interrupted`, plus the cell's `config_sha256`, `n_episodes`, `T_eff`, and (on completion) `optimizer_steps` / `tokens_consumed`.
- **Skip rule:** a cell is skipped only when `config.json`, `reproducibility.json`, and `checkpoints/final.pt` all exist, both JSON files parse, and the on-disk `config.json` hashes equal to the expected config. Anything else — missing file, truncated JSON, or a config from a different profile — re-runs the cell. `--force` re-runs unconditionally.
- **Atomic writes (locked):** *every* manifest mutation, including the `KeyboardInterrupt` flush, goes through one writer that writes a sibling `manifest.json.tmp`, `fsync`s it, then `os.replace`s it onto the target. The temp file must share the target's directory: `os.replace` is only atomic within one volume. A reader therefore observes either the previous or the next manifest, never a truncated one.
- **Failure isolation:** a crashing cell is recorded as `failed` with its exception and the matrix continues; the script exits non-zero if any cell failed.

---

## 5. Aggregation across replications

**Status: specified, not implemented.** Aggregation consumes `metrics_eval.json`, and `eval/` has no producer yet, so there is no input to aggregate. The runner therefore writes no aggregates and stubs no placeholder eval files. Only `full_v0` runs are ever eligible for aggregation; `tier0` / `tier1` artifacts are excluded by both their `interpretation` field and their separate path subtree.

Per `EXPERIMENT_PLAN.md` §5: **report median + IQR**; bootstrap optional.

### 5.1 Per cell

For each `(codec, match_regime, eval_split)` and each primary metric \(m\) (eval tasks from the plan: next-state, regime, intervention, …):

- Collect \(m_0,\ldots,m_{R-1}\) from the \(R\) runs — each \(m_r\) from that run’s **final** checkpoint only (§2.6), taken from `metrics_eval.json[eval_split][m]`.
- **Primary report:** \(\mathrm{median}(m)\), \(\mathrm{IQR}(m) = Q_3 - Q_1\) (or report \([Q_1, Q_3]\) explicitly).
- **Quartile convention (locked):** compute \(Q_1\) / median / \(Q_3\) via `numpy.percentile(..., method="linear")` (NumPy ≥ 1.22; alias of the historical default linear interpolation). This convention is fixed for all aggregation runs — do not switch percentile methods between cells or papers.
- **Optional:** bootstrap CI on the median (same seed for bootstrap RNG logged).
- **\(R = 3\) caveat:** with only three values, \(Q_1\)/\(Q_3\) sit near min/max; treat IQR at \(R = 3\) as a **rough spread indicator**, not a precise quartile statistic. Prefer **\(R = 5\)** for any claim intended as a primary result (aligned with `EXPERIMENT_PLAN.md` §5’s 3–5 range and preference for adequate replication).

Do **not** report mean ± std as the headline for §5 claims (variance in small models; plan specifies median + IQR).

### 5.2 Cross-codec comparison

- Compare **medians** within the same `match_regime` **and** the same `eval_split`.
- Accept directional claims only if both match regimes agree, **or** qualify (*wins under info-matched, loses under compute-matched*, etc.).
- Never pool metrics across `match_regime` without splitting.
- Never pool metrics across eval splits (`val` vs `ood_*`) without explicitly comparing them — this is the mechanism eval task #7 (OOD generalization) depends on.

### 5.3 Aggregation artifact

**Choice:** keep `eval_split` as a **path segment** (first-class, parallel to `match_regime`), not only a nested JSON key:

```text
artifacts/runs/aggregates/{match_regime}/{eval_split}/summary.json
```

Examples: `.../compute_matched/val/summary.json`, `.../info_matched/ood_partial_obs/summary.json`.

Schema sketch:

```json
{
  "match_regime": "compute_matched",
  "eval_split": "val",
  "n_replications": 5,
  "aggregation": "median_iqr",
  "metrics": {
    "next_state_mse": {
      "A": {"median": 0.0, "q1": 0.0, "q3": 0.0, "values": []},
      "B": {"median": 0.0, "q1": 0.0, "q3": 0.0, "values": []}
    },
    "regime_accuracy": {
      "A": {"median": 0.0, "q1": 0.0, "q3": 0.0, "values": []},
      "B": {"median": 0.0, "q1": 0.0, "q3": 0.0, "values": []}
    }
  }
}
```

`eval_split` is duplicated inside the file for self-description; the path is authoritative for matrix layout.
---

## 6. Relation to other docs

| Doc | Role |
|-----|------|
| `COMPUTE_MATCH_BUDGETS.md` | \(T\), floor, reshuffle, masking, E-inherit |
| `EXPERIMENT_PLAN.md` §5 | regimes + median/IQR replications |
| `PROGRAM_SPEC.md` | typed pipeline; log fields |
| This file | runner / config / aggregation contract for `scripts/run_training_matrix.py` |

---

## 7. Resolved model/train decisions (was open)

Documented with rationales in `model/README.md` (and mirrored in `PROGRAM_SPEC.md` §Model):

1. **`context_length_L = 1024`** + document-masked packer with explicit `segment_ids` (`train/packer.py`).
2. **Vocab-size parity:** pad embedding / LM-head tables to shared size **48** (Decision 2a) — exact param-count parity across codecs.
3. **`model_init_seed`:** trunk shared; embedding + output head codec-tagged via `blake2b_u64` (Decision 3).

**Also locked with `model/`:**

| Field | Value |
|-------|--------|
| `architecture_id` | `dec_only_v0_L1024_d384_l6_h6_ff1536_vpad48` |
| `optimizer_id` | `adamw_v0_lr3e4_wd0.1_bs8` |
| `shared_vocab_size` | `48` |

---

## 8. Implementation status

| Area | Status |
|------|--------|
| §2 config schema + validation | `train/config.py` |
| §2.2.1 budget profiles (derived, drift-guarded) | `train/config.py`, `tests/test_matrix_runner.py` |
| §3 seed pairing | `train/config.py`, enforced in preflight |
| §4.2 expansion, §4.5 resume + manifest | `train/matrix.py` |
| §4.3 per-run outputs (minus `metrics_eval.json`) | `train/loop.py` |
| §4.4 preflight gates + pool guard | `scripts/run_training_matrix.py` |
| §4.3.1 `metrics_eval.json` | **not implemented** — no `eval/` producer |
| §5 aggregation | **not implemented** — no input |
| 50 000-episode re-encode + streaming packer | **follow-up**, gated on tiers 0/1 passing; required only for `full_v0` |
