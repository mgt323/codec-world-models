# Codec World Models (`codec-world-models`)

`codec-world-models` is a controlled scientific experiment investigating how **observation codecs** (representation surface and structural inductive bias) affect the learnability of causal world models in autoregressive sequence models under **matched compute** and **matched information**. Rather than evaluating surface prose quality, the pipeline projects ground-truth 1D causal world trajectories (`State`) through a strict information barrier (`observe: State -> Observation`) that strips all latent variables (`hidden_c`, causal regime labels, oracle posteriors, and active `do()` intervention flags) before encoding observations into distinct codec alphabets (`A` essentialist captions, `B` relational event chains, `C` evidential records without oracle posteriors, `D` flat structured key-value baselines, and `E` structure-destroyed controls `B_shuffle`, `B_reverse`, `B_bag`).

Every primary codec satisfies deterministic information parity (`parse_k(encode_k(O)) == F(O)` under shared quantization bins) and feeds an identical decoder-only Transformer trunk (`dec_only_v0_L1024_d384_l6_h6_ff1536_vpad48`, 11,056,896 parameters) trained exclusively on next-token prediction over document-masked, fixed-length ($L = 1024$) packed episode windows. Scientific comparisons are evaluated across two mandatory regimes—**compute-matched** (fixed nominal token-update budget $T_{\mathrm{target}}$ with per-codec floored episode counts and deterministic reshuffle-per-epoch repetition) and **info-matched** (identical unique world episodes across codecs)—with post-hoc evaluation and frozen linear probing conducted strictly in world-state and intervention space.

---

## Documentation Map (Source of Truth)

| Document | Role | When to Read |
| :--- | :--- | :--- |
| [`docs/PROGRAM_SPEC.md`](docs/PROGRAM_SPEC.md) | Typed pipeline & access boundaries | Read before touching module interfaces, `State`/`Observation` types, dataset split definitions, or import boundaries (`obs_codecs/` and `train/` must never access `LatentState`). |
| [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) | Scientific specification & MVP design | Read to understand the research hypotheses, codec definitions (`A`–`E`), claim hierarchy, 8 evaluation tasks, representation similarity analysis (RSA), and readiness gates. |
| [`docs/COMPUTE_MATCH_BUDGETS.md`](docs/COMPUTE_MATCH_BUDGETS.md) | Token-rate & budget arithmetic | Read when configuring or auditing `compute_matched` vs. `info_matched` budgets, `regex_v4` token rates, floor rounding, fixed-$L$ document-masked packing, or pilot tiers (`tier0`, `tier1`, `full_v0`). |
| [`docs/TRAINING_RUNNER_CONTRACT.md`](docs/TRAINING_RUNNER_CONTRACT.md) | Runner & aggregation specification | Read when working on `RunConfig`, seed pairing (`model_init_seed` and `data_shuffle_seed`), `scripts/run_training_matrix.py`, resume/manifest atomicity, final-only checkpoint rules, or replication aggregation. |
| [`docs/INVARIANTS.md`](docs/INVARIANTS.md) | Non-negotiable laws (`I1`–`I15`) | Read before proposing any code, config, or evaluation change to verify that none of the 15 machine- and review-enforced scientific invariants are violated. |
| [`docs/TECH_STACK.md`](docs/TECH_STACK.md) | Runtime & dependency constraints | Read when setting up the Python 3.13.15 environment with `uv`, checking approved libraries, or verifying language and typing conventions. |
| [`CURRENT_STATE.md`](CURRENT_STATE.md) | **Reality-vs-Documentation Audit** | Read first to see the exact implementation status of every module, `I1`–`I15` verification results, spec-vs-code discrepancies, and the prioritized engineering backlog. |

Additional locked architectural rationales for the shared Transformer and initialization policy are documented in [`model/README.md`](model/README.md).

---

## Repository Structure

The repository is organized around strict module access boundaries (note that generated `artifacts/` are gitignored and must be materialized locally via the pipeline scripts below):

```text
codec-world-models/
├── .cursor/                     # Project rules (01-04), experiment-auditor agent, and workflow skills
├── docs/                        # Specifications: PROGRAM_SPEC, EXPERIMENT_PLAN, COMPUTE_MATCH_BUDGETS, TRAINING_RUNNER_CONTRACT, INVARIANTS, TECH_STACK
├── configs/
│   └── dataset_v0.py            # Locked v0 split specs (train, val, ood_partial_obs), base seeds, and scales
├── world/
│   ├── schema.py                # State, LatentState, Observation, FactRecord, observe(), quantize_value()
│   ├── simulate.py              # Deterministic 1D causal simulator, do() interventions, DEFAULT_DIFFICULTY_V0
│   └── parity_fixtures.py       # Deterministic 48-state and 500-state fixtures for parity & surface audits
├── obs_codecs/
│   ├── encode_a.py              # Codec A: substance / essentialist captions + parse_A()
│   ├── encode_b.py              # Codec B: process / relational event chain + ordered/unordered parsers
│   ├── encode_c.py              # Codec C: evidential record (constant alts, no oracle p) + parse_C()
│   ├── encode_d.py              # Codec D: flat structured key=value baseline + parse_D()
│   ├── transforms_e.py          # E-controls on B: shuffle_b(), reverse_b(), bag_b(), derive_transform_seed()
│   ├── diagnostics.py           # Locked regex_v4 provisional tokenizer and surface diagnostics
│   ├── vocab.py                 # Train-only vocab builder (shared B-family vocab) and OOV reporter
│   ├── parity_audit.py          # Reusable codec-agnostic parse(encode(O)) == F(O) auditor
│   ├── token_rates_train_v4.json        # Authoritative full-train regex_v4 token rates for budget derivation
│   ├── diagnostics_A.json               # regex_v4 single-codec diagnostic artifact (500-sample parity fixture)
│   ├── diagnostics_comparison_ABCD.json # 500-sample surface comparison across codecs A, B, C, D
│   └── diagnostics_comparison_B_E.json  # 500-sample surface comparison across B and E-transforms
├── data/
│   ├── generate_episodes.py     # Streaming split generator + intervention and counterfactual pair primitives
│   ├── hdf5_episodes.py         # Flat-column HDF5 persistence and canonical SHA-256 stream checksums
│   └── encode_episodes.py       # Offline HDF5 -> per-codec Parquet string encoder with 100% sample validation
├── model/
│   ├── README.md                # Locked fairness decisions (L=1024, shared vocab pad 48, trunk/emb init)
│   ├── architecture.py          # DecoderOnlyLM (d=384, l=6, h=6, ff=1536, vpad=48) + segment_ids mask
│   └── init.py                  # Shared trunk RNG + codec-tagged blake2b_u64 embedding/LM-head init
├── train/
│   ├── config.py                # RunConfig schema, Decimal budget derivation (tier0, tier1, full_v0), validation
│   ├── data.py                  # Parquet -> token ID loader, pool_fingerprint(), reshuffle-per-epoch ordering
│   ├── packer.py                # <bos>/<eos> episode framing and fixed L=1024 window packing with segment_ids
│   ├── loop.py                  # Stage-1 next-token NLL training loop, checkpointing, reproducibility.json
│   └── matrix.py                # Cartesian matrix expansion, seed pairing check, atomic manifest & resume
├── eval/
│   ├── regime_baseline.py       # World v0 calibration baselines (oracle, majority, O-only correlation heuristic)
│   └── difficulty_sweep_*.json  # Pre-computed difficulty sweep outputs used to lock DEFAULT_DIFFICULTY_V0
├── scripts/                     # CLI entrypoints for dataset generation, encoding, audits, and training
└── tests/                       # 26 pytest suites enforcing determinism, parity, leakage, and runner contracts
```

*(Note: The `probes/` directory referenced in `docs/EXPERIMENT_PLAN.md` §10 and `docs/PROGRAM_SPEC.md` §4.1 does not exist on disk yet.)*

---

## Quickstart & Environment Setup

### 1. Environment Installation

The project requires **Python 3.13.15** (pinned in `.python-version` and `pyproject.toml`) and uses [`uv`](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync
```

### 2. Run the Test Suite

Run the unit test suite (26 test files; 6 tests are marked `@pytest.mark.integration`. The 3 in `tests/test_packing_step_accounting.py` and `tests/test_real_a_document_masking_integration.py` auto-skip until `artifacts/encoded/train_A.parquet` and `artifacts/vocabs/vocab_A.json` are generated; the 3 CLI integration tests in `tests/test_matrix_runner.py` use their own setup, so see that file):

```bash
uv run pytest
```

### 3. End-to-End Pipeline Execution Order (Materializing Gitignored Artifacts)

Because `artifacts/` (`*.h5`, `*.parquet`, `vocabs/`, `runs/`) is gitignored, you must execute the pipeline scripts in the following exact sequence before running the training matrix:

#### Step 3.1 — Generate Raw World v0 HDF5 Splits
Generate `train.h5`, `val.h5`, `ood_partial_obs.h5`, and `seed_checksums.json`. Use `--scale dry_run` (2,000 episodes/split, sufficient for `tier0` and `tier1`) or `--scale v0` (50,000 train/val and 5,000 `ood_partial_obs`, required for `full_v0`):

```bash
# Dry-run scale (2,000 episodes per split -> required for --tier 0 and --tier 1)
uv run python scripts/generate_dataset.py \
  --scale dry_run \
  --out-dir artifacts/dataset_v0_dry_run

# Production v0 scale (50,000 train/val, 5,000 ood_partial_obs -> required for --budget-profile full_v0)
# uv run python scripts/generate_dataset.py --scale v0 --out-dir artifacts/dataset_v0
```

#### Step 3.2 — Encode Raw Splits into Per-Codec Parquet Corpora
Project each `State` through `observe(state)` and encode strings for all 7 variants (`A`, `B`, `C`, `D`, `B_shuffle`, `B_reverse`, `B_bag`), auditing a deterministic 1,000-row sample per file for 100% `FactRecord` recovery:

```bash
uv run python scripts/encode_dataset.py \
  --raw-dir artifacts/dataset_v0_dry_run \
  --out-dir artifacts/encoded
```

#### Step 3.3 — Build Train-Only `regex_v4` Vocabularies
Build `vocab_{A,B,C,D,B_shuffle,B_reverse,B_bag}.json` and `vocab_report.json` (with shared B-family vocabulary across `B` and its E-transforms):

```bash
uv run python scripts/build_vocab.py \
  --encoded-dir artifacts/encoded \
  --out-dir artifacts/vocabs
```

#### Step 3.4 — (Optional) Re-Verify Full-Train Token Rates & Codec Audits
`obs_codecs/token_rates_train_v4.json` is already committed in the repository, but can be regenerated or cross-checked alongside the parity audits:

```bash
uv run python scripts/compute_token_rates_train.py --encoded-dir artifacts/encoded
uv run python scripts/run_parity_audit.py --codec A
uv run python scripts/run_codec_comparison.py
uv run python scripts/run_b_e_diagnostics.py
```

#### Step 3.5 — Execute the Stage-1 Training Matrix
Note that `run_training_matrix.py` enforces an exact pool-size guard (`len(pool) == profile.pool_episodes`). If `artifacts/encoded/train_*.parquet` was generated at `dry_run` scale (2,000 episodes), `--tier 1` (`pool_episodes = 2000`) runs directly out of the box; for `--tier 0` (`pool_episodes = 200`) or `--budget-profile full_v0` (`pool_episodes = 50000`), pass `--dry-run` to inspect the plan or run when the corresponding corpus is present:

```bash
# Inspect the matrix plan and preflight checks without launching training
uv run python scripts/run_training_matrix.py --tier 1 --replications 3 --dry-run

# Launch Tier 1 pilot matrix (7 codecs x 2 match regimes x 3 replications)
uv run python scripts/run_training_matrix.py --tier 1 --replications 3

# Post-run artifact verification
uv run python scripts/verify_tier_artifacts.py tier1
```

### 4. What Is Not Runnable Yet

- **Trained-Model Evaluation (`eval/` Tasks 1–8 & `metrics_eval.json`)**: Only the non-learned World v0 calibration baseline (`eval/regime_baseline.py` / `scripts/run_regime_difficulty_sweep.py`) exists today. There is no script or module that loads `checkpoints/final.pt` to evaluate a trained Transformer on the 8 benchmark tasks or emit `metrics_eval.json`.
- **Cross-Replication Aggregation (`docs/TRAINING_RUNNER_CONTRACT.md` §5)**: Blocked until a `metrics_eval.json` producer is implemented.
- **Post-Hoc Frozen Linear Probes & RSA (`probes/`)**: The `probes/` package has not been created yet.
