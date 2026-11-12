# Current State & Reality-vs-Documentation Audit (`codec-world-models`)

## 1. Audit Metadata

| Field | Value |
| :--- | :--- |
| **Audit Date** | `2026-11-12` |
| **Repository Snapshot** | Live working tree (git repo present; specs moved to `docs/`; `README.md` and `CURRENT_STATE.md` untracked at audit time) |
| **Last Recorded Git Commit** | Git HEAD `56256d9386ee75eb47e6baf8ba2c68ec9b411bb6`. The commit `b80345c7f10ea9d786079efd2a8c3b2627178534` recorded in `obs_codecs/token_rates_train_v4.json` is the commit at which the token rates were generated, not HEAD. |
| **Target Branch / Version** | `main` / `v0.1.0` (`pyproject.toml`, Python `3.13.15` in `.python-version`) |
| **Governing Specifications** | `docs/PROGRAM_SPEC.md`, `docs/EXPERIMENT_PLAN.md`, `docs/COMPUTE_MATCH_BUDGETS.md`, `docs/TRAINING_RUNNER_CONTRACT.md`, `docs/INVARIANTS.md`, `docs/TECH_STACK.md` |

---

## 2. Executive Summary

- **World v0 Simulator, Information Contract & Codecs (`A`–`D` + `B` E-transforms) are 100% Built & Tested:** `world/schema.py`, `world/simulate.py`, and `obs_codecs/` cleanly enforce $O_t = \mathrm{Obs}(S_t)$ with zero latent or intervention leakage, deterministic `PCG64(seed)` trajectories, locked `DEFAULT_DIFFICULTY_V0`, hand-written `parse_k` parsers satisfying `parse_k(encode_k(O)) == F(O)` on shared quantization bins, and locked `regex_v4` digit-level tokenization (`obs_codecs/diagnostics.py`, `obs_codecs/vocab.py`).
- **Offline Data Pipeline & Stage-1 Training Infrastructure are Production-Ready:** `data/generate_episodes.py`, `data/hdf5_episodes.py`, and `data/encode_episodes.py` generate streaming HDF5 splits and validated Parquet string corpora; `model/architecture.py`, `model/init.py`, `train/packer.py`, `train/loop.py`, and `train/matrix.py` implement the shared 11.06M-parameter decoder-only Transformer (`L=1024`, `shared_vocab_size=48`), `segment_ids`-driven document attention masking, exact `Decimal` budget derivation (`tier0`, `tier1`, `full_v0`), and atomic `manifest.json` resume semantics.
- **Critical Gap — Zero Trained-Model Evaluation (`metrics_eval.json`) or Post-Hoc Probes (`probes/`):** While `eval/regime_baseline.py` implements non-learned oracle/majority/correlation-heuristic baselines to calibrate World v0 difficulty, **none of the 8 model evaluation tasks in `docs/EXPERIMENT_PLAN.md` §7 exist in code**, `metrics_eval.json` (`docs/TRAINING_RUNNER_CONTRACT.md` §4.3.1) has no producer, cross-replication median/IQR aggregation (§5) is unbuilt, and the `probes/` package (frozen linear probes + CKA/SVCCA/Procrustes RSA) does not exist on disk.
- **Subtle Runner & Data Edge Cases Require Hardening:** (1) `scripts/run_training_matrix.py._preflight()` passes `max_episodes=profile.pool_episodes` and asserts `len(pool) == profile.pool_episodes`, whereas `train/loop.py:run_training()` loads the **entire** Parquet file without truncating to `profile.pool_episodes` when called directly without an injected `episode_pool` (and silently falls back to `synthetic_episodes` if the Parquet path is missing); (2) `train/loop.py` packs all episode presentations in memory at once (`pack_episodes(presented)`), which will allocate ~108.4M tokens (~105.9k windows of length 1024) in RAM per `full_v0` run until a streaming packer is added.
- **Test Coverage is High for Built Modules (26 Pytest Suites):** All implemented invariants (`I1`–`I6`, `I8`–`I11`, `I13`) are guarded by automated AST inspection, byte-identity checks, and integration tests under `tests/`.

---

## 3. Component Implementation Matrix

Status legend: `[x] Done` (implemented and tested) · `[~] Partial/Stub` (partially built or limited to calibration/subset) · `[ ] Not Started` (missing on disk)

| Component / Contract | Status | Code Location | Notes / Missing Pieces |
| :--- | :---: | :--- | :--- |
| **World Schema & Observer (`State`, `LatentState`, `Observation`, `FactRecord`, `observe`)** | `[x] Done` | `world/schema.py`, `tests/test_world_schema.py`, `tests/test_fact_record.py`, `tests/test_information_contract.py` | Frozen `slots=True` dataclasses; `observe()` strips `LatentState` and `active_intervention`; shared 10-bin `VALUE_BIN_EDGES` quantization and `schema_checksum()`. |
| **World v0 1D Simulator & `do()` Interventions** | `[x] Done` | `world/simulate.py`, `tests/test_simulate.py`, `tests/test_simulate_intervention_asymmetry.py`, `tests/test_simulate_regime_correlation.py` | Deterministic `np.random.PCG64(seed)` draw order; implements `common_cause`, `a_causes_b`, `b_causes_a`, `spurious`; locks `DEFAULT_DIFFICULTY_V0`; raises `NotImplementedError` if `regime_switch_rate != 0.0` (expected for v0). |
| **World v0 Difficulty Calibration & Baselines** | `[x] Done` | `eval/regime_baseline.py`, `scripts/run_regime_difficulty_sweep.py`, `eval/difficulty_sweep_*.json`, `tests/test_regime_baseline.py` | Implements `predict_regime_oracle`, `majority_regime_accuracy`, `predict_regime_heuristic`, and `difficulty_sweep`. Calibrates the directional regime operating point and documents the single-episode `common_cause` vs. `spurious` ceiling. |
| **Codec A (Substance / Essentialist Caption)** | `[x] Done` | `obs_codecs/encode_a.py`, `tests/test_codec_a.py` | Implements `encode_A`, `parse_A`, `observation_from_A`. Fixed zero-bit `Relation: A and B are similar.` line excluded from `FactRecord`. Only `A-ordered` is implemented; optional `A-bag` control is not implemented. |
| **Codec B (Process / Relational Event Chain)** | `[x] Done` | `obs_codecs/encode_b.py`, `tests/test_codec_b.py` | Implements `encode_B`, `parse_B`, `observation_from_B` (positional), and `observation_from_B_unordered` (role-matched for E-transforms). Hidden `do()` never lexicalized. |
| **Codec C (Evidential Record, No Oracle $p$)** | `[x] Done` | `obs_codecs/encode_c.py`, `tests/test_codec_c.py` | Implements `encode_C`, `parse_C`, `observation_from_C`. Hardcodes `alts: {common_cause, a_causes_b, b_causes_a, spurious}` without importing `CausalRegime` or leaking oracle posteriors. Optional `C-shuffle` is not implemented. |
| **Codec D (Canonical Structured Baseline)** | `[x] Done` | `obs_codecs/encode_d.py`, `tests/test_codec_d.py` | Implements `encode_D`, `parse_D`, `observation_from_D` (`A.x=...; A.obs=...; ...`). Lean zero-decoration control. |
| **Codec E (Structure-Destroyed Controls on B)** | `[x] Done` | `obs_codecs/transforms_e.py`, `tests/test_transforms_e.py` | Implements `shuffle_b`, `reverse_b`, `bag_b`, and `derive_transform_seed(episode_seed, timestep, variant_name)` via `blake2b_u64`. Optional `A-bag` and `C-shuffle` mentioned in `docs/EXPERIMENT_PLAN.md` §3 are not implemented. |
| **Information-Parity Auditor (`parse_k(encode_k(O)) == F(O)`)** | `[x] Done` | `obs_codecs/parity_audit.py`, `world/parity_fixtures.py`, `scripts/run_parity_audit.py`, `tests/test_parity_audit.py` | Codec-agnostic auditor verified across 48-state and 500-state diverse fixtures covering all regimes, noise bins, partial-obs masks, and interventions. |
| **Tokenizer (`regex_v4`), Diagnostics & Train-Only Vocab Builder** | `[x] Done` | `obs_codecs/diagnostics.py`, `obs_codecs/vocab.py`, `scripts/build_vocab.py`, `scripts/compute_token_rates_train.py`, `obs_codecs/token_rates_train_v4.json`, `tests/test_diagnostics.py`, `tests/test_vocab.py` | Locked `regex_v4` splits numeric spans into character tokens (`0`–`9`, `-`, `.`), keeping vocab bounded (30–42 tokens). Shared B-family vocab includes both `→` and `|`. |
| **Raw Dataset Split Generation & HDF5 Persistence** | `[~] Partial/Stub` | `configs/dataset_v0.py`, `data/generate_episodes.py`, `data/hdf5_episodes.py`, `scripts/generate_dataset.py`, `tests/test_generate_episodes.py`, `tests/test_generate_dataset.py` | Streaming HDF5 writer + SHA-256 manifest complete for `train`, `val`, and `ood_partial_obs`, plus `generate_intervention_episode` and `generate_counterfactual_pair` helpers. However: (1) `intervention_fraction != 0.0` in `generate_split()` raises `NotImplementedError("reserved for Prompt B2")`; (2) `ood_noise`, `ood_confound`, and `ood_regime_switch` splits are deferred (`docs/PROGRAM_SPEC.md`); (3) no standalone CLI script generates the separate intervention/counterfactual evaluation datasets (`docs/EXPERIMENT_PLAN.md` §9). |
| **Offline Codec String Encoding (HDF5 $\rightarrow$ Parquet)** | `[x] Done` | `data/encode_episodes.py`, `scripts/encode_dataset.py`, `tests/test_encode_dataset.py` | Encodes all 7 variants (`A`, `B`, `C`, `D`, `B_shuffle`, `B_reverse`, `B_bag`) into Parquet with mandatory 1,000-sample parity validation (aborts if pass rate $< 100\%$). |
| **Shared Decoder-Only Transformer & Deterministic Init** | `[x] Done` | `model/architecture.py`, `model/init.py`, `model/README.md`, `tests/test_model_init.py`, `tests/test_real_a_document_masking_integration.py` | Locked `dec_only_v0_L1024_d384_l6_h6_ff1536_vpad48` (11,056,896 params across all codecs). Derives causal + `segment_ids` document allow mask inside `forward()`; trunk shares `model_init_seed`, while `tok_emb` and `lm_head` use `derive_codec_init_seed`. |
| **Document-Masked Packer ($L = 1024$) & Step Accounting** | `[~] Partial/Stub` | `train/packer.py`, `train/loop.py`, `tests/test_packer.py`, `tests/test_packing_step_accounting.py` | Correctly wraps episodes in `<bos>`/`<eos>`, emits parallel `segment_ids`, pads tail with `PAD_SEGMENT_ID = -1`, and trains partial final batches (`ceil(windows / batch_size)`). **Gap:** Materializes all packed windows in memory at once (`docs/TRAINING_RUNNER_CONTRACT.md` §8 notes a streaming packer is needed for `full_v0`). |
| **Stage-1 Training Loop & Matrix Runner** | `[x] Done` | `train/config.py`, `train/data.py`, `train/loop.py`, `train/matrix.py`, `scripts/run_training_matrix.py`, `scripts/verify_tier_artifacts.py`, `tests/test_train_smoke.py`, `tests/test_matrix_runner.py`, `tests/test_loss_target_policy.py` | Implements `RunConfig`, exact `Decimal` budget derivation (`tier0`, `tier1`, `full_v0`), preflight gates, `fsync` + `os.replace` atomic `manifest.json`, resume skip logic, and `reproducibility.json` logging. |
| **Trained-Model Evaluation Suite (8 Tasks) & `metrics_eval.json` Producer** | `[ ] Not Started` | `eval/` (missing model evaluation modules) | `docs/EXPERIMENT_PLAN.md` §7 tasks 1–8 (next-obs MSE, learned regime classification, intervention accuracy, counterfactual twins, ECE/Brier calibration, partial-obs stress, OOD regime, cross-codec symbol-aligned transfer) and `docs/TRAINING_RUNNER_CONTRACT.md` §4.3.1 `metrics_eval.json` are completely unbuilt. |
| **Cross-Replication Aggregation (`summary.json`)** | `[ ] Not Started` | `train/matrix.py` (explicitly deferred in docstring) | `docs/TRAINING_RUNNER_CONTRACT.md` §5 (`median` + `IQR` via `numpy.percentile(..., method="linear")` written to `artifacts/runs/aggregates/{match_regime}/{eval_split}/summary.json`) is not implemented because `metrics_eval.json` does not exist yet. |
| **Post-Hoc Frozen Linear Probes & RSA (CKA / SVCCA / Procrustes)** | `[ ] Not Started` | `probes/` (directory missing) | `docs/EXPERIMENT_PLAN.md` §6 / §7 / §10 and `docs/PROGRAM_SPEC.md` §2 / §4.1 specify `probes/` for frozen activation extraction, linear probe training, and 10k-state CKA/SVCCA/orthogonal Procrustes representation similarity analysis. Directory does not exist. |

---

## 4. Invariants Verification Matrix (`I1`–`I15`)

Every invariant from `docs/INVARIANTS.md` was cross-checked against the actual implementation and test suite on disk:

| ID | Invariant Rule | Verification Status | Enforcing Code & Automated Tests | Audit Findings / Residual Risks |
| :--- | :--- | :---: | :--- | :--- |
| **I1** | All codecs consume exactly `Observation`. | **Verified (Enforced)** | `world/schema.py:observe`, `obs_codecs/encode_{a,b,c,d}.py`, `tests/test_information_contract.py`, `tests/test_codec_{b,c,d}.py` | All `encode_*` functions type-check and accept only `Observation`. Static signature and module attribute checks verify `State`, `LatentState`, and `Intervention` are absent from codec modules. |
| **I2** | No training component may read hidden state. | **Verified (Enforced)** | `obs_codecs/`, `train/`, `tests/test_no_latent_leakage_package_wide.py`, `tests/test_regime_baseline.py` | Package-wide AST test verifies `obs_codecs/` never imports `LatentState`, `CausalRegime`, or `oracle_regime_posterior`. `test_regime_baseline.py` checks `train/` never imports `regime_baseline`. *(Note: no dedicated AST test scans `train/` for `world.schema.LatentState`, though `train/` does not import `world` at all.)* |
| **I3** | World seed determines episode exactly. | **Verified (Enforced)** | `world/simulate.py:simulate`, `data/generate_episodes.py:episode_seed`, `obs_codecs/transforms_e.py:derive_transform_seed`, `tests/test_simulate.py`, `tests/test_generate_dataset.py` | Uses local `np.random.Generator(np.random.PCG64(seed))` with fixed per-step draw order and `blake2b_u64` for transform/init seeds (never Python `hash()`). Verified across fresh subprocesses. |
| **I4** | Dataset split is identical across codecs. | **Verified (Enforced)** | `configs/dataset_v0.py`, `scripts/generate_dataset.py`, `data/encode_episodes.py`, `tests/test_encode_dataset.py` | Raw `State` splits (`train.h5`, `val.h5`, `ood_partial_obs.h5`) are generated once with disjoint base seeds (`1_000_000`, `2_000_000`, `3_000_000`) and encoded into per-codec Parquet files. |
| **I5** | Model architecture is identical. | **Verified (Enforced)** | `model/architecture.py:TransformerConfig`, `model/init.py`, `tests/test_model_init.py:test_param_count_parity_across_codecs` | `SHARED_VOCAB_SIZE = 48` pads all codecs' embedding and untied LM head tables so every codec has exactly `11,056,896` parameters and identical trunk initial weights per replication. |
| **I6** | Optimizer configuration is identical. | **Verified (Enforced)** | `train/config.py` (`OPTIMIZER_ID = "adamw_v0_lr3e4_wd0.1_bs8"`), `train/loop.py:run_training` | Fixed `AdamW(lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))` and `BATCH_SIZE = 8` validated by `validate_run_config()` before every run. |
| **I7** | Evaluation uses ground-truth world state independently of codec. | **Partial / Unbuilt** | `eval/regime_baseline.py` (calibration baseline only) | Respected in `eval/regime_baseline.py`, which scores against `state.latent.regime`. However, **trained-model evaluation is not yet implemented**, so `I7` must be enforced when building `eval/` tasks 1–8. |
| **I8** | C receives no oracle probabilities or causal labels. | **Verified (Enforced)** | `obs_codecs/encode_c.py`, `tests/test_codec_c.py` | `encode_C` emits a static, state-independent string literal `alts: {common_cause, a_causes_b, b_causes_a, spurious}` and contains zero `p=` or posterior fields. Tested for byte-identity across all regimes and `hidden_c` values. |
| **I9** | A contains no explicit event-chain structure. | **Verified (Enforced)** | `obs_codecs/encode_a.py`, `tests/test_codec_a.py` | Essentialist entity/attribute lines plus constant `Relation: A and B are similar.`; tests assert `"→" not in text` and `"hits" not in text.lower()`. |
| **I10** | B preserves event ordering. | **Verified (Enforced)** | `obs_codecs/encode_b.py`, `tests/test_codec_b.py` | Emits `(observe A=...) → (co-vary B=...) → (meta ...)` in fixed intra-step order and chronological timestep order. |
| **I11** | E destroys B organization while preserving token inventory as closely as practical. | **Verified (Enforced)** | `obs_codecs/transforms_e.py`, `tests/test_transforms_e.py` | `shuffle_b` and `reverse_b` preserve exact `regex_v4` token multisets; `bag_b` replaces `→` 1:1 with `|`. Unordered parser `observation_from_B_unordered` recovers 100% of `FactRecord`s. |
| **I12** | Probes are trained only after model freezing. | **Half-Enforced (Stage-1 Clean; Probes Unbuilt)** | `model/architecture.py:DecoderOnlyLM`, `train/loop.py` | Stage-1 training has no auxiliary latent head (pure next-token NLL), satisfying the stage-1 half of `I12`. The post-hoc `probes/` package itself is not yet built. |
| **I13** | Every reported result is tied to a git commit. | **Verified (Enforced)** | `train/loop.py:git_commit`, `train/matrix.py:build_manifest`, `obs_codecs/vocab.py`, `scripts/verify_tier_artifacts.py` | Recorded in `reproducibility.json`, `manifest.json`, `vocab_*.json`, and `token_rates_train_v4.json`. *(Minor note: does not record a dirty-tree flag as suggested in `docs/INVARIANTS.md` §I13).* |
| **I14** | All primary comparisons use multiple training seeds. | **Partial (Runner Supports $R \ge 3$; Aggregation Unbuilt)** | `scripts/run_training_matrix.py`, `train/matrix.py:expand_matrix` | Matrix runner supports `--replications` (default `3`), pairs `model_init_seed` and `data_shuffle_seed` per replication, and checks seed pairing in preflight. However, `--replications 1` is permitted by `expand_matrix` (`if n_replications < 1` check), and §5 median/IQR aggregation is not yet implemented. |
| **I15** | Methodological changes require explicit approval. | **Verified (Process & Docs)** | `docs/COMPUTE_MATCH_BUDGETS.md`, `model/README.md`, `tests/test_loss_target_policy.py` | Locked decisions (`regex_v4`, `L=1024`, `vpad=48`, `all_non_pad` loss target policy) are documented with explicit trade-off analyses and regression tests. |

---

## 5. Spec vs. Code Discrepancies & Invariant Risks

1. **Direct `run_training()` Pool Truncation & Silent Synthetic Fallback (`train/loop.py` vs. `docs/TRAINING_RUNNER_CONTRACT.md` §2.2.1 & §4.4.4):**
   - **Location:** `train/loop.py:run_training()` (pool loading, ~lines 253–269) vs. `scripts/run_training_matrix.py:_preflight()` (~line 101).
   - **Issue:** When `run_training_matrix.py` runs, `_preflight()` calls `episodes_from_parquet(parquet, vocab, max_episodes=profile.pool_episodes)` and enforces `len(pool) == profile.pool_episodes`, then injects `episode_pool` into `run_training()`. However, if `run_training(config)` is called directly with `episode_pool=None` (e.g., from a custom script or single-run runner):
     1. If `dataset_path` exists, it calls `episodes_from_parquet(dataset_path, vocab)` **without** `max_episodes=profile.pool_episodes`. A `tier0` config (`pool_episodes=200`) run against a 2,000-episode Parquet file would therefore sample from all 2,000 episodes instead of the locked 200-episode pool.
     2. If `dataset_path` does **not** exist, `run_training()` silently falls back to `synthetic_episodes(...)` instead of raising `FileNotFoundError`.
2. **In-Memory Full-Stream Packing Bottleneck for `full_v0` (`train/loop.py` vs. `docs/TRAINING_RUNNER_CONTRACT.md` §8):**
   - **Location:** `train/loop.py:run_training()` (the `pack_episodes(...)` call, ~line 306) and `train/packer.py:pack_episodes`.
   - **Issue:** `run_training()` materializes the entire multi-epoch presentation list `presented` (up to `78,770` episodes for Codec B in `full_v0`) and packs all `~108.4M` tokens (`~105.9k` windows of length `1024`) into a Python `list[PackedWindow]` of tensors before step 1. While acknowledged in `docs/TRAINING_RUNNER_CONTRACT.md` §8, this is a concrete blocker for running `full_v0` efficiently without a streaming/iterable packer.
3. **Matrix Runner Default Replications (`R=3` in Code vs. `R=5` in Contract):**
   - **Location:** `scripts/run_training_matrix.py` (line 157: `parser.add_argument("--replications", type=int, default=3)`) vs. `docs/TRAINING_RUNNER_CONTRACT.md` §1 & §4.1 (`Default for full v0 reporting: R = 5` / `n_replications: 3–5 (default 5)`).
   - **Issue:** Running `scripts/run_training_matrix.py --budget-profile full_v0` without `--replications 5` defaults to `3` replications instead of the contract's default of `5`. Furthermore, `train/matrix.py:expand_matrix` only rejects `n_replications < 1`, so `full_v0` does not enforce `n_replications >= 3` (Invariant `I14`).
4. **Stale `obs_codecs/diagnostics_A.json` Artifact (`regex_v0` vs. Locked `regex_v4`):**
   - **Location:** `obs_codecs/diagnostics_A.json`.
   - **Issue:** `obs_codecs/diagnostics_A.json` on disk still contains output from `regex_v0: ident|number|:=,. ; whitespace-separated` (`avg_tokens_per_observation: 52.0`, `vocab_size: 50` with unsplit float tokens like `"0.41"`), whereas `obs_codecs/diagnostics_comparison_ABCD.json` and `obs_codecs/token_rates_train_v4.json` were updated to `regex_v4` (`avg_tokens_per_observation: 54.156` on the 500-state parity fixture, `vocab_size: 41`). Re-running `scripts/run_codec_a_diagnostics.py` would overwrite `diagnostics_A.json` with `regex_v4` numbers.
5. **Unused Dependencies & Storage Stack Drift (`pyproject.toml` vs. `docs/TECH_STACK.md`):**
   - **Location:** `pyproject.toml`, `docs/TECH_STACK.md`, `data/hdf5_episodes.py`, `data/encode_episodes.py`.
   - **Issue:** `docs/TECH_STACK.md` lists `zarr` for array storage and `pydantic` and/or `dataclasses` for data models, and omits `h5py` and `pyarrow`. In reality, the codebase uses `dataclasses`, `h5py` (HDF5), and `pyarrow` (Parquet) exclusively; `zarr`, `pydantic`, `scipy`, `matplotlib`, and `pandas` (apart from `to_parquet` with the `pyarrow` engine) are listed in `pyproject.toml` dependencies but are never imported anywhere in the repository.
6. **Rule Hygiene Drift in `.cursor/` (RESOLVED in this revision):**
   - **Location:** `.cursor/skills/codec-audit/SKILL.md` (line 13) and `.cursor/skills/build-experiment/SKILL.md` (line 59). Also, `.cursor/rules/03-python-conventions.mdc` still lists `zarr`/`pydantic` as the stack (same drift as `docs/TECH_STACK.md`, item 5).
   - **Issue (fixed):** `.cursor/skills/codec-audit/SKILL.md` contained Polish text (a "Goal" line written in Polish), violating `.cursor/rules/04-english-only.mdc`. `.cursor/skills/build-experiment/SKILL.md` said `Typed Python 3.12`, conflicting with `Python 3.13.15` in `docs/TECH_STACK.md`, `.python-version`, and `pyproject.toml`.
7. **Missing Dirty-Tree Flag in `reproducibility.json` (`docs/INVARIANTS.md` §I13):**
   - **Location:** `train/loop.py:git_commit()`.
   - **Issue:** `docs/INVARIANTS.md` §I13 states *"Every reported metric bundle records the git commit (and dirty-tree flag if any)."* `train/loop.py:git_commit()` runs `git rev-parse HEAD` but does not record whether `git status --porcelain` was dirty at run time.

---

## 6. Experiment Readiness (`docs/EXPERIMENT_PLAN.md` §11 & §13)

### §11 Readiness Checklist (10 Gates Before Full Scientific Matrix)

- [x] **1. One seed fully reproduces an episode:** Verified in `world/simulate.py` and `tests/test_simulate.py` (`PCG64(seed)`).
- [x] **2. A–E pass info-equality tests on $O_t$ (`parse_k ∘ encode_k = F`, shared quantization):** Verified in `obs_codecs/parity_audit.py`, `tests/test_parity_audit.py`, and `tests/test_transforms_e.py`.
- [x] **3. C has no oracle $p$ / true causes in training inputs:** Verified in `obs_codecs/encode_c.py` and `tests/test_codec_c.py`.
- [x] **4. A is essentialist (not eventive); ambient time contract documented:** Verified in `obs_codecs/encode_a.py` and `docs/PROGRAM_SPEC.md` (`A-ordered` implemented; optional `A-bag` not built).
- [x] **5. E (shuffle/reverse/bag) exists:** Verified in `obs_codecs/transforms_e.py` (`B_shuffle`, `B_reverse`, `B_bag`).
- [x] **6. Both match regimes are implemented and logged:** Verified in `train/config.py`, `train/loop.py`, and `train/matrix.py` (`compute_matched` and `info_matched`).
- [~] **7. $\ge 1$ task where a priori $B > A$ or $C > A$ is expected:** World v0 dynamics and intervention asymmetry (`do(A)` vs `do(B)`) are built in `world/simulate.py` and `data/generate_episodes.py`, **but the model evaluation harness to score trained checkpoints on these tasks is not implemented.**
- [ ] **8. Evaluation does not require fluent natural language:** Ground-truth state/intervention space evaluation is specified in `docs/EXPERIMENT_PLAN.md` §7, **but no trained-model evaluator exists in `eval/` yet.**
- [x] **9. Multi-seed runner works from one script:** Verified in `scripts/run_training_matrix.py` and `tests/test_matrix_runner.py`.
- [x] **10. Difficulty pilot on D / O-only heuristic locked regime-accuracy $\in [\sim 0.55, \sim 0.80]$:** Locked at `DEFAULT_DIFFICULTY_V0` (`noise_scale=0.15, partial_obs_rate=0.2, confounding_strength=0.0, regime_switch_rate=0.0`) with heuristic accuracy `0.72` in `eval/difficulty_sweep_noise015_partial.json`.

### §13 Research Stage Execution Status

| Stage (`docs/EXPERIMENT_PLAN.md` §13) | Current Status | Runnable Today? | Blocking Dependencies |
| :--- | :---: | :---: | :--- |
| **1. Spec: `State` / `Obs` / causal regimes + fact schema $F$** | `[x] Complete` | **Yes** (`uv run pytest tests/test_world_schema.py tests/test_fact_record.py`) | None |
| **2. Simulator + interventions + difficulty knobs** | `[x] Complete` | **Yes** (`uv run pytest tests/test_simulate*.py`) | None (`regime_switch_rate > 0` deferred per v0 spec) |
| **3. Difficulty pilot (lock operating point)** | `[x] Complete` | **Yes** (`uv run python scripts/run_regime_difficulty_sweep.py`) | None (`DEFAULT_DIFFICULTY_V0` locked) |
| **4. Codecs A–E + info-equality tests (`parse∘encode`)** | `[x] Complete` | **Yes** (`uv run python scripts/run_parity_audit.py --codec A`) | None (optional `A-bag` / `C-shuffle` deferred) |
| **5. Dataset + sanity checks + vocab build** | `[x] Complete` | **Yes** (`scripts/generate_dataset.py` $\rightarrow$ `encode_dataset.py` $\rightarrow$ `build_vocab.py`) | None (requires running CLI scripts to populate gitignored `artifacts/`) |
| **6. Train baselines (2 regimes $\times$ seeds)** | `[~] Pilot-Ready` | **Yes for `tier0`/`tier1`** (`scripts/run_training_matrix.py`) | `full_v0` needs 50k dataset generation + streaming packer (`train/packer.py`) to avoid high RAM usage |
| **7. Eval suite (Tasks 1–8) + cross-codec transfer + §5 aggregation** | `[ ] Blocked` | **No** | Missing `eval/` model evaluation modules, `metrics_eval.json` producer, and `train/matrix.py` §5 aggregator |
| **8. Frozen linear probes + RSA (CKA / SVCCA / Procrustes)** | `[ ] Blocked` | **No** | Missing `probes/` package and activation extraction pipeline |

---

## 7. Prioritized Next Steps (Actionable Backlog)

1. **Implement the Trained-Model Evaluation Suite & `metrics_eval.json` Producer (`eval/`)**
   - *Governing Specs:* `docs/EXPERIMENT_PLAN.md` §7, `docs/PROGRAM_SPEC.md` §2 (Evaluator & Interventions), `docs/TRAINING_RUNNER_CONTRACT.md` §2.6 & §4.3.1, `docs/INVARIANTS.md` (`I7`, `I12`).
   - *Action:* Build the evaluation harness in `eval/` that loads `checkpoints/final.pt` (refusing non-final checkpoints per §2.6) and evaluates trained models on `val` and `ood_partial_obs` across the primary world-state tasks (Task 1: next-observation MSE; Task 2: regime classification; Task 3: `do(A)`/`do(B)` intervention accuracy; Task 4: counterfactual twins using `generate_counterfactual_pair`; Task 5: ECE/Brier calibration; Task 6/7: partial-observability & OOD splits), writing split-keyed `metrics_eval.json` into each run directory.
2. **Implement Cross-Replication Median/IQR Aggregation (`train/matrix.py` & `scripts/run_training_matrix.py`)**
   - *Governing Specs:* `docs/TRAINING_RUNNER_CONTRACT.md` §5 (`§5.1`–`§5.3`), `docs/EXPERIMENT_PLAN.md` §5, `docs/INVARIANTS.md` (`I13`, `I14`).
   - *Action:* Once `metrics_eval.json` is produced, implement §5 aggregation computing `median`, `q1`, and `q3` via `numpy.percentile(..., method="linear")` across replications for `full_v0` runs (explicitly excluding `tier0`/`tier1` `plumbing_only` runs) and writing `artifacts/runs/aggregates/{match_regime}/{eval_split}/summary.json`. Also align `scripts/run_training_matrix.py` `--replications` default with `docs/TRAINING_RUNNER_CONTRACT.md` (`R=5` or enforce `R >= 3` for `full_v0`).
3. **Harden `train/loop.py` Pool Truncation & Replace In-Memory Packing with a Streaming Packer (`train/packer.py`, `train/loop.py`)**
   - *Governing Specs:* `docs/TRAINING_RUNNER_CONTRACT.md` §2.2.1, §4.4.4, §8; `docs/COMPUTE_MATCH_BUDGETS.md` §4.
   - *Action:* (a) In `train/loop.py:run_training()`, pass `max_episodes=get_budget_profile(config.budget_profile).pool_episodes` when loading from Parquet directly and require an explicit flag before falling back to `synthetic_episodes`; (b) add an iterable/streaming window packer in `train/packer.py` so `full_v0` (`108.4M` token-updates) streams batches without allocating ~106k `PackedWindow` tensors in memory at once; (c) add a `git_dirty` boolean to `reproducibility.json` (`I13`).
4. **Build the Post-Hoc Frozen Linear Probes & RSA Pipeline (`probes/`)**
   - *Governing Specs:* `docs/EXPERIMENT_PLAN.md` §6 (Probes) & §7 (RSA), `docs/PROGRAM_SPEC.md` §2 & §4.1, `docs/INVARIANTS.md` (`I12`).
   - *Action:* Create the `probes/` package that loads frozen `checkpoints/final.pt` (`requires_grad_(False)`, `model.eval()`), extracts layer activations over a shared ~10k world-state fixture across codecs `A`–`D` and `E`, trains identical linear probes for `hidden_c`, `CausalRegime`, and next-observation targets, and computes CKA, SVCCA, and orthogonal Procrustes similarity matrices.
5. **Implement Cross-Codec Transfer (Task 8) & Optional Controls (`A-bag`, `C-shuffle`, Unlocked OOD Axes)**
   - *Governing Specs:* `docs/EXPERIMENT_PLAN.md` §2.1, §3, §7 (Task 8); `docs/PROGRAM_SPEC.md` (Dataset splits).
   - *Action:* Add symbol-aligned zero/few-shot cross-codec evaluation (`B -> A`, `A -> B`), optional episode-level `A-bag` and field-level `C-shuffle` controls, and empirical sweeps for `noise_scale > 0.3` and `confounding_strength > 0.3` to unlock `ood_noise` and `ood_confound`.
