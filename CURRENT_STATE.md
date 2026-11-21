# Current State & Reality-vs-Documentation Audit (`codec-world-models`)

## 1. Audit Metadata

| Field | Value |
| :--- | :--- |
| **Audit Date** | `2026-11-21` |
| **Repository Snapshot** | Live working tree (`master`) |
| **Last Recorded Git Commit** | `c18fa3f` (`Record dataset provenance and close audit gaps in tests and artifact checks.`). |
| **Target Branch / Version** | `master` / `v0.1.0` (`pyproject.toml`, Python `3.13.15` in `.python-version`) |
| **Governing Specifications** | `docs/PROGRAM_SPEC.md`, `docs/EXPERIMENT_PLAN.md`, `docs/COMPUTE_MATCH_BUDGETS.md`, `docs/TRAINING_RUNNER_CONTRACT.md`, `docs/INVARIANTS.md`, `docs/TECH_STACK.md` |

---

## 2. Executive Summary

- **World v0 Simulator, Information Contract & Codecs (`A`–`D` + `B` E-transforms) are 100% Built & Tested:** `world/schema.py`, `world/simulate.py`, and `obs_codecs/` cleanly enforce $O_t = \mathrm{Obs}(S_t)$ with zero latent or intervention leakage, deterministic `PCG64(seed)` trajectories, locked `DEFAULT_DIFFICULTY_V0`, hand-written `parse_k` parsers satisfying `parse_k(encode_k(O)) == F(O)` on shared quantization bins, and locked `regex_v4` digit-level tokenization (`obs_codecs/diagnostics.py`, `obs_codecs/vocab.py`).
- **Offline Data Pipeline & Stage-1 Training Infrastructure are Production-Ready:** `data/generate_episodes.py`, `data/hdf5_episodes.py`, and `data/encode_episodes.py` generate streaming HDF5 splits and validated Parquet string corpora; `model/architecture.py`, `model/init.py`, `train/packer.py`, `train/device.py`, `train/loop.py`, and `train/matrix.py` implement the shared 11.06M-parameter decoder-only Transformer (`L=1024`, `shared_vocab_size=48`), `segment_ids`-driven boolean SDPA document masking, streaming batch packing for `full_v0`, ROCm `bf16_mixed` autocast + `torch.compile`, exact `Decimal` budget derivation (`tier0`, `tier1`, `full_v0`), `allow_synthetic=False` + exact pool-size enforcement on direct `run_training()` calls, and atomic `manifest.json` resume semantics.
- **Trained-Model Evaluation, §5 Aggregation, Reporting & Post-Hoc Probes are Implemented:** `eval/checkpoint.py`, `eval/decode.py`, `eval/metrics.py`, `eval/score.py`, `scripts/generate_eval_splits.py`, and `scripts/evaluate_runs.py` implement frozen `fp32` scoring (`eval_v2`, with training-distribution intervention sampling, `parse_fail_rate`, and schema/decoder metadata) and write `metrics_eval.json` and contract §5 median/IQR `summary.json` aggregates (enforcing `scoring_checkpoint == "final"` and $\ge 3$ replications). `eval/report.py` and `scripts/summarize_results.py` render summary tables, scientific verdicts, and headless matplotlib figures. `probes/linear.py`, `probes/rsa.py`, and `scripts/run_probes.py` implement frozen final-layer held-out ridge probes (`PROBE_RIDGE_ALPHA = 1.0`, episode-level 50/50 split) and CKA/SVCCA/Procrustes RSA.
- **Scope & Deferred Items:** (1) Stage 7 v0 evaluation uses shared sample caps (`EVAL_MAX_EPISODES = 64`, `PROBE_TRAIN_EPISODES = 128`, `EVAL_SUITE_EPISODES = 32`), keeps `ood_regime_accuracy = null`, defers multi-point `noise_scale` curves, and evaluates Task 8 (`transfer_aligned_next_state_mse`) on `A <-> B` surface-token alignment only; (2) optional controls (`A-bag`, `C-shuffle`) and unlocked OOD splits (`ood_noise`, `ood_confound`, `ood_regime_switch`) remain deferred per `docs/PROGRAM_SPEC.md`.
- **Test Coverage is High (32 Pytest Suites):** All implemented invariants (`I1`–`I15`) are guarded by automated AST inspection (covering `obs_codecs/`, `train/`, and `model/`), byte-identity checks, eval/report tests, probe tests, and integration tests under `tests/`.

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
| **Raw Dataset Split Generation & HDF5 Persistence** | `[x] Done` | `configs/dataset_v0.py`, `data/generate_episodes.py`, `data/hdf5_episodes.py`, `scripts/generate_dataset.py`, `scripts/generate_eval_splits.py`, `tests/test_generate_episodes.py`, `tests/test_generate_dataset.py` | Streaming HDF5 writer + SHA-256 manifest complete for `train`, `val`, `ood_partial_obs`, plus `scripts/generate_eval_splits.py` for intervention/counterfactual eval splits (`eval_schema_version`). `configs/dataset_v0.py` inventories deferred `SplitSpec` rows for `ood_noise`, `ood_confound`, `ood_regime_switch` (not generated). |
| **Offline Codec String Encoding (HDF5 $\rightarrow$ Parquet)** | `[x] Done` | `data/encode_episodes.py`, `scripts/encode_dataset.py`, `tests/test_encode_dataset.py` | Encodes all 7 variants (`A`, `B`, `C`, `D`, `B_shuffle`, `B_reverse`, `B_bag`) into Parquet with mandatory 1,000-sample parity validation (aborts if pass rate $< 100\%$). |
| **Shared Decoder-Only Transformer & Deterministic Init** | `[x] Done` | `model/architecture.py`, `model/init.py`, `model/README.md`, `tests/test_model_init.py`, `tests/test_real_a_document_masking_integration.py` | Locked `dec_only_v0_L1024_d384_l6_h6_ff1536_vpad48` (11,056,896 params across all codecs). Derives causal + `segment_ids` boolean SDPA mask inside `forward()`; trunk shares `model_init_seed`, while `tok_emb` and `lm_head` use `derive_codec_init_seed`. |
| **Document-Masked Packer ($L = 1024$) & Step Accounting** | `[x] Done` | `train/packer.py`, `train/loop.py`, `tests/test_packer.py`, `tests/test_packing_step_accounting.py`, `tests/test_gpu_optimizations_and_checkpoint_compat.py` | Wraps episodes in `<bos>`/`<eos>`, emits parallel `segment_ids`, pads tail with `PAD_SEGMENT_ID = -1`, trains partial final batches (`ceil(windows / batch_size)`), and supports streaming batch packing for `full_v0`. |
| **Stage-1 Training Loop & Matrix Runner** | `[x] Done` | `train/config.py`, `train/data.py`, `train/device.py`, `train/loop.py`, `train/matrix.py`, `scripts/run_training_matrix.py`, `scripts/verify_tier_artifacts.py`, `tests/test_train_smoke.py`, `tests/test_matrix_runner.py`, `tests/test_loss_target_policy.py`, `tests/test_verify_tier_artifacts.py` | Implements `RunConfig`, exact `Decimal` budget derivation (`tier0`, `tier1`, `full_v0`), ROCm `bf16_mixed` + `torch.compile`, preflight gates, `fsync` + `os.replace` atomic `manifest.json` (`eval_status="pending"`), resume skip logic, and `reproducibility.json` (`git_dirty`, `dataset_base_seed`, `schema_version`, `loss_target_policy`). |
| **Trained-Model Evaluation Suite & `metrics_eval.json` Producer** | `[x] Done` (v0 scope) | `eval/checkpoint.py`, `eval/decode.py`, `eval/metrics.py`, `eval/score.py`, `scripts/evaluate_runs.py`, `tests/test_device_and_eval.py` | Enforces `checkpoints/final.pt` only in frozen `fp32`, scores passive splits, interventions (`eval_v2` training-distribution sampling), counterfactual twins, and `A <-> B` surface-token transfer (`null` for `C`/`D`/`E`). Uses locked caps (`EVAL_MAX_EPISODES=64`, `PROBE_TRAIN_EPISODES=128`, `EVAL_SUITE_EPISODES=32`); `ood_regime_accuracy` is `null` (3-point `noise_scale` curves and `ood_noise`/`ood_confound`/`ood_regime_switch` deferred). |
| **Cross-Replication Aggregation (`summary.json`) & Report Generator** | `[x] Done` | `train/matrix.py`, `eval/report.py`, `scripts/evaluate_runs.py`, `scripts/summarize_results.py`, `tests/test_summarize_results.py` | Implements §5 `median` + `IQR` (`numpy.percentile(..., method="linear")`), refusing `scoring_checkpoint != "final"` (§2.6) and `< 3` replications (`I14`, unless `--allow-pilot`); `summary.json` / `probes.json` inherit training `git_commit`/`git_dirty` from `manifest.json` while recording `eval_git_commit`/`probe_git_commit`. |
| **Post-Hoc Frozen Linear Probes & RSA (CKA / SVCCA / Procrustes)** | `[x] Done` | `probes/linear.py`, `probes/rsa.py`, `scripts/run_probes.py`, `tests/test_run_probes.py`, `tests/test_rsa.py` | Loads frozen `checkpoints/final.pt`, extracts final-layer hidden states at the last token of framed episode prefixes `states[0..t]` (original `t` preserved), fits held-out ridge probes (`PROBE_RIDGE_ALPHA = 1.0`, deterministic episode-level 50/50 split with train-only standardization) for `hidden_c_r2`, `next_b_r2` (`state_{t+1}.b`), and `regime_accuracy` (ridge-to-one-hot + softmax), plus CKA/SVCCA/Procrustes RSA. Supports single-cell and `--matrix` modes. |

---

## 4. Invariants Verification Matrix (`I1`–`I15`)

Every invariant from `docs/INVARIANTS.md` was cross-checked against the actual implementation and test suite on disk:

| ID | Invariant Rule | Verification Status | Enforcing Code & Automated Tests | Audit Findings / Residual Risks |
| :--- | :--- | :---: | :--- | :--- |
| **I1** | All codecs consume exactly `Observation`. | **Verified (Enforced)** | `world/schema.py:observe`, `obs_codecs/encode_{a,b,c,d}.py`, `tests/test_information_contract.py`, `tests/test_codec_{b,c,d}.py` | All `encode_*` functions type-check and accept only `Observation`. Static signature and module attribute checks verify `State`, `LatentState`, and `Intervention` are absent from codec modules. |
| **I2** | No training component may read hidden state. | **Verified (Enforced)** | `obs_codecs/`, `train/`, `model/`, `tests/test_no_latent_leakage_package_wide.py`, `tests/test_regime_baseline.py` | Package-wide AST guards verify `obs_codecs/`, `train/`, and `model/` never import `LatentState`, `CausalRegime`, `oracle_regime_posterior`, or `eval`. `test_regime_baseline.py` also checks `train/` never imports `regime_baseline`. |
| **I3** | World seed determines episode exactly. | **Verified (Enforced)** | `world/simulate.py:simulate`, `data/generate_episodes.py:episode_seed`, `obs_codecs/transforms_e.py:derive_transform_seed`, `tests/test_simulate.py`, `tests/test_generate_dataset.py` | Uses local `np.random.Generator(np.random.PCG64(seed))` with fixed per-step draw order and `blake2b_u64` for transform/init seeds (never Python `hash()`). Verified across fresh subprocesses. |
| **I4** | Dataset split is identical across codecs. | **Verified (Enforced)** | `configs/dataset_v0.py`, `scripts/generate_dataset.py`, `data/encode_episodes.py`, `tests/test_encode_dataset.py` | Raw `State` splits (`train.h5`, `val.h5`, `ood_partial_obs.h5`) are generated once with disjoint base seeds (`1_000_000`, `2_000_000`, `3_000_000`) and encoded into per-codec Parquet files. |
| **I5** | Model architecture is identical. | **Verified (Enforced)** | `model/architecture.py:TransformerConfig`, `model/init.py`, `tests/test_model_init.py:test_param_count_parity_across_codecs` | `SHARED_VOCAB_SIZE = 48` pads all codecs' embedding and untied LM head tables so every codec has exactly `11,056,896` parameters and identical trunk initial weights per replication. |
| **I6** | Optimizer configuration is identical. | **Verified (Enforced)** | `train/config.py` (`OPTIMIZER_ID = "adamw_v0_lr3e4_wd0.1_bs8"`), `train/loop.py:run_training` | Fixed `AdamW(lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))` and `BATCH_SIZE = 8` validated by `validate_run_config()` before every run. |
| **I7** | Evaluation uses ground-truth world state independently of codec. | **Verified (Enforced)** | `eval/regime_baseline.py`, `eval/score.py`, `eval/decode.py`, `scripts/evaluate_runs.py`, `tests/test_device_and_eval.py` | Both calibration baselines and frozen `fp32` model evaluation score decoded/predicted outputs against ground-truth `State` / `LatentState` targets independently of codec surface. |
| **I8** | C receives no oracle probabilities or causal labels. | **Verified (Enforced)** | `obs_codecs/encode_c.py`, `tests/test_codec_c.py` | `encode_C` emits a static, state-independent string literal `alts: {common_cause, a_causes_b, b_causes_a, spurious}` and contains zero `p=` or posterior fields. Tested for byte-identity across all regimes and `hidden_c` values. |
| **I9** | A contains no explicit event-chain structure. | **Verified (Enforced)** | `obs_codecs/encode_a.py`, `tests/test_codec_a.py` | Essentialist entity/attribute lines plus constant `Relation: A and B are similar.`; tests assert `"→" not in text` and `"hits" not in text.lower()`. |
| **I10** | B preserves event ordering. | **Verified (Enforced)** | `obs_codecs/encode_b.py`, `tests/test_codec_b.py` | Emits `(observe A=...) → (co-vary B=...) → (meta ...)` in fixed intra-step order and chronological timestep order. |
| **I11** | E destroys B organization while preserving token inventory as closely as practical. | **Verified (Enforced)** | `obs_codecs/transforms_e.py`, `tests/test_transforms_e.py` | `shuffle_b` and `reverse_b` preserve exact `regex_v4` token multisets; `bag_b` replaces `→` 1:1 with `|`. Unordered parser `observation_from_B_unordered` recovers 100% of `FactRecord`s. |
| **I12** | Probes are trained only after model freezing. | **Verified (Enforced)** | `model/architecture.py:DecoderOnlyLM`, `train/loop.py`, `probes/linear.py`, `probes/rsa.py`, `scripts/run_probes.py`, `tests/test_run_probes.py` | Stage-1 training has no auxiliary latent head (pure next-token NLL). `scripts/run_probes.py` loads `checkpoints/final.pt` via `load_scoring_checkpoint` with gradients disabled before fitting held-out ridge probes (`PROBE_RIDGE_ALPHA = 1.0`, episode-level split) or computing RSA. |
| **I13** | Every reported result is tied to a git commit. | **Verified (Enforced)** | `train/loop.py` (`git_commit`, `git_dirty`), `train/matrix.py`, `eval/score.py`, `scripts/evaluate_runs.py`, `scripts/run_probes.py` | Recorded in `reproducibility.json`, `manifest.json`, `vocab_*.json`, `token_rates_train_v4.json`, `metrics_eval.json`, `summary.json` (with `eval_git_commit`/`eval_git_dirty`), and `probes.json` (with `probe_git_commit`/`probe_git_dirty`). |
| **I14** | All primary comparisons use multiple training seeds. | **Verified (Runner + §5 Aggregation)** | `scripts/run_training_matrix.py`, `train/matrix.py` (`expand_matrix`, `aggregate_scientific_metrics`), `scripts/evaluate_runs.py --aggregate` | Matrix runner pairs `model_init_seed` and `data_shuffle_seed` per replication, checks seed pairing in preflight, enforces `min_replications=3` for scientific runs, and §5 aggregation refuses `< 3` replications or `tier0`/`tier1` unless explicitly overridden. |
| **I15** | Methodological changes require explicit approval. | **Verified (Process & Docs)** | `docs/COMPUTE_MATCH_BUDGETS.md`, `model/README.md`, `probes/linear.py`, `tests/test_loss_target_policy.py` | Locked decisions (`regex_v4` 60k-obs prefix rates, `L=1024`, `vpad=48`, `all_non_pad` loss target policy, `PROBE_RIDGE_ALPHA=1.0`, prefix probe semantics) are documented with explicit trade-off analyses and regression tests. |

---

## 5. Spec vs. Code Discrepancies & Invariant Risks

1. **Direct `run_training()` Pool Truncation & Silent Synthetic Fallback (`train/loop.py` vs. `docs/TRAINING_RUNNER_CONTRACT.md` §2.2.1 & §4.4.4 — RESOLVED):**
   - **Location:** `train/loop.py:run_training()`.
   - **Resolution:** `run_training()` now enforces `allow_synthetic: bool = False` (raising `FileNotFoundError` when the Parquet split is missing), passes `max_episodes=profile.pool_episodes` on direct Parquet loads, and asserts `len(episode_pool) == profile.pool_episodes` (tested by `tests/test_device_and_eval.py::test_missing_parquet_does_not_fall_back_to_synthetic`).
2. **In-Memory Full-Stream Packing Bottleneck for `full_v0` (RESOLVED in `3ad608c`):**
   - **Location:** `train/packer.py`, `train/loop.py`.
   - **Resolution:** Streaming batch packing was added alongside ROCm `bf16_mixed` autocast and boolean SDPA, avoiding full `full_v0` tensor materialization in RAM.
3. **Matrix Runner Default Replications (`R=3` vs. `R=5` in Contract — RESOLVED):**
   - **Location:** `scripts/run_training_matrix.py`, `train/matrix.py:expand_matrix`.
   - **Resolution:** `scripts/run_training_matrix.py` defaults `--replications` to `5` and sets `min_replications = 3` for scientific runs (unless `--smoke-max-steps` is active), enforced by `train/matrix.py:expand_matrix` (tested by `tests/test_device_and_eval.py::test_full_v0_min_replications_gate`).
4. **Stale `obs_codecs/diagnostics_A.json` Artifact (`regex_v0` vs. Locked `regex_v4` — RESOLVED):**
   - **Location:** `obs_codecs/diagnostics_A.json`.
   - **Resolution:** `obs_codecs/diagnostics_A.json` is updated to locked `regex_v4` (`avg_tokens_per_observation: 54.156`, `vocab_size: 41`, `n_observations: 500`).
5. **Storage Stack & Optional Extras (`pyproject.toml` vs. `docs/TECH_STACK.md` — RESOLVED):**
   - **Resolution:** `docs/TECH_STACK.md` was updated to match `pyproject.toml` (`h5py`, `pyarrow`, `dataclasses`, `torch`/`triton-rocm`, `matplotlib` in `dev` and `analysis`, `scipy` in `analysis`, and explicitly no `zarr` or `pydantic`).
6. **Rule Hygiene Drift in `.cursor/` (RESOLVED):**
   - **Location:** `.cursor/skills/codec-audit/SKILL.md` (line 13) and `.cursor/skills/build-experiment/SKILL.md` (line 59).
   - **Issue (fixed):** `.cursor/skills/codec-audit/SKILL.md` contained Polish text, violating `.cursor/rules/04-english-only.mdc`, and `.cursor/skills/build-experiment/SKILL.md` said `Typed Python 3.12` instead of `Python 3.13.15`.
7. **Missing Dirty-Tree Flag in `reproducibility.json` (RESOLVED):**
   - **Location:** `train/loop.py:git_dirty()`, `train/matrix.py`, `scripts/evaluate_runs.py`.
   - **Resolution:** `git_dirty` is now recorded in `reproducibility.json`, `summary.json`, and evaluation outputs (`eval_git_dirty`).

---

## 6. Experiment Readiness (`docs/EXPERIMENT_PLAN.md` §11 & §13)

### §11 Readiness Checklist (10 Gates Before Full Scientific Matrix)

- [x] **1. One seed fully reproduces an episode:** Verified in `world/simulate.py` and `tests/test_simulate.py` (`PCG64(seed)`).
- [x] **2. A–E pass info-equality tests on $O_t$ (`parse_k ∘ encode_k = F`, shared quantization):** Verified in `obs_codecs/parity_audit.py`, `tests/test_parity_audit.py`, and `tests/test_transforms_e.py`.
- [x] **3. C has no oracle $p$ / true causes in training inputs:** Verified in `obs_codecs/encode_c.py` and `tests/test_codec_c.py`.
- [x] **4. A is essentialist (not eventive); ambient time contract documented:** Verified in `obs_codecs/encode_a.py` and `docs/PROGRAM_SPEC.md` (`A-ordered` implemented; optional `A-bag` not built).
- [x] **5. E (shuffle/reverse/bag) exists:** Verified in `obs_codecs/transforms_e.py` (`B_shuffle`, `B_reverse`, `B_bag`).
- [x] **6. Both match regimes are implemented and logged:** Verified in `train/config.py`, `train/loop.py`, and `train/matrix.py` (`compute_matched` and `info_matched`).
- [x] **7. $\ge 1$ task where a priori $B > A$ or $C > A$ is expected:** World v0 dynamics, intervention asymmetry (`do(A)` vs `do(B)`), and frozen `fp32` model evaluation (`eval/score.py`, `scripts/evaluate_runs.py`) are implemented.
- [x] **8. Evaluation does not require fluent natural language:** Ground-truth state/intervention space evaluation is implemented in `eval/score.py`, `eval/decode.py`, and `scripts/evaluate_runs.py`.
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
| **6. Train baselines (2 regimes $\times$ seeds)** | `[x] Complete` | **Yes** (`scripts/run_training_matrix.py`) | None (streaming packer + ROCm `bf16_mixed` implemented; requires 50k dataset generation in `artifacts/` for `full_v0`) |
| **7. Eval suite (Tasks 1–8) + cross-codec transfer + §5 aggregation + report** | `[~] v0 Scope` | **Yes** (`scripts/generate_eval_splits.py`, `scripts/evaluate_runs.py`, `scripts/summarize_results.py`) | Implemented with locked sample caps (`EVAL_MAX_EPISODES=64`, `PROBE_TRAIN_EPISODES=128`, `EVAL_SUITE_EPISODES=32`), `ood_regime_accuracy=null` (deferred 3-point `noise_scale` curves and `ood_noise`/`ood_confound`/`ood_regime_switch`), and `A<->B`-only surface-token transfer. |
| **8. Frozen linear probes + RSA (CKA / SVCCA / Procrustes)** | `[x] Complete` | **Yes** (`scripts/run_probes.py --matrix`) | Held-out final-layer prefix ridge probes (`probes/linear.py`, `PROBE_RIDGE_ALPHA=1.0`, episode-level split) + CKA/SVCCA/Procrustes (`probes/rsa.py`). |

---

## 7. Prioritized Next Steps (Actionable Backlog)

1. **Execute `full_v0` Matrix, Evaluation, Probes & Reporting**
   - Generate 50k `v0` dataset, encode Parquet corpora, build vocabs, run `scripts/run_training_matrix.py --budget-profile full_v0 --replications 5`, score via `scripts/evaluate_runs.py --runs-root artifacts/runs/full_v0 --raw-dir artifacts/dataset_v0 --budget-profile full_v0 --aggregate`, probe via `scripts/run_probes.py --runs-root artifacts/runs/full_v0 --raw-dir artifacts/dataset_v0 --budget-profile full_v0 --matrix`, and render figures/verdict via `scripts/summarize_results.py`.
2. **Optional Controls & Unlocked OOD Axes**
   - Add optional episode-level `A-bag` and field-level `C-shuffle` controls, and empirical sweeps for `noise_scale > 0.3` (including 3-point `noise_scale` curves) and `confounding_strength > 0.3` to unlock `ood_noise` and `ood_confound` (`docs/PROGRAM_SPEC.md`).
