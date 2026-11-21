# Verification of the post-`probes_v2` analyses

**Status:** findings and remediation plan only. No training, evaluation, probe, or reporting code was changed to produce this document.

**Sources adjudicated:** [analysis.txt](../analysis.txt), three independent write-ups of the same `full_v0` checkpoints after the `probes_v2` rerun (`GEMINI 4 NEON`, `CLAUDE SONNET 5.5`, `CHATGPT`).

**Scientific specification:** [docs/EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) §5–§8 and [docs/INVARIANTS.md](INVARIANTS.md). Allowed conclusions stop at §8. This document does not add a conclusion the plan does not allow.

## Evidence used

Published artifacts, re-read while writing this document:

- Training manifest `artifacts/runs/full_v0/manifest.json`.
- Eval manifests `artifacts/runs/full_v0_eval_v2/manifest.json` (`decoder_mode=skeleton`) and `artifacts/runs/full_v0_eval_v2_ftd/manifest.json` (`decoder_mode=field_to_delimiter`). Both set `eval_splits_dir` to `artifacts/eval_splits_v2`.
- Aggregates `artifacts/runs/aggregates_eval_v2` and `artifacts/runs/aggregates_eval_v2_ftd`.
- Probes `artifacts/probes/full_v0_v2`.
- Tables and verdicts regenerated with `eval.report.load_report_bundle`, `format_summary_tables`, and `format_verdict_summary`. Every median below is that regeneration, not a retyped number from `analysis.txt`.

Checkpoint re-scores already run on these artifacts (five `final.pt` replications, `field_to_delimiter`, `eval_splits_v2`) are used only where the official `DoAcc` medians they reproduced match the published report. Analytical baselines that do not load a model were recomputed for this document on `artifacts/dataset_v0/val.h5` and `artifacts/eval_splits_v2/intervention.h5`.

Locked provenance:

| Role | Commit | Dirty flag on disk |
|---|---|---|
| Stage-1 training | `eed26bed189081c05991ff3cb86e1f9d52f3a306` | `git_dirty=false` |
| `eval_v2` scoring (both decoders) | `dcdcd31fcc6248672e2344db35460b59af40d1d9` | `eval_git_dirty=true` |
| `probes_v2` | `3345286e991349640356f0500eeb46d1a1b3f507` | `probe_git_dirty=false` |

`format_verdict_summary` prints the training commit, `git_dirty`, the eval commit, and the probe commit. It does not print `eval_git_dirty` or `probe_git_dirty`. Under I13 the eval numbers are tied to a commit and to a dirty eval tree.

All comparisons below keep `compute_matched` and `info_matched` separate. Medians are over `n_replications=5`. An IQR label from `eval/report.py` is a descriptive overlap rule, not a significance test.

---

## A. Findings and claim-by-claim adjudication

### A.1 Result in one pass

The three write-ups agree on the printed medians. Those medians are correct for `decoder_mode=field_to_delimiter` plus `probes_v2`. The interpretations that treat those medians as decoded behavior, as a win for Codec C on interventions, or as evidence that `hidden_c` and next-step `b` are barely represented, do not survive the scoring code.

Three measurement facts change the reading:

1. **`ValRegime` and `OODRegime` never decode tokens.** Both are frozen linear regime probes on the final-layer hidden state (`regime_probe_scores` in `eval/score.py`). Decoder-invariance of those columns is required by the implementation. `rho(valReg, probeReg)` correlates two probes. The report heading “Behavioral vs Probe Alignment” overstates what that correlation is.
2. **The intervention ranking is a three-episode parse asymmetry.** On `intervention.h5`, 3 of 32 episodes are not fully observed (`3/32 = 0.09375`, printed `0.094`). Codec B and Codec D fail those steps under teacher-forced `field_to_delimiter` and succeed on the other 29. On those 29, B, C, and D tie. A constant predictor of `0.0` already scores `12/32 = 0.375`, equal to B’s compute-matched `DoAcc`.
3. **`ProbeC_R2 ≈ 0.05` and `NextB_R2 ≈ 0.65` are properties of the world, not of a weak probe.** `hidden_c` is drawn in every episode and enters the observables only in `COMMON_CAUSE`. Held-out ridge `R²` on that subset alone is about `0.81–0.95` (compute-matched `rep0`). Next-step `b` is conditionally i.i.d. given `(regime, hidden_c)`. Last-value persistence explains `R² = 0.423` of it. The models sit on the running-mean integrator (`R² = 0.667`), under the latent Bayes ceiling (`R² = 0.715`).

What remains, and is allowed by §8 only as a codec comparison:

- In `compute_matched`, the two regime probes rank B and C above A, D, and the permuted controls. The report’s own rule marks B above A on `val.regime_accuracy` as `IQR>`.
- In `info_matched`, that B advantage shrinks to a median edge with overlapping IQR. C’s `ProbeReg` stays higher (`0.584`, IQR `0.026`) at a smaller token budget than A or D.
- Codec D does not match Codec B on the compute-matched regime probes, so the compute-matched gap is not “any key-value record will do.”
- Nothing in the intervention, counterfactual, OOD, or CKA columns supports a process-versus-evidence split of causal reasoning. §8 still forbids reading a codec win as an ontology.

### A.2 The record the write-ups quote

`field_to_delimiter`, median (IQR), five replications. Probe columns are identical for the skeleton bundle because probes do not use the decoder.

`compute_matched`:

| Codec | ParseRate | ValRegime | DoAcc | DoParseFail | CFDivMSE | OODRegime | ProbeC_R2 | ProbeReg | NextB_R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 0.774 (0.002) | 0.688 (0.078) | 0.375 (0.000) | 0.094 (0.000) | 1.07e-04 (4.48e-04) | 0.531 (0.062) | 0.056 (0.020) | 0.617 (0.014) | 0.669 (0.024) |
| C | 0.913 (0.054) | 0.672 (0.016) | 0.438 (0.031) | 0.031 (0.062) | 1.15e-04 (2.99e-05) | 0.594 (0.156) | 0.055 (0.017) | 0.599 (0.043) | 0.658 (0.010) |
| D | 0.771 (0.003) | 0.547 (0.078) | 0.375 (0.062) | 0.094 (0.000) | 1.08e-04 (3.65e-05) | 0.531 (0.031) | 0.042 (0.053) | 0.530 (0.067) | 0.651 (0.048) |
| A | 0.980 (0.003) | 0.562 (0.078) | 0.406 (0.031) | 0.000 (0.000) | 2.17e-03 (6.06e-04) | 0.500 (0.047) | 0.051 (0.015) | 0.523 (0.022) | 0.650 (0.017) |
| B_reverse | 0.979 (0.004) | 0.594 (0.141) | 0.438 (0.031) | 0.000 (0.000) | 1.06e-04 (2.40e-06) | 0.484 (0.016) | 0.066 (0.015) | 0.549 (0.058) | 0.648 (0.013) |
| B_bag | 0.152 (0.002) | 0.531 (0.047) | 0.062 (0.125) | 0.938 (0.156) | 1.87e-04 (9.39e-05) | 0.531 (0.031) | 0.019 (0.037) | 0.502 (0.028) | 0.652 (0.019) |
| B_shuffle | 0.167 (0.008) | 0.516 (0.078) | 0.031 (0.062) | 0.844 (0.062) | 2.10e-04 (1.47e-04) | 0.453 (0.016) | 0.029 (0.034) | 0.508 (0.030) | 0.647 (0.010) |

`info_matched` (same probe columns, decoder-dependent columns only):

| Codec | ParseRate | ValRegime | DoAcc | DoParseFail | CFDivMSE | OODRegime | ProbeReg | NextB_R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 0.776 (0.002) | 0.594 (0.109) | 0.406 (0.031) | 0.094 (0.000) | 1.90e-04 (3.95e-04) | 0.531 (0.078) | 0.521 (0.092) | 0.653 (0.008) |
| C | 0.959 (0.014) | 0.609 (0.141) | 0.469 (0.031) | 0.000 (0.000) | 1.43e-04 (4.20e-05) | 0.594 (0.047) | 0.584 (0.026) | 0.653 (0.001) |
| D | 0.772 (0.002) | 0.531 (0.016) | 0.406 (0.031) | 0.094 (0.000) | 1.44e-04 (6.66e-05) | 0.547 (0.000) | 0.508 (0.040) | 0.651 (0.005) |
| A | 0.982 (0.005) | 0.562 (0.062) | 0.406 (0.031) | 0.000 (0.000) | 1.15e-03 (2.18e-03) | 0.516 (0.000) | 0.525 (0.041) | 0.646 (0.008) |
| B_reverse | 0.981 (0.001) | 0.562 (0.109) | 0.406 (0.031) | 0.000 (0.031) | 1.39e-04 (9.12e-04) | 0.500 (0.062) | 0.583 (0.111) | 0.657 (0.074) |
| B_bag | 0.145 (0.014) | 0.516 (0.062) | 0.125 (0.000) | 0.750 (0.125) | 4.03e-04 (2.65e-04) | 0.453 (0.125) | 0.460 (0.023) | 0.637 (0.016) |
| B_shuffle | 0.156 (0.013) | 0.531 (0.016) | 0.094 (0.062) | 0.812 (0.062) | 3.49e-04 (1.76e-04) | 0.500 (0.062) | 0.471 (0.008) | 0.643 (0.046) |

Token budgets from the training manifest (identical across the five seeds inside each cell):

| Codec | compute `n_episodes` | compute `T_eff` | info `n_episodes` | info `T_eff` |
|---|---:|---:|---:|---:|
| B, B_reverse, B_bag, B_shuffle | 78,770 | 1.08438e8 | 50,000 | 6.8832e7 |
| C | 73,330 | 1.08437e8 | 50,000 | 7.3938e7 |
| D | 65,372 | 1.08436e8 | 50,000 | 8.2938e7 |
| A | 50,000 | 1.08438e8 | 50,000 | 1.08438e8 |

`78,770 / 50,000 = 1.575`. Gemini’s “~1.58 epochs”, “68.8M”, “73.9M”, “82.9M”, and “108.4M” match `T_eff` to the printed precision. `tokens_consumed` medians sit within 0.05% of `T_eff`.

Report section (d), both decoders, because it is computed from probes and from `val.regime_accuracy`:

- `compute_matched`: `rho(valReg, probeReg)=0.929`, `rho(valReg, hiddenC_R2)=0.857` with the low-spread flag, spread `0.047`, `CKA(B,{C,D,A})=0.68` vs `CKA(B,{B_bag,B_shuffle})=0.69` (`B_reverse=0.70`).
- `info_matched`: `0.873`, `0.709` with the low-spread flag, spread `0.042`, `0.78` vs `0.81` (`B_reverse=0.62`).

Gemini’s probe table, Claude’s probe and CKA sentences, and ChatGPT’s probe and CKA tables match these figures. Skeleton-only behavioral cells (ParseRate, DoAcc, DoParseFail, CFDivMSE) also match ChatGPT’s skeleton-versus-`ftd` comparison for B and Claude’s skeleton counterfactual pair. No quoted median in the three write-ups disagrees with the regenerated tables.

### A.3 Adjudication matrix

Verdicts: **Verified** (the claim matches code and artifacts), **Corrected** (the numbers match; the stated mechanism or label does not), **Refuted**, **Open** (not measured, and not established).

| ID | Claim | Verdict |
|---|---|---|
| G1 | `probes_v2` medians replace the old flat probe picture, and `rho(valReg, probeReg)` shows that behavioral regime accuracy tracks the hidden state | Corrected |
| G2 | C is the stable winner at matched information; B’s regime-probe jump appears only after matched compute | Verified as numbers |
| G3 | C wins `DoAcc`; B does not beat A | Corrected |
| G4 | CKA no longer separates B from `{B_bag, B_shuffle}` because CKA is dominated by high-variance numeric trajectories | Numbers verified; subspace story open |
| G5 | Skeleton defers intervention comparisons; `ftd` still has ParseRate ≈ 0.77 for B and D and nine MSE outliers | Corrected |
| G6 | C’s trailing `alts:` line may be a register-token confound; test `C_no_alts` | Open, and a new condition |
| G7 | The run restores Scenario 3 (evidential/process advantage over A and D) | Refuted as a confirmed conclusion |
| S1 | The previous probe was in-sample OLS of the current `b`; after the fix, token order is not supported by `NextB_R2` or CKA | Verified |
| S2 | B and C separate on both regime probes under matched compute; the gap might be `n_episodes` rather than the codec | Separation verified; confound refuted |
| S3 | A’s `CFDivMSE` is about an order of magnitude worse than parsers that survive the intervention step | Verified; step-15 split open |
| S4a | `DoAcc` has no chance or oracle baseline | Verified as a reporting gap |
| S4b | `ProbeC_R2 ≈ 0.05` says nothing about whether `hidden_c` is readable | Refuted |
| S4c | `NextB_R2 ≈ 0.65` is last-value persistence | Refuted |
| S5a | `CLEAR_IQR_SEPARATION` is still printed for B versus `{B_bag, B_shuffle}` on intervention accuracy while those controls fail to parse | Verified |
| S5b | Skeleton `counterfactual=CLEAR_IQR_WORSE` for B versus `B_reverse` is a small median gap with a huge outlier | Verified |
| S5c | Provenance prints the three commits and omits `eval_git_dirty` and `probe_git_dirty` | Verified |
| S5d | Outlier lines say `rep_id=missing`, and index 2 recurs | Verified |
| S6 | Current conclusions are exploratory: decoder, thresholds, and the probe protocol were changed after results were seen | Verified |
| C1 | The priority-1 defect is the `B_bag` / `B_shuffle` parser | Refuted on the parser; verified on verdict deferral |
| C2 | Intervention results favor C over B, and B does not beat A, so a process codec is not helping causal reasoning | Corrected |
| C3 | B wins regime classification, and decoder-invariance shows the ranking is behavioral | Corrected |
| C4 | `NextB_R2` is moderate evidence that B helps dynamics | Refuted |
| C5 | MSE outliers up to `6.91e18` are OOD model instability | Refuted |
| C6 | The probe rewrite (held-out ridge, true `b_{t+1}`, prefix states) is the change that made probe numbers interpretable | Verified, with one penalty correction |
| C7 | CKA no longer supports “B’s representation differs from the E controls” | Verified |
| C8 | ChatGPT’s hypothesis scoreboard | Superseded by A.6 |

### A.4 G1 / C3 / C6 — two probes, two penalties, no decoded regime score

`format_summary_tables` labels `eval` split `regime_accuracy` as `ValRegime` and the `ood_partial_obs` copy as `OODRegime`. `score_passive_split` fills that field from `regime_probe_scores`. That function:

- reads `episode_hidden` at the last token of the framed episode;
- fits `_fit_regime_probe` on up to `PROBE_TRAIN_EPISODES = 128` training episodes;
- scores up to `EVAL_MAX_EPISODES = 64` eval episodes.

`_fit_regime_probe` is unstandardized ridge-to-one-hot with `PROBE_L2 = 1e-2` on the weights and no penalty on the bias, then a softmax. It does not call `tokenize`, `reconstruct_field_to_delimiter`, or a codec parser. `OODRegime` is the same function on the partial-observation split. Equal numbers across skeleton and `field_to_delimiter` are the expected consequence. They are not evidence that a parser change left a behavioral ranking untouched.

`ProbeReg` is a second use of the same `_fit_regime_probe`, called from `scripts/run_probes.py` on prefix states `states[0..t]` (10,000 states, 334 leading episodes of `val.h5`, episode-level 50/50 split). It is not the `PROBE_RIDGE_ALPHA = 1.0` path. That constant, with train-only standardization, is what `heldout_ridge_r2` uses for `hidden_c_r2` and `next_b_r2`. The on-disk `probe_protocol` records both facts: `"ridge_alpha": 1.0` and `"regime_probe": "eval.score._fit_regime_probe (ridge to one-hot)"`.

ChatGPT’s sentence “the new probe uses ridge regression with `alpha=1.0`” is true for `NextB_R2` and `ProbeC_R2` and false for `ProbeReg`. `docs/EXPERIMENT_PLAN.md` §6 currently says `regime_accuracy` uses `PROBE_RIDGE_ALPHA = 1.0`. The implementation, and the completed probe audit, intentionally left the regime probe on `PROBE_L2 = 1e-2`. This document does not change that plan sentence; it records the mismatch so later text does not treat the two penalties as one knob.

The old-probe defects named by all three write-ups are the defects fixed in the repository audit: `next_b` previously regressed the current state’s `b`, and `R²` was in-sample unregularized least squares. Current `scripts/run_probes.py` predicts `state_{t+1}.b` from the prefix and scores held-out episodes. The audit marked that target change as an I15 semantic change. Old and new probe numbers are not comparable. The write-ups are right to treat the drop from ~0.9 to ~0.65 as a measurement correction.

`rho = 0.929` / `0.873` is then agreement between the eval-split full-episode probe and the prefix probe. It is evidence that the regime signal is stable across those two readouts. It is not evidence that generated text tracks the hidden state.

### A.5 G3 / C2 / S4a — intervention accuracy

Official `ftd` medians are what ChatGPT and Gemini quote: C is highest (`0.438` / `0.469`), B matches D (`0.375` / `0.406`) and does not exceed A (`0.406` / `0.406`), and `B_reverse` matches C in `compute_matched` (`0.438`). B and D have `DoParseFail = 0.094` with IQR `0.000` in both budgets. A and, in `info_matched`, C have `DoParseFail = 0.000`.

`artifacts/eval_splits_v2/intervention.h5` has 32 episodes. The scored step is `t = 15`. Three episodes are missing at least one sensor. Re-scoring the five locked checkpoints with the same decoder, and restricting the hit rate to the 29 steps where both sensors are observed, gives:

| Codec | compute `DoAcc` (32) | compute, both observed (29) | info `DoAcc` (32) | info, both observed (29) |
|---|---:|---:|---:|---:|
| B | 0.375 | 0.414 | 0.406 | 0.448 |
| C | 0.438 | 0.414 | 0.469 | 0.448 |
| D | 0.375 | 0.414 | 0.406 | 0.448 |
| A | 0.406 | 0.345 | 0.406 | 0.379 |
| B_reverse | 0.438 | 0.414 | 0.406 | 0.379 |

The published `DoAcc32` column of that re-score matches the official medians, so the subset uses the same episodes the report used. On the fully observed steps, B, C, and D are the same number in both budgets. `B_reverse` joins that tie under matched compute and falls to A’s number under matched information. A is below the tie in both budgets. The headline “C wins, B loses to A” is the three partial-observation episodes, which B and D fail to parse and which therefore contribute failures in `intervention_accuracy` (`hits / total`, with a parse exception counted as a non-hit).

Why those three episodes fail for B and D, and not for A, C, or `B_reverse`, follows from teacher-forced reconstruction plus the parsers.

`reconstruct_field_to_delimiter` writes predicted tokens into the true string’s token slots. A partial observation changes which tokens exist:

- Codec B emits `(miss X)` and only then the meta event that carries `source` (`encode_B`). On `info_matched` `rep0`, first 16 episodes of `val.h5` (370 both-observed next-steps, 94 partial), B parsed 0/94 partial steps and 360/370 both-observed steps. `B_reverse` puts the meta event first (`reverse_b`) and parsed 92/94 partial steps. The same event vocabulary with the source earlier in the string does not show the failure.
- Codec D requires the literal `None` whenever `X.obs=0`, and `X.x` is written before `X.obs` (`observation_from_D`). The same 16 episodes: 0/94 partial, 365/370 both-observed.
- Codec A, when `status=absent`, discards the value token (`_parse_entity`). Partial parse was 93/94.
- Codec C accepts any `float()`-parseable token in the one-character `?` slot (`_parse_value_field`), so a predicted digit counts as a successful parse of an observed value. Partial parse was 87/94.

The val ParseRate of B (`0.774`) and D (`0.771`) is the arithmetic consequence: about one fifth of steps are partial (`DEFAULT_DIFFICULTY_V0.partial_obs_rate = 0.2`; 94/464 = 0.203 in that slice), those steps do not parse, and both-observed steps mostly do. This is not a random 23% syntax failure. `DoParseFail = 0.094` with IQR 0 is the same fact on the intervention file: every seed fails the same three episodes.

`quantize_value` uses right-open edges `i/10`. Predicting the constant `0.0` is a hit whenever the true affected value is `< 0.1`. On this intervention file that happens on **12/32 = 0.375** steps. B’s compute-matched `DoAcc` equals that floor, with IQR 0. A noise-free graph oracle (predict the intervened parent when the scored variable is its child, `hidden_c` under `COMMON_CAUSE`, otherwise `0`, then the same binning) scores **18/32 = 0.562**. No model median clears that oracle. C’s info-matched `0.469` is above the zero floor and below the oracle, and it is not higher than B or D once the 29 fully observed episodes are isolated.

`intervention_accuracy` also drops a step from the numerator, while keeping it in the denominator, when the affected variable is unobserved. Partial steps can therefore move `DoAcc` without sharing a single mechanism. The both-observed subset is the comparison that removes that split. It is not in the published report. Adding it is an additional diagnostic (Part B), not a correction of the world or of the codecs.

### A.6 C1 / S5a — permuted controls are not a broken parser

`eval/score.py` parses `B_bag`, `B_shuffle`, and `B_reverse` with `observation_from_B_unordered`. That parser classifies events by role, accepts arrow-separated and bag-separated strings, and requires one A event, one B event, and one meta event. Ground-truth strings produced by `shuffle_b` and `bag_b` are in its domain. The low score is not “the parser still demands canonical order.”

`shuffle_b` and `bag_b` permute the three events with `random.Random(derive_transform_seed(...)).shuffle`, and `derive_transform_seed` is `blake2b_u64` of `episode:timestep:variant`. There are `3! = 6` orders. Teacher-forced decoding fills the true permutation’s slots. A model that emits one preferred order yields three role-valid events only when the true permutation is that order, which is 1/6 of steps if the hash is uniform over the symmetric group. `1/6 = 0.1667`.

Measured val ParseRate: `B_shuffle` `0.167` (compute) and `0.156` (info); `B_bag` `0.152` and `0.145`. `B_shuffle` under matched compute lands on 1/6. `B_bag` is a few points lower, which fits the different separator (` | `) changing the token skeleton, not a parser that rejects true bag strings. `PERMUTED_PARSE_RATE_MAX = 0.25` is the sanity gate that *requires* permuted val ParseRate to sit below 0.25. It does not cause the rate. Raising the rate by teaching the scorer to ignore slot identity would change what an E control is. I11 says E destroys B’s organization and keeps the token inventory. A parser fix that makes a random permutation score like canonical B would remove that control.

Intervention `DoParseFail` for these codecs is **not** `1 - 1/6 ≈ 0.833`. The `ftd` medians are `0.938` / `0.750` (`B_bag`) and `0.844` / `0.812` (`B_shuffle`). The intervention step adds the partial-observation failure mode of A.5 on top of the permutation. ChatGPT’s round “about 0.83” is a rough average of those four medians, not an identity.

The verdict defect is real. `INTERVENTION_PARSE_FAIL_COMPARE_MAX = 0.20` is applied only inside baseline pairs `B vs A`, `B vs D`, and `C vs A`. It is not applied to `b_vs_e_organization`, `b_vs_e_direction`, or `b_vs_e_outcomes`. On `ftd` intervention accuracy the report prints:

- `compute_matched`: `intervention=CLEAR_IQR_WORSE` with `org=CLEAR_IQR_SEPARATION`;
- `info_matched`: `intervention=NO_ADVANTAGE` with `org=CLEAR_IQR_SEPARATION`.

Those organization labels are computed from `DoAcc` while `B_bag` / `B_shuffle` fail to parse on 75–94% of the scored steps. Skeleton does defer the A/D baselines (`DoParseFail` 0.188–0.594 on the ordered codecs) and still prints `org=CLEAR_IQR_SEPARATION` against the E controls. Claude’s deferral request and ChatGPT’s “do not read B’s win over bag/shuffle as a cognitive win” are the right reporting change. The right code change is to run the existing 0.20 parse-fail guard on the E comparisons. It is not a new parser.

### A.7 S4b / S4c / C4 — `hidden_c` and next-step `b`

`simulate` draws `hidden_c ~ Uniform(0, 1)` for every episode, then `_generate_observables` adds it to both sensors only in `CausalRegime.COMMON_CAUSE`. The other three regimes do not read `hidden_c`. The regime is one of four equally likely values, so the design fraction of usable targets is 1/4. The probe fixture realized 96/334 episodes (`0.287`) in `COMMON_CAUSE`. Pooling the other episodes asks a linear probe to predict a uniform noise label from a representation of observables that do not depend on it. A global `R²` near 0.05 is what that mixture does. The report already flags the spread (`0.047` and `0.042`) as `low-spread; not a signal`. The positive `rho(valReg, hiddenC_R2)` is a correlation on that collapsed number. It is not an independent latent-variable result, which is what the flag says.

The same `alpha = 1.0` held-out ridge, restricted to `COMMON_CAUSE` rows, on compute-matched `rep0`:

| Codec | pooled `hidden_c_r2` | `COMMON_CAUSE` only |
|---|---:|---:|
| B | 0.029 | 0.949 |
| C | 0.082 | 0.941 |
| D | 0.021 | 0.809 |
| A | 0.037 | 0.913 |
| B_reverse | 0.027 | 0.870 |
| B_bag | -0.015 | 0.949 |
| B_shuffle | 0.058 | 0.939 |

These are one replication, not the five-seed median. The pooled column is that seed’s global `R²`, which is why it is not identical to the median `ProbeC_R2` in A.2. The contrast is the point: the latent is linearly readable on the regime that actually emits it, including from `B_bag` and `B_shuffle`. Codec D is the weakest of the seven on this single seed (`0.809`), not proof that ordered codecs uniquely store `hidden_c`. Claude’s alternative “the probe is too weak at `alpha=1.0`” is not needed to explain the pooled number, and this slice does not support it.

Next-step baselines use the same 10,000-state fixture, the same episode split, and the same held-out rows that have a `next_b` (`n = 4824`). `R²` is against the test-set mean.

| Predictor of `b_{t+1}` | Held-out `R²` |
|---|---:|
| Last observed `b` (persistence) | 0.423 |
| Last latent `b`, ignoring the mask | 0.429 |
| Mean of observed `b` on the prefix `0..t` | 0.667 |
| Mean of latent `b` on the prefix | 0.676 |
| `E[b_{t+1} \| regime, hidden_c]` = `hidden_c` in `COMMON_CAUSE`, else `0` | 0.715 |

Each step draws fresh `noise_a`, `noise_b`, and `shared`. Given `(regime, hidden_c)`, `b_{t+1}` does not depend on `b_t`. There is no step-to-step dynamical residue for a world model to learn beyond estimating that latent. Persistence is the wrong ceiling: the model medians `0.646–0.669` sit on the running mean, about 0.05 below the latent conditional mean. B’s compute-matched `0.669` is the highest median and is 0.019 above A’s `0.650`. That gap is smaller than the distance from persistence to the running mean. It is not evidence that B learned dynamics the other codecs missed. ChatGPT’s “weak support that B improves next-state prediction” should be read as a small difference in how well the prefix mean is integrated, on a metric that is almost flat across codecs, including bag and shuffle (`0.637–0.652`).

### A.8 S2 — episode count does not explain the regime probes

Under matched compute, B, `B_reverse`, `B_bag`, and `B_shuffle` share `n_episodes = 78,770` and `T_eff = 1.08438e8`. Their `ValRegime` medians run from `0.688` to `0.516`, and their `ProbeReg` medians from `0.617` to `0.502`. The permuted controls saw every state B saw. D saw more episodes than A (`65,372` versus `50,000`) and does not lead A on `ValRegime` (`0.547` versus `0.562`) or on `ProbeReg` (`0.530` versus `0.523`). A “more states, higher regime score” confound does not fit these two comparisons.

The budget interaction the write-ups describe is still in the table. Info-matched B has the smallest `T_eff` (`6.8832e7`) and its `ProbeReg` falls to `0.521` (IQR `0.092`), level with A and D, while C at `7.3938e7` tokens holds `0.584` (IQR `0.026`). Calling the info-matched B run “undertrained” is an interpretation of that wider IQR and the shorter budget. It is not a separate diagnostic.

### A.9 G4 / C7 / S1 — CKA

The CKA medians in A.2 match all three write-ups. The earlier separation (ordered versus permuted) is gone: B versus `{C, D, A}` and B versus `{B_bag, B_shuffle}` differ by 0.01 in `compute_matched` and by 0.03 in `info_matched`, with the permuted pair slightly higher in both. `NextB_R2` being flat is consistent with a shared numeric subspace, which is Gemini’s proposed reason linear CKA would not see the regime signal that `ProbeReg` sees. No variance decomposition or subspace ablation was run for this verification. The numerical claim is verified. The subspace mechanism stays open. §8’s “beh ≠, repr ≠” cell is not supported by this CKA.

### A.10 S3 / S5b — counterfactual MSE

`ftd` compute-matched medians: A `2.17e-3`, B `1.07e-4`, C `1.15e-4`, D `1.08e-4`, `B_reverse` `1.06e-4`. Info-matched A is `1.15e-3` against `1.39e-4`–`1.90e-4` for B, C, D, and `B_reverse`. Claude’s “about 8–20× versus the codecs that parse the intervention step” matches A versus C and A versus `B_reverse`. `B_bag` and `B_shuffle` are higher than B (`1.87e-4` and `2.10e-4` in compute) and are the wrong contrast if the concern is parse failure, which Claude already says. Whether A’s excess error is concentrated after the intervention step was not measured. The metric remains a single mean over the paired episodes.

Skeleton compute-matched counterfactual is the comparison Claude flags: B median `1.39e-4` (IQR `6.73e-4`) versus `B_reverse` `1.09e-4` (IQR `4.37e-6`), and the verdict is `CLEAR_IQR_WORSE` on the direction contrast. The median ratio is about 1.3. One B replication is the `6.17e11` outlier printed in that bundle. The label has no minimum effect size. The same contrast under `field_to_delimiter` is `NO_ADVANTAGE` (`1.07e-4` versus `1.06e-4`). The skeleton wording does not survive the decoder that the write-ups treat as the cleaned evaluation.

### A.11 G5 / C5 / S5d — MSE outliers and missing replica ids

`aggregate_scientific_metrics` stores `median`, `q1`, `q3`, `iqr`, and `values` for each codec. It does not store `rep_ids`. `eval/report.py` then labels every extreme cell `rep_id=missing(idx=k)`. Under the current `rep{0..4}` directory layout, sorted glob order makes `idx` equal the replica number: the `ftd` value lists match `metrics_eval.json` of `rep{k}` one-for-one, including `info_matched` / `ood_partial_obs` / `next_state_mse` / D, where index 2 is `rep2` and the value is `6.91239861063523e18`. The label is still wrong as provenance. A future non-sequential layout would make `idx` ambiguous. Persisting `rep_ids` is a reporting fix.

The `ftd` verdict prints nine `MSE>1000` lines. Three of them are the same cell as another line. `score_passive_split` sets `partial_obs_next_state_mse = next_state_mse` whenever `partial_obs` is true, and the outlier walk emits both names. The three duplicated pairs are:

- `compute_matched` / `ood_partial_obs` / B / `rep2` = `4.54e12`
- `compute_matched` / `ood_partial_obs` / `B_reverse` / `rep2` = `2.92e12`
- `info_matched` / `ood_partial_obs` / D / `rep2` = `6.91e18`

The other three lines are distinct: compute val B `rep2` `5.5e10`, compute val D `rep3` `1.71e12`, compute counterfactual B `rep2` `3.95e11`. Six distinct cells, nine lines. ChatGPT’s four examples are four of those cells. The count “nine” is the printed list, not nine independent failures. The skeleton bundle prints eight lines by the same duplication.

The `6.91e18` cell was traced on `info_matched/D/rep2`, `ood_partial_obs` episode 39, step 9. The true row is fully observed `sensor_both`:

```text
A.x=1.1145690765; A.obs=1; B.x=0.6688629971; B.obs=1; noise=medium; n=3; source=sensor_both
```

The reconstruction is:

```text
A.x=1.0111111999; A.obs=1; B.x=109922121393; B.obs=1; noise=medium; n=3; source=sensor_both
```

`B.x` is a 12-character integer. Codec D renders a float in a fixed `regex_v4` character span (`0.6688629971` is 12 characters). Dropping the decimal-point token inside that span turns an O(1) value into an O(1e11) integer. One step of that size dominates the split mean (`(109922121393 - 0.669)² ≈ 1.21e22` before averaging). The step is not a partial observation, and it is not evidence of a separate OOD dynamical instability. The same class of error is the credible account of the other `MSE>1000` cells: a mean is not a robust summary of a squared float-token error. Medians of the non-outlier seeds stay near `0.05–0.12`. The sanity gate stays `PASS` because the values are finite. The warning is justified; the instability story is not.

### A.12 G2 / G6 / G7 — what the regime probes support, and what `C_no_alts` would be

Compute-matched `ProbeReg`: B `0.617` (IQR `0.014`), C `0.599` (IQR `0.043`), D `0.530` (IQR `0.067`), A `0.523` (IQR `0.022`), permuted controls `0.502` and `0.508`. Compute-matched `ValRegime`: B `0.688`, C `0.672`, A `0.562`, D `0.547`, with the report marking B versus A and B versus D as `IQR>`. Info-matched `ProbeReg`: C `0.584` (IQR `0.026`) and `B_reverse` `0.583` (IQR `0.111`) are the high medians; B `0.521` (IQR `0.092`) is not separated from A `0.525` or D `0.508`. Gemini’s budget split of B versus C matches this table. D tracking A, rather than B, also matches it, in `compute_matched`. In `info_matched`, B does not sit above D, so “concise structure is not enough” is a compute-matched statement.

Codec C’s last line is the constant `alts: {common_cause, a_causes_b, b_causes_a, spurious}`. It carries no state bits and is excluded from `FactRecord`, by the same design as Codec A’s constant `Relation: A and B are similar.` A trailing constant is therefore not unique to C, and A’s `ProbeReg` (`0.523` / `0.525`) stays with D. That does not isolate the `alts` line: A’s whole string differs from C’s. It does mean “any deterministic tokens at the end of the block” is not sufficient for C’s info-matched `ProbeReg`. A `C_no_alts` run would be a new codec condition. It changes a codec definition. I15 requires an explicit methodological flag and a plan update before that comparison is treated as part of the locked A–E result. It is a legitimate next experiment. It is not a bug fix, and this document does not authorize it.

G7’s upgrade to a confirmed evidential/process advantage is not available. The intervention half of that upgrade is the artifact in A.5. The regime-probe half is a real median pattern with five seeds, a probe penalty that §6 describes differently from the code, and a protocol that was revised after earlier numbers were inspected (S6). §8 allows “C > A on calibration/intervention, without oracle `p`” only as “evidential framing improves uncertainty/causal learning,” and the intervention comparison that would carry that sentence is not clean. §8 forbids “C is Bayesian” and “B thinks processually.”

### A.13 S5c / S6 — provenance and exploratory status

Training `git_dirty` is false and is printed. Eval `git_dirty` is true on both eval manifests and on both aggregate trees, and is not printed. Probe `git_dirty` is false on `probes.json` and is not printed. Claude’s provenance sentence is accurate. I13 is satisfied for the training commit and only partly satisfied for eval: the commit is recorded, the dirty tree is recorded on disk, and the human-facing verdict hides it. A publication bundle wants `eval_git_dirty=false`, which means re-scoring from a clean tree (Part B), not re-training.

Claude’s exploratory label matches the audit history. The decoder (`skeleton` versus `field_to_delimiter`), the intervention parse-fail guards, and the probe target (`current b`, in-sample OLS, versus `b_{t+1}`, held-out ridge) were all chosen after looking at scores. The locked checkpoints themselves were not retrained for those choices. Confirming the regime-probe pattern wants a pre-registered comparison on new training seeds and a fresh eval split. That is a new experiment, not a repair of this bundle.

### A.14 Corrected hypothesis board

This replaces ChatGPT’s section 9. “Supported” here means “the printed comparison agrees, under the measurement actually implemented,” not “§8’s ontological reading is licensed.”

| Hypothesis | Status after verification |
|---|---|
| Codecs change what the frozen regime probes can read | Supported in `compute_matched` for B and C above A, D, and the permuted controls. Info-matched support is C (and a noisy `B_reverse`), not B. |
| That regime score is a decoded behavioral classification | Refuted. It is a linear probe. Decoder-invariance is uninformative. |
| B beats A and D at matched episode count | Not supported. Info-matched B’s `ProbeReg` overlaps A and D. The compute-matched lead co-occurs with 78,770 versus 50,000–73,330 episodes. The E controls rule out raw episode count, not token budget versus codec. |
| Organization beyond bag/shuffle helps regime probes | Supported on `ProbeReg` / `ValRegime` in `compute_matched`, where parse rate is not the metric. Not established on intervention `DoAcc`. |
| Natural chronological order is required | Not supported. `B_reverse` matches or exceeds B on several `ftd` columns and on info-matched `ProbeReg`. |
| B improves next-step dynamics | Not supported. The series is conditionally i.i.d.; all codecs match a running mean. |
| B improves intervention reasoning | Not supported on the 29 fully observed episodes, where B ties C and D and beats A. The published `DoAcc` ranking is the partial-observation parse gap. |
| C is better than B at interventions | Not supported on those 29 episodes. |
| `hidden_c` is absent from the hidden state | Refuted for `COMMON_CAUSE` on the `rep0` slice, including bag and shuffle. |
| B improves counterfactual or OOD regime generalization | Not supported. A is worse on `CFDivMSE`. OOD regime medians are a tight, low cluster led by C at `0.594`. |
| Internal representations separate ordered codecs from E | Not supported by linear CKA. |
| Extreme MSE shows OOD instability | Refuted for the traced cell. It is a missing decimal point in a fixed-width float. |

---

## B. Required changes, I15 gates, and rerun matrix

Nothing in this section has been implemented. Items are split by whether they change a measured quantity.

I15, from `docs/INVARIANTS.md`: a change to the information contract, codec definitions, matching, evaluation targets, primary endpoints, or seed and split policy needs an explicit methodological flag and a plan update. A reporting fix that prints the same `metrics_eval.json` differently does not. An added diagnostic key that leaves the locked keys byte-identical in definition is a recording change; it still needs to be identified as such, and it needs a rerun of the stage that computes it. It does not authorize editing one codec, one loss, or one parser.

### B.1 Reporting and provenance (no new scores)

These edits do not change training, codecs, or the definition of any locked metric. Afterward, regenerate the verdict from the existing JSON. Do not re-score and do not re-probe.

1. **`train/matrix.py` `aggregate_scientific_metrics`.** Parse the integer replica id from `rep_dir.name` and store `rep_ids` beside `values`. Outlier lines should then say `rep2` / `rep3`. Today `idx` happens to match that name only because the glob is sorted.
2. **`eval/report.py` `format_verdict_summary` and bundle provenance.** Copy `eval_git_dirty` from the aggregate summaries and `probe_git_dirty` from `probes.json` into `ReportBundle.provenance`, and print both. I13. The current `ftd` and skeleton bundles would print `eval_git_dirty=true`, `probe_git_dirty=false`.
3. **`eval/report.py` extreme-outlier walk.** When `partial_obs_next_state_mse` is the same series as `next_state_mse`, emit one line. The `ftd` warning count should fall from 9 lines to 6 cells.
4. **`eval/report.py` E-control intervention labels.** Apply `INTERVENTION_PARSE_FAIL_COMPARE_MAX` (`0.20`) to `b_vs_e_organization`, `b_vs_e_direction`, and `b_vs_e_outcomes` for `intervention.intervention_accuracy`. With present `ftd` rates, `B vs {B_bag, B_shuffle}` becomes `DEFERRED` in both budgets. Do not raise `PERMUTED_PARSE_RATE_MAX` and do not change `observation_from_B_unordered`.
5. **`scripts/summarize_results.py` `render_fig3_probes_and_rsa`.** The probe panel plots only `hidden_c_r2`, the metric A.7 shows is diluted by construction. Plot or annotate `ProbeReg` (`regime_accuracy`) on that figure so the regime-probe result is visible without being mistaken for the pooled `hidden_c` number.
6. **Tests.** Extend `tests/test_summarize_results.py` and `tests/test_device_and_eval.py` for `rep_ids`, the dirty-flag line, outlier de-duplication, the E-control deferral, and the figure’s `ProbeReg` series.

### B.2 Additive diagnostics (locked keys unchanged)

These add keys. They do not retune `PROBE_RIDGE_ALPHA`, `PROBE_L2`, the episode split, the codecs, or the existing metric definitions. Implementing them is still a recorded analysis change: say so in the run metadata. Do not silently replace `DoAcc` or `hidden_c_r2` with the new keys.

**Probes, in `scripts/run_probes.py`**, written next to the locked trio:

- `hidden_c_r2_common_cause`: the same `alpha=1.0` held-out ridge, rows whose regime is `COMMON_CAUSE` only.
- `persistence_next_b_r2`: held-out `R²` of last observed `b`.
- `running_mean_next_b_r2`: held-out `R²` of the observed prefix mean.
- `oracle_next_b_r2`: held-out `R²` of `E[b_{t+1} | regime, hidden_c]`.
- Untrained-init scores for the same three locked probes, same fixture and split, so a trained `R²` has an initialization floor.

**Evaluation, in `eval/score.py`, only if an eval refresh is wanted.** Do not change `obs_codecs/`.

- `intervention_accuracy_both_observed`: the current hit rule on steps with both sensors observed.
- `intervention_accuracy_parsed`: hits over successfully parsed steps.
- Split-level `zero_baseline_acc` and `oracle_acc` (constant `0`, and the noise-free graph mean used in A.5), stored as metadata so 0.375 and 0.562 sit beside `DoAcc`.
- Paired error of the counterfactual metric at `t=15` versus the other steps, so A’s `CFDivMSE` can be attributed.
- Median and winsorized next-state error beside raw `next_state_mse`. Keep the raw mean. The float-token failures in A.11 will otherwise keep dominating any future mean.

### B.3 Not a fix

- Do not rewrite `encode_B` or `encode_D` so that partial observations parse. That is a codec-definition change (I15) and would move the information layout of B and D relative to A, C, and E.
- Do not “repair” `B_bag` / `B_shuffle` by parsing them as canonical B. I11.
- Do not add `C_no_alts` inside this matrix. New condition, I15, own budget row.
- Do not retrain `full_v0`. Commit `eed26bed` is clean and is not implicated by any finding above.
- Do not increase seeds from 5 to 15–20 as part of repairing this report. That is a new training matrix. It is the right confirmation step after B.1 and B.2, and only under a comparison written down before the run (S6).

### B.4 Rerun matrix

| Change | Stage-1 train | Probes | Eval | Report |
|---|---|---|---|---|
| None (this document) | no | no | no | no |
| B.1 only | no | no | no | Re-aggregate existing `metrics_eval.json` or rerun `scripts/summarize_results.py` |
| B.2 probes | no | `scripts/run_probes.py --matrix` | no | summarize after probes |
| B.2 eval keys, or a clean `eval_git_dirty=false` bundle | no | no | `scripts/evaluate_runs.py` for `skeleton` and `field_to_delimiter` | summarize after eval |
| `C_no_alts`, new seeds, new splits | new experiment under I15 | after that train | after that train | after those runs |

B.1 does not need a GPU. B.2 probes reuses the locked checkpoints. B.2 eval is optional and is also the only way to clear `eval_git_dirty=true` without rewriting history. Neither eval decoder needs to be rerun to stop the verdict from calling a 94% parse failure `CLEAR_IQR_SEPARATION`; that is B.1 item 4.
