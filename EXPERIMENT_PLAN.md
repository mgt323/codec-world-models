# Codec World Models — Experimental Plan (MVP)

**Working title:** Effect of observation codecs on learnable world models under matched compute and matched information.

**Not:** a direct test of “process ontology of mind.”  
**Yes:** a controlled test of **representation / codec inductive bias**. Ontology is an interpretive hypothesis *after* results (especially after control E + RSA).

---

## 0. Goal of the MVP

```text
One simulator → multiple codecs → one architecture → same training procedure
→ evaluation in world-state space (not text style)
```

Anything that does not serve this chain is deferred.

---

## 1. Information contract (hard law)

For every timestep \(t\):

\[
O_t = \mathrm{Obs}(S_t), \quad
X^{(k)}_t = f_k(O_t), \quad
k \in \{A,B,C,D,E\}
\]

Rules:

- No training codec may read \(H_t\) (hidden masses, true causal graph, oracle posteriors).
- Full state \(S_t, H_t\) exists only for: world generation, evaluation targets, probe targets.
- Automated **info-equality suite**: recoverable facts from \(O_t\) must match across codecs (see §1.1).

If C needs `source` / `noise` / `n_samples`, those fields must be part of \(O_t\) (or recoverable from the same observation history available to A/B/D).

### 1.1 Info-equality suite (operational)

Codecs are synthetic and generated — equality is **deterministic invertibility** to a shared fact schema, not string match across codecs and not a trained probe.

1. **Canonical schema** \(F(O_t)\): finite field list from `Obs`, e.g.  
   `{A_obs, A_val_q, B_obs, B_val_q, noise_bin, n_samples, source, …}`.  
   Continuous values use **one shared quantization** across all codecs.
2. **Per codec \(k\)**: `encode_k: O → string` and `parse_k: string → F` (hand-written parsers; no ML).
3. **Equality test** (fixed seed grid / \(N\) random \(O_t\)):  
   \(\mathrm{parse}_k(\mathrm{encode}_k(O)) = F(O)\) for all \(k \in \{A,B,C,D,E\}\).  
   For E after structure destruction: parser still recovers the **per-\(t\) fact multiset**, not order — assert separately.
4. **MVP levels:** L1 exact match on quantized \(F\); optional L2 re-encode stability.  
   L3 (frozen encoder + linear probe) is deferred — unnecessary when \(f_k\) is fully controlled.

Readiness item 2 is green only when this suite passes on the locked seed set + schema checksum.

---

## 2. World v0 — start here (not 2D physics)

**1D causal toy** as proof of concept:

- Observables: \(A_t, B_t\)
- Hidden: \(C\) (fixed per episode or rarely changing)
- Generative regimes (sampled per episode):
  1. \(C \to A,\ C \to B\) (common cause)
  2. \(A \to B\) (+ noise)
  3. \(B \to A\) (+ noise)
  4. spurious / mixed confounding
- Partial observation: sometimes only \(A\), only \(B\), or both
- Interventions: `do(A=a)`, `do(B=b)`
- Deterministic seed → full reproducibility

**Primary MVP claim to test:** whether some codecs make *correlation ≠ causation* easier to learn under interventions.

**Escalation path:** 1D causation → 1D colliders → 2D physics (3–8 bodies, gravity, contacts, 1–2 hidden causes). Do not start with rich physics.

### 2.1 Difficulty axis (not a fixed world)

Four regimes alone can ceiling out under mild noise. Treat difficulty as a **searched / reported axis**, not a constant:

- Parameters: `noise_scale`, `regime_switch_rate` (0 = fixed regime per episode), `partial_obs_rate`, optional confounding strength.
- **Difficulty pilot (gate before the full training matrix):** on codec D (or a small oracle probe from \(O\)), find an operating point where regime-classification accuracy ∈ **[~0.55, ~0.80]** — above chance, below ceiling.
- Primary results: curves vs `noise_scale` (at least 3 points), not a single default world.
- OOD (eval task 7): higher noise / more frequent switches than train — only after the calibrated operating point is locked.

### Simulator requirements

- Deterministic seed
- Full access to latent state (for eval only)
- Controlled stochasticity
- Interventions `do(X)`
- Step-by-step trajectory log
- Configurable difficulty parameters above

**Deliverable:** `simulate(seed, interventions, difficulty) -> list[State]` with `Obs(State) -> O`

**Stack:** See `TECH_STACK.md` (Python 3.13.15, PyTorch, uv, …). Light custom dynamics or simple Euler / pymunk later for 2D.

---

## 3. Codecs (revised)

### Temporal availability (ambient time contract)

Any sequential LM has access to order via position. That is an ambient artifact of the training format, **not** a codec-specific “process ontology.”

- **A-ordered, B, C, D** receive chronologically ordered timestep blocks (positional time shared).
- Destroying order is an **E control**, not a hidden property of A: `B-shuffle` / `B-reverse` / `B-bag`, and optionally **A-bag** (same essentialist blocks, randomly permuted within the episode).
- Therefore A vs B measures **how change is lexicalized** (static attributes vs eventive / relational scaffolding), not whether time exists in the stream.

**Claim hierarchy (use in §8):**

1. \(B > A\text{-ordered}\) ⇒ linguistic scaffolding of change helps (weaker, honest thesis).
2. \(B > A\text{-bag}\) and \(B > B\text{-shuffle}\) ⇒ process *organization* helps, not token presence alone.
3. \(B \approx B\text{-shuffle}\) ⇒ surface-distribution artifact.

Do **not** ship A-bag as the only A variant: that double-handicaps A (vocabulary + no order) and confounds interpretation. Full non-temporal / Minkowski language stays deferred (§12).

### A — Substance / essentialist caption

Maximize entities + static attributes; **minimize eventive verbs**.

```text
Entity A: value=0.41, status=present, label=source-like.
Entity B: value=0.38, status=present, label=target-like.
Relation: A and B are similar.
```

**A-ordered (default):** concatenate such per-timestep blocks in temporal order.  
**A-bag (optional E-analog):** same blocks, shuffled within the episode.

Deliberately weak for dynamics — that is the contrast, not a bug.  
Do **not** use captions like “Red ball hits blue box; box tips over” (too process-like; collapses A vs B).

### B — Process / relational event chain

Explicit transitions and order:

```text
(observe A↑) → (co-vary B↑) → (link A~B strength=med)
```

### C — Evidential observational record (**no oracle p / causes**)

Do **not** feed simulator posteriors. Feed observation metadata; the model must estimate belief:

```text
obs: A=0.41, B=?
source: sensor_A
n: 3
noise: high
alts: {common_cause, A_causes_B, B_causes_A, noise}
```

Only then does ECE/Brier measure learned calibration rather than parroting a provided `p=0.73`.

### D — Canonical structured baseline

Control for “structure beats prose,” without philosophical framing:

```text
A.x=0.41; A.obs=1; B.x=0.38; B.obs=1; noise=high; n=3
```

If \(D \approx B \gg A\), the effect is mostly **structurality**, not process ontology.

### E — Structure-destroyed controls (on B and/or C, optional A)

Same tokens / similar entropy; destroyed organization:

- `B-shuffle` — permute event order
- `B-reverse` — reverse the chain
- `B-bag` — multiset without order markers
- optional `A-bag` — permute essentialist per-timestep blocks (see ambient time contract)
- optional `C-shuffle` of fields

**Inference rule:**

- \(B \gg A\text{-ordered}\) and \(B \gg B\text{-}shuffle\) ⇒ **organization** helps, not vocabulary.
- \(B \gg A\text{-ordered}\) but \(B \approx B\text{-}shuffle\) ⇒ surface-distribution artifact.
- \(B \gg A\text{-bag}\) (with \(B \gg B\text{-}shuffle\)) ⇒ stronger support that process organization, not ambient position alone, carries the effect.

---

## 4. Naming discipline

If B wins, you do **not** yet know whether:

- process ontology is better, or
- explicit relational/causal scaffolding is simply easier to learn.

Call the experiment a **representation-effect test**. Ontology talk comes after D/E + probes + RSA.

---

## 5. Compute matching (closed decision)

Always report:

| Metric | Why |
|---|---|
| FLOPs | compute budget |
| optimizer steps | optimization dynamics |
| tokens | autoregressive length |
| episodes / unique world states | world exposure |
| wall-clock | engineering reality |
| vocab size, avg tokens/event | codec diagnostics |

### Two mandatory regimes

1. **Compute-matched:** fixed FLOPs (or fixed token-updates).
2. **Info-matched:** fixed number of episodes / states \(O_{1:T}\).

Accept “B is better” only if direction agrees in both regimes, **or** explicitly qualify  
(*B wins under matched info, loses under matched FLOPs* = more expensive representation).

### Replications

**3–5 seeds** per codec × regime. Report median + IQR / bootstrap. Without this, \(B > A\) will not survive review—variance in small models often exceeds codec gaps.

---

## 6. Model

- Identical architecture for all codecs (~5–20M for v0; up to ~10–100M later)
- Objective stage 1: **next-token prediction only** in that codec’s alphabet
- Same optimizer family; regime-specific budgets as above
- Config difference only: `codec` + `match_regime` + `seed`

### Probes — post hoc only

Do **not** train a shared latent state head during stage 1 (it can force representational convergence).

After training:

1. Freeze model
2. Extract activations on identical world states
3. Train identical linear probes → hidden cause / regime / next \(O\) / etc.

---

## 7. Evaluation (8 tasks)

All in observation/state/intervention space — not text aesthetics:

1. **Next-obs prediction**
2. **Hidden-cause / regime classification** from history \(O\)
3. **Intervention accuracy** — `do(A)`: does predicted \(B\) match true graph?
4. **Counterfactual twins** — same noise seed, different intervention
5. **Calibration** — ECE/Brier on regime or value preds (C without provided \(p\))
6. **Partial-observability stress**
7. **OOD regime** — new noise scales / graph priors
8. **Cross-codec transfer** — after symbol alignment: zero/few-shot B→A inputs, A→B, train A+D test B, etc.

### Representation similarity (RSA)

Pass ~10k identical world states through A/B/C/D; compare activations with CKA / SVCCA / orthogonal Procrustes.

|  | repr ≈ | repr ≠ |
|---|---|---|
| **beh ≈** | shared world model | different solutions, same readout |
| **beh ≠** | difference mostly policy/head | strongest support for representation hypothesis |

Also interesting: **beh ≈, repr ≠** — different internal solutions, same behavior.

---

## 8. Allowed vs forbidden conclusions

| Result | Allowed conclusion |
|---|---|
| \(B > A\text{-ordered}\), \(B > B\text{-}shuffle\), \(D \approx B\) | explicit relational structure helps; not “magic of process” |
| \(B > A\text{-bag}\), \(B > B\text{-}shuffle\) | organization of change helps beyond ambient positional order |
| \(C > A\) on calibration/intervention, no oracle \(p\) | evidential framing improves uncertainty/causal learning |
| beh ≠, repr ≠ | different codecs → different internal solutions |
| beh ≠, repr ≈ | difference mostly surface generation |
| beh ≈, repr ≈ | codecs are style; world model converges |

**Forbidden shortcuts:** “B thinks processually,” “C is Bayesian,” until E + probes + RSA narrow the claim.

---

## 9. Dataset

- Scale v0: enough episodes for clear separation (tens of thousands; scale up as needed)
- Splits: train / val / OOD (new noise, layouts, graph priors)
- Separate intervention and counterfactual suites
- Storage: raw states in HDF5/zarr; codec strings tokenized offline or on the fly
- One-script generator + seed checksums

---

## 10. Repo layout

```text
codec-world-models/
  EXPERIMENT_PLAN.md
  world/          # simulator, State/Obs schema, interventions
  obs_codecs/     # encode_A..E + info-equality tests (not stdlib codecs)
  data/           # generate_episodes.py
  model/          # identical transformer
  train/          # single entrypoint + configs
  eval/           # state-based benchmarks + transfer
  probes/         # frozen linear probes + CKA/SVCCA
  configs/        # differ only by codec / match_regime / seed
```

Rule: **only** `codec`, `match_regime`, and `seed` differ across comparable runs. Locked `difficulty` (from §2.1 pilot) is shared within a comparison; vary it only when reporting difficulty curves.

---

## 11. Readiness checklist

Training comparison may start when:

1. One seed fully reproduces an episode
2. A–E pass info-equality tests on \(O_t\) (`parse_k ∘ encode_k = F`, shared quantization; §1.1)
3. C has no oracle \(p\) / true causes in training inputs
4. A is essentialist (not eventive); ambient time contract documented (A-ordered default; A-bag optional)
5. E (shuffle/reverse/bag) exists
6. Both match regimes are implemented and logged
7. ≥1 task where a priori \(B>A\) or \(C>A\) is expected
8. Evaluation does not require fluent natural language
9. Multi-seed runner works from one script
10. Difficulty pilot on D locked regime-accuracy ∈ [~0.55, ~0.80] (§2.1)

---

## 12. Explicitly deferred

- Wikipedia / web-scale “translated” corpora
- Full multi-agent emergent communication
- Full Minkowski / non-temporal language
- Billion-parameter models
- Human annotators for “process style”
- Shared state-prediction head during initial training

---

## 13. Timeline (realistic)

| Stage | Time |
|---|---|
| Spec: State / Obs / causal regimes + fact schema \(F\) | 2–4 days |
| Simulator + interventions + difficulty knobs | ~1 week |
| Difficulty pilot on D (lock operating point) | 1–2 days |
| Codecs A–E + info-equality tests (`parse∘encode`) | 1–2 weeks |
| Dataset + sanity checks | 3–5 days |
| Train baselines (2 regimes × seeds) | ~1 week |
| Eval suite + transfer | ~1 week |
| Frozen probes + RSA | 3–7 days |

**MVP end-to-end:** ~4–6 weeks full-time / ~6–10 part-time for the 1D causal toy.

**Cheapest “already visible” slice:** 1D + A/B/D/E + compute+info match + intervention + calibration — often ~2–3 weeks.

---

## 14. Methodological holes closed by this revision

1. **Codec ≠ ontology** — named and controlled via D/E + RSA.
2. **Same model ≠ same experiment** — dual match regimes + full metric table.
3. **C calibration leak** — no oracle \(p\); evidential metadata only.
4. **A too process-like** — rewritten as essentialist captions.
5. **Missing negative control** — E shuffle/reverse/bag (+ optional A-bag).
6. **No seeds** — 3–5 replications mandatory.
7. **Shared probe contamination** — probes only post hoc on frozen models.
8. **World too rich too early** — 1D causation first.
9. **No transfer test** — cross-codec transfer added.
10. **Hard \(f(O)\) contract** — codecs never see \(H_t\) in training.
11. **Temporal leak in A** — ambient time contract; A vs B = lexicalization of change; order destruction via E / A-bag; claim hierarchy in §3/§8.
12. **Info-equality unoperational** — `parse_k ∘ encode_k = F` on shared quantized schema (§1.1); no probe gate.
13. **Causal regimes too easy** — difficulty axis + D pilot before full matrix (§2.1).

---

## 15. Bottom line

The experiment is **feasible and worth doing** if it measures **codec inductive bias** under:

1. \(f(O)\) information parity via `parse∘encode = F`
2. no oracle probabilities in C  
3. essentialist A vs relational B under ambient time (order destruction = E / A-bag)  
4. structured D and destroyed-structure E  
5. compute-matched **and** info-matched reporting  
6. multi-seed stats  
7. 1D causation before 2D physics; difficulty calibrated so regime task does not ceiling  
8. post-hoc probes + representation similarity  

Only on that backbone is it honest to reopen the larger question: whether alternative representational ontologies change what kind of cognition emerges.