# Program Specification — Codec World Models

**Scientific specification:** `EXPERIMENT_PLAN.md`  
**This document:** how the **program** must enforce those contracts in types, modules, and runtime flow.

Code conforms to the experiment. It does not redefine it.

---

## 0. Pipeline

```text
LatentState / full State
        ↓  Observer (Obs)
Observation O_t
        ↓  Codec f_k
TokenSequence X^{(k)}
        ↓  Transformer (shared)
Logits / Prediction
        ↓  Evaluator (train-free targets)
Metrics
```

Training path uses only: `Observation → Codec → Tokens → Model → next-token loss`.  
Evaluation path may join predictions with ground truth from full state / latents.

---

## 1. Type layering

| Type | Contents | Visible to |
|---|---|---|
| `LatentState` (\(H_t\)) | hidden cause, true regime/graph, oracle posteriors, anything not in `Obs` | Simulator, Evaluator, probe targets |
| `State` | full simulator state (observables + latents as needed for dynamics) | Simulator, Observer, Evaluator |
| `Observation` (\(O_t\)) | only fields allowed in the information contract | Codec, Dataset (train), Model inputs via codec |
| `TokenSequence` | codec alphabet tokens | Model, Train |
| `Prediction` | model outputs (logits / decoded preds in eval space) | Evaluator |
| `GroundTruth` | labels derived from State / LatentState / future Obs | Evaluator only |
| `Metrics` | scalar / structured scores | logs, analysis |

**Invariant:** there is no legal path `LatentState → Codec` or `GroundTruth → Train loss` (except targets that are next tokens of the same codec stream — not eval labels).

---

## 2. Component contracts

### Simulator

```text
step: (State, Action | Intervention, RNG) -> State'
simulate: (seed, interventions, difficulty) -> list[State]
```

- Owns dynamics and stochasticity.
- Must be deterministic given seed + inputs.
- May use and store `LatentState` internally.
- Must not call Codec or Model.

### Observer

```text
observe: State -> Observation
```

- Pure projection: strips everything not in the Obs schema.
- Same `Observation` for all codecs at time \(t\).
- Must not add regime labels, oracle \(p\), or other `H_t` fields “for convenience.”

### Codec

```text
encode_k: Observation -> TokenSequence   # or string then tokenize
parse_k:  TokenSequence | string -> FactRecord F
```

- Family \(k \in \{A,B,C,D,E\}\) (+ ordered/bag/shuffle variants as in the plan).
- **MUST NOT** depend on `LatentState` / full `State` / evaluator outputs.
- Signature accepts **only** `Observation` (and codec-local RNG for E / A-bag shuffles).
- Info-equality: \(\mathrm{parse}_k(\mathrm{encode}_k(O)) = F(O)\) with shared quantization (§1.1 of the plan).

#### Codec A — Substance / essentialist caption

Format and style follow `EXPERIMENT_PLAN.md` §3 (entities + static attributes; no eventive chains).

The Relation line is a fixed, value-independent categorical utterance
by design — it is NOT meant to be truthful or informative about actual
A/B similarity. This is intentional: it demonstrates substance ontology's
blindness to dynamic relational content, without introducing eventive
language. It contributes zero bits to FactRecord and MUST remain
excluded from parity/info-equality checks. Do not 'fix' it to reflect
real similarity — that would make A relational and collapse the A/B
contrast the experiment depends on.

#### Codec B — Process / relational event chain

Format (per timestep):

```text
(observe A=<val>) → (co-vary B=<val>) → (meta n=<n> source=<src> noise=<bin>)
```

Codec B does not use directional markers (e.g. '↑') at the
single-Observation encoding level, since no genuine trend
information is available without prior-timestep context — this
is a deliberate omission, not a missing feature. Noise/confidence
metadata is expressed once, under the 'meta' event, using the
same underlying field as Codec C's evidential 'noise' — never
mislabeled as relational 'strength'.

### Model

```text
forward: TokenSequence -> Logits
```

- One shared architecture for all codecs in a comparison.
- Stage-1 objective: next-token prediction in that codec’s alphabet only.
- Must not take `Observation`, `State`, or `LatentState` as side inputs.

### Trainer

```text
train_step: (TokenSequence batch, Model) -> loss
```

- **MUST NOT** depend on evaluator targets (regime id, intervention outcomes, counterfactuals, probe labels).
- Loss = autoregressive NLL (or declared equivalent) on codec tokens.
- No auxiliary heads trained in stage 1 that require `LatentState`.

### Evaluator

```text
evaluate: (Prediction, GroundTruth, …) -> Metrics
```

- **MAY** depend on `LatentState` / full `State` (hidden cause, true graph, counterfactual twins).
- Runs **after** (or separately from) training; does not feed gradients in stage 1.
- Metrics live in observation / state / intervention space — not text aesthetics.

### Interventions (option B — hidden `do()`)

`State` may record `active_intervention` for simulation and evaluation.  
`observe()` **must not** copy intervention flags into `Observation`.  
Codecs and stage-1 training therefore never see an explicit `do()` / `is_intervened` signal.

**Eval task #3 (Intervention accuracy) does not assume the model knows about the intervention at training time — ground truth about `do()` is available only to the evaluator, not to the model.**

The model must infer interventional structure (if at all) from the observation stream alone; the evaluator may use full `State` / intervention records when scoring predicted effects under `do(A)` / `do(B)`.

### Probes (post hoc)

```text
probe: frozen activations + GroundTruth -> probe Metrics
```

- Fit only on frozen models; identical probe protocol across codecs.

### World v0 default difficulty (pilot lock — closed)

Locked from `eval/difficulty_sweep_noise_c0.json`,
`eval/difficulty_sweep_noise015_confound.json`, and
`eval/difficulty_sweep_noise015_partial.json` (O-only heuristic + confusion
matrices; not from the LatentState cheat oracle).

Single source of truth in code: `world.simulate.DEFAULT_DIFFICULTY_V0`
(re-exported from `world`). Do not duplicate field literals.

```text
DEFAULT_DIFFICULTY_V0 = Difficulty(
    noise_scale=0.15,
    regime_switch_rate=0.0,
    partial_obs_rate=0.2,
    confounding_strength=0.0,
)
```

- **`noise_scale=0.15`:** stable directional heuristic recall (~88% on
  `a_causes_b` / `b_causes_a`); away from the NoiseBin LOW/MEDIUM boundary
  at 0.1.
- **`confounding_strength=0.0`:** confound sweep at fixed `noise_scale=0.15`
  did **not** collapse spurious’s distinct heuristic recall (~23–25/25 at
  0.0–0.2; mild dip to 22/25 at 0.3). Keep 0.0 as the default; no need to
  raise confound for separability under the current O-only rule.
- **`partial_obs_rate=0.2`:** chosen from a difficulty sweep at fixed
  `noise_scale=0.15`: directional-regime recall stays ~0.70-0.72 through
  rate 0.3, degrading only at rate 0.5 (~0.61, more `a_causes_b`→spurious
  confusion). 0.2 preserves full heuristic accuracy while giving
  eval task #6 (partial-observability stress) a non-trivial fraction
  (~19%) of under-observed timesteps during training.
- **`regime_switch_rate=0.0`:** fixed regime per episode (World v0).

All four `Difficulty` fields are locked. Comparable runs that claim this
default must log / load `DEFAULT_DIFFICULTY_V0` (or an exact field match).

**Accepted ceiling (eval task #2 / regime classification from one episode):**
Under World v0, `hidden_c` is fixed within an episode, so within-episode
statistics cannot separate `common_cause` from `spurious` (documented in
`eval/regime_baseline.py`; confirmed by explore diagnostics). The difficulty
pilot / O-only heuristic gate therefore applies to the **directional subset**
(`a_causes_b`, `b_causes_a`); the `common_cause`/`spurious` collapse is a
**known identifiability ceiling**, not a defect in `simulate.py` or the
baseline harness. This ceiling is unchanged under the locked
`partial_obs_rate=0.2` — the confusion pattern (common_cause → mostly
spurious) was stable across all `partial_obs_rate` values tested in
`difficulty_sweep_noise015_partial.json`. Do not “fix” by changing
dynamics without an explicit methodological change to `EXPERIMENT_PLAN.md`.

### Dataset splits

Machine-readable difficulty contract for `data/generate_episodes.py`.
Do not invent OOD field values outside this table.

**Split inventory (v0):** multiple **single-axis** OOD splits — not one
combined “everything harder” Difficulty. Attribution for eval task #7
requires knowing which axis failed. Combine axes only in a later, named
multi-axis OOD suite after each single-axis cell is locked.

| Split name | Status (v0) | Difficulty relative to train |
|---|---|---|
| `train` | active | `DEFAULT_DIFFICULTY_V0` |
| `val` | active | `DEFAULT_DIFFICULTY_V0` (same Difficulty; disjoint seeds / episode IDs) |
| `ood_partial_obs` | **active** | train, but `partial_obs_rate=0.5` |
| `ood_noise` | deferred | see axis table — no v0 generation |
| `ood_confound` | deferred | see axis table — no v0 generation |
| `ood_regime_switch` | deferred | see axis table — no v0 generation |

**Axis table** (train vs intended OOD escalation):

| Axis | Train (`DEFAULT_DIFFICULTY_V0`) | OOD | Rationale |
|------|-------------------------------|-----|-----------|
| noise_scale | 0.15 | **deferred** — not varied in v0 OOD | Noise sweep at `confounding_strength=0.0` (`difficulty_sweep_noise_c0.json`) kept heuristic accuracy ~0.65–0.72 across **0.05–0.3**; no interior point is a clear “harder than train” OOD. Escalation must go **beyond 0.3** (untested). Do **not** pick an untested value (e.g. 0.5) until a confirmation sweep exists. `generate_episodes.py` must not emit `ood_noise` until that sweep locks a value. |
| partial_obs_rate | 0.2 | **0.5** (`ood_partial_obs` only) | Partial-obs sweep at fixed `noise_scale=0.15` (`difficulty_sweep_noise015_partial.json`): heuristic 0.70→**0.61** at 0.5; `a_causes_b` recall 24→**16**/25. Empirically harder than train; other axes stay at train values in this split. |
| confounding_strength | 0.0 | **deferred** — not varied in v0 OOD | Confound sweep at `noise_scale=0.15` (`difficulty_sweep_noise015_confound.json`) only covered 0.0–0.3; mild dip (heuristic 0.70→0.65 at 0.3), not a strong OOD gate. Values **>0.3** are untested. Keep train at 0.0; do not invent an OOD confound until a harder cell is piloted. |
| regime_switch_rate | 0.0 | **deferred** — not varied in v0 OOD | EXPERIMENT_PLAN §2.1 / eval task #7 name mid-episode regime switches as an OOD axis, **separate** from noise / partial-obs. World v0 `simulate.py` raises `NotImplementedError` for `regime_switch_rate > 0`. `generate_episodes.py` must not request switches until dynamics implement them; when implemented, use a dedicated `ood_regime_switch` split (other axes = train), not a combined mega-OOD. |

**v0 generator rules (normative for `data/generate_episodes.py`):**

1. Emit `train` and `val` under `DEFAULT_DIFFICULTY_V0` only.
2. Emit exactly one active OOD split: `ood_partial_obs` =
   `Difficulty(noise_scale=0.15, regime_switch_rate=0.0, partial_obs_rate=0.5, confounding_strength=0.0)`.
3. Do **not** generate `ood_noise`, `ood_confound`, or `ood_regime_switch` until the corresponding axis is unlocked above.
4. Do **not** create a combined OOD Difficulty that raises multiple axes at once in v0.

**`ood_noise` split deferred for v0 MVP** — no `noise_scale` value beyond the
tested 0.05–0.3 range has been empirically characterized. Eval task #7 (OOD
generalization) for the v0 MVP uses `ood_partial_obs` only. `ood_noise`
requires a dedicated sweep beyond 0.3 before it can be locked — tracked as a
follow-up, not blocking the current training matrix.

---

## 3. Hard access rules (non-negotiable)

```text
Codec      MUST NOT depend on LatentState.
Evaluator  MAY     depend on LatentState.
Training   MUST NOT depend on evaluator targets.
```

Corollaries:

1. `encode_*` APIs take `Observation`, never `State`.
2. Dataset builders materialize train examples as `(tokens,)` or `(tokens, next_tokens)` from `Observation` only; latents stored in parallel files for eval, not passed into `encode`.
3. Train loop imports must not reference evaluator label constructors.
4. C must not receive oracle \(p\) or true causes unless those fields are part of `Observation` for **all** codecs (they must not be).
5. Intervention / `do()` records stay on `State` (and evaluator ground truth); they are **not** Observation fields (option B above).
---

## 4. How the program guarantees this

### 4.1 Module boundaries

```text
world/       State, LatentState, Observer, simulate
obs_codecs/  encode_*/parse_*  — import Observation only
             (named obs_codecs to avoid shadowing stdlib codecs)
data/        builds token datasets from Observation; latents in eval-only stores
model/       Transformer(tokens -> logits)
train/       next-token loop; no eval target imports
eval/        Prediction + GroundTruth(+ LatentState) -> Metrics
probes/      frozen activations only
```

Forbidden edges (lint / review / auditor):

- `obs_codecs/` → `LatentState` or eval label modules
- `train/` → `eval/` target builders or `LatentState`
- `model/` → `world` state types as forward inputs

### 4.2 Types and constructors

- `Observation` is a distinct type/dataclass from `State`.
- Prefer `observe(state) -> Observation` as the **only** bridge into codecs.
- Tests that call `encode_k(state)` are rejected; use `encode_k(observe(state))`.

### 4.3 Automated gates

| Gate | Guarantees |
|---|---|
| Info-equality suite | codecs carry the same \(F(O)\); no silent extra facts |
| Leakage test | same \(O\), different \(H\) ⇒ identical encodings |
| Train import / API test | training batch schema has no regime/graph/`p` fields |
| Reproducibility | same world seed ⇒ same `list[State]` / same Obs stream |
| Config discipline | comparable runs differ only by `codec`, `match_regime`, `seed` (shared locked `difficulty`) |

### 4.4 Runtime data flow (one timestep)

```text
state = simulate(...)
obs = observe(state)           # strips H
tokens = encode_k(obs)         # no access to state
logits = model(tokens)
loss = nll(logits, tokens)     # train: tokens only

# eval-only (no grad into stage-1 train):
gt = ground_truth(state, ...)  # may use LatentState
metrics = evaluate(pred, gt)
```

---

## 5. Matching regimes (program view)

| Regime | What the trainer holds fixed |
|---|---|
| Compute-matched | FLOPs or token-updates (and logged architecture) |
| Info-matched | number of episodes / states \(O_{1:T}\) |

Token-update accounting for compute-matched runs uses the locked provisional scheme `regex_v3` (`obs_codecs.diagnostics.TOKENIZATION_SCHEME`) until a trained tokenizer is declared. Optional C-shuffle / A-bag are reorder-only and must not add punctuation without a scheme version bump and budget recomputation.

Both must log: git commit, config, seeds, dataset checksum, param count, tokens, steps, FLOPs estimate (see reproducibility rule).

---

## 6. Relation to other docs

| Doc | Role |
|---|---|
| `EXPERIMENT_PLAN.md` | what we claim scientifically; codecs A–E; eval tasks; difficulty |
| `PROGRAM_SPEC.md` (this file) | typed pipeline and access control that make those claims implementable |
| `TECH_STACK.md` | Python 3.13.15 + PyTorch + uv + approved libraries |
| `INVARIANTS.md` | I1–I15 enforcement checklist |
| `.cursor/rules/01-experiment-integrity.mdc` | agent must not silently break contracts |
| `.cursor/skills/codec-audit` | verify information parity / leakage |
| `.cursor/agents/experiment-auditor` | read-only methodology audit |

If code and plan diverge, **fix the code** or flag an explicit methodological change to the plan — do not silently widen codec or train access.
