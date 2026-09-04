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

### Probes (post hoc)

```text
probe: frozen activations + GroundTruth -> probe Metrics
```

- Fit only on frozen models; identical probe protocol across codecs.

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

---

## 4. How the program guarantees this

### 4.1 Module boundaries

```text
world/     State, LatentState, Observer, simulate
codecs/    encode_*/parse_*  — import Observation only
data/      builds token datasets from Observation; latents in eval-only stores
model/     Transformer(tokens -> logits)
train/     next-token loop; no eval target imports
eval/      Prediction + GroundTruth(+ LatentState) -> Metrics
probes/    frozen activations only
```

Forbidden edges (lint / review / auditor):

- `codecs/` → `LatentState` or eval label modules
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
