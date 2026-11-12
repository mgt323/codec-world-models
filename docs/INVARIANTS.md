# Invariants — Codec World Models

Machine-checkable and review-enforced laws. Violating any invariant invalidates the affected comparison unless an explicit methodological change is approved and recorded.

**Related:** `EXPERIMENT_PLAN.md` (science), `PROGRAM_SPEC.md` (typed pipeline), `.cursor/rules/01-experiment-integrity.mdc`.

---

| ID | Invariant |
|---|---|
| **I1** | All codecs consume exactly `Observation`. |
| **I2** | No training component may read hidden state. |
| **I3** | World seed determines episode exactly. |
| **I4** | Dataset split is identical across codecs. |
| **I5** | Model architecture is identical. |
| **I6** | Optimizer configuration is identical. |
| **I7** | Evaluation uses ground-truth world state independently of codec. |
| **I8** | C receives no oracle probabilities or causal labels. |
| **I9** | A contains no explicit event-chain structure. |
| **I10** | B preserves event ordering. |
| **I11** | E destroys B organization while preserving token inventory as closely as practical. |
| **I12** | Probes are trained only after model freezing. |
| **I13** | Every reported result is tied to a git commit. |
| **I14** | All primary comparisons use multiple training seeds. |
| **I15** | Methodological changes require explicit approval. |

---

## I1 — Codecs consume exactly Observation

`encode_k` accepts only `Observation` (plus codec-local RNG for shuffle/bag variants). No `State`, no `LatentState`, no evaluator outputs.

**Check:** API types; codec-audit same-\(O\)/different-\(H\) encodings match.

## I2 — Training never reads hidden state

Trainer, train batches, and stage-1 losses use codec tokens only. No regime id, true graph, hidden masses, or oracle \(p\) in the training path.

**Check:** batch schema; import boundaries (`train/` ↛ latents / eval targets).

## I3 — World seed determines episode exactly

Same `(seed, interventions, difficulty)` ⇒ identical `list[State]` (and thus identical `Observation` stream).

**Check:** determinism test; no uncontrolled global RNG in `simulate`.

## I4 — Dataset split identical across codecs

Train / val / OOD partitions are defined on world episodes (or Obs streams), then encoded. Codecs do not redefine splits.

**Check:** shared split manifest / checksum across codec runs in a comparison.

## I5 — Model architecture identical

Within a comparable run set, architecture, depth/width, vocab handling policy (aside from codec alphabet content), and parameterisation family are fixed. Only `codec` (and thus token content) differs among experimental conditions—not the network definition.

**Check:** config hash excluding codec identity fields; param count parity (allowing vocab-size edge effects only if declared).

## I6 — Optimizer configuration identical

Same optimizer class, learning rate schedule, batching policy, and step budget rules under the declared match regime.

**Check:** shared optimizer config; regime-specific budgets applied uniformly across codecs.

## I7 — Evaluation is codec-independent ground truth

Metrics compare model predictions to ground truth from world state / latents / future observations—not to codec prose quality. The same GT applies regardless of which codec trained the model.

**Check:** eval entrypoints take `GroundTruth` from `State`/`LatentState`; no codec-specific metric definitions for primary endpoints.

## I8 — C has no oracle probabilities or causal labels

Codec C may include evidential metadata that is part of `Observation` for all codecs (e.g. noise, \(n\), source). It must not include simulator posteriors or true causes.

**Check:** encode_C fixtures; leakage / info-equality suite.

## I9 — A has no explicit event-chain structure

A is essentialist captions (entities + static attributes). No eventive chains, `→` process markers, or “X hits Y” style dynamics language. Default A-ordered still concatenates per-timestep blocks; that is ambient time, not event-chain vocabulary (see plan ambient-time contract).

**Check:** encode_A snapshots / style tests; reject process-like templates.

## I10 — B preserves event ordering

Process codec B emits relational/event structure in temporal order with explicit transition scaffolding.

**Check:** encode_B on known trajectories; order of events matches world time.

## I11 — E destroys B organization, keeps token inventory

E variants (`B-shuffle`, `B-reverse`, `B-bag`, optional `A-bag` / `C-shuffle`) break organization (order/structure) while preserving token multiset as closely as practical. Fact recoverability follows plan §1.1 (multiset, not order).

**Check:** token-inventory / multiset tests; order destroyed; info-equality on facts.

## I12 — Probes only after freeze

Linear (or other) probes train on frozen activations. No shared state head during stage-1 training that forces representational convergence.

**Check:** probe pipeline loads checkpoint with `requires_grad=False` / eval mode; no probe loss in `train/`.

## I13 — Results tied to git commit

Every reported metric bundle records the git commit (and dirty-tree flag if any). Untagged numbers are not results.

**Check:** run metadata schema; analysis refuses rows without commit.

## I14 — Primary comparisons use multiple seeds

Primary codec comparisons require **3–5** training seeds; report median + IQR / bootstrap. Single-seed “wins” are exploratory only.

**Check:** result-analysis / experiment-runner gates on \(n_{\text{seeds}}\).

## I15 — Methodological changes need explicit approval

Changing information contract, codec definitions, matching methodology, eval targets, primary endpoints, or seed/split policy requires an explicit methodological flag and plan update—not a silent code edit.

**Check:** review / experiment-auditor; PR or chat must label the change; update `EXPERIMENT_PLAN.md` / this file when invariants move.

---

## Enforcement map

| Layer | Role |
|---|---|
| Types + module imports | I1, I2, I7, I12 |
| World + RNG tests | I3 |
| Data manifests | I4 |
| Shared configs | I5, I6 |
| Codec tests + codec-audit | I8, I9, I10, I11 |
| Run metadata | I13, I14 |
| Human / auditor approval | I15 |

When an invariant and an implementation disagree, **fix the implementation** or obtain approval under I15—do not weaken the invariant quietly.
