---
name: codec-audit
description: >-
  Audit codecs A–E for identical recoverable information under the experiment
  info-equality contract. Use when checking encode_A..E, info-equality tests,
  hidden-state leakage, oracle metadata in C, or when the user asks to verify
  that codecs carry the same facts from O_t.
disable-model-invocation: true
---

# Codec Audit

**Cel:** Sprawdź, czy A/B/C/D/E rzeczywiście mają identyczną informację.

Treat `EXPERIMENT_PLAN.md` §1 / §1.1 as the contract. Equality is
`parse_k ∘ encode_k = F(O)` on a shared quantized fact schema — not string equality across codecs.

## Workflow

```text
Audit progress:
- [ ] 1. Generate same observations
- [ ] 2. Encode all codecs
- [ ] 3. Compare recoverable facts
- [ ] 4. Detect hidden-state leakage
- [ ] 5. Detect additional metadata
- [ ] 6. Report violations
```

### 1. Generate same observations

From the world simulator, build a fixed set of `O_t` (and short trajectories if needed):

- locked seeds / checksummed grid
- cover partial obs, noise bins, all regime types, interventions if present in `Obs`
- never feed `H_t` into encoders

### 2. Encode all codecs

For each `O` and each `k ∈ {A, B, C, D, E}` (incl. A-ordered / A-bag / B-shuffle variants if implemented):

- `x_k = encode_k(O)`
- `f_k = parse_k(x_k)` → canonical fact record `F`

If `parse_k` is missing, that is itself a violation of §1.1 — implement or flag before claiming equality.

### 3. Compare recoverable facts

Assert against canonical `F(O)` (shared quantization):

- `f_k == F(O)` for every primary codec
- cross-codec: `f_A == f_B == f_C == f_D` (and E on **fact multiset**, not order)

Do **not** require `encode_A(O) == encode_B(O)` as strings.

### 4. Detect hidden-state leakage

Fail if any encoded string / parsed fields contain (or are deterministic functions only of):

- true causal graph / regime label from `H`
- hidden masses / latent `C` when not in `O`
- oracle posteriors `p(...)`
- simulator-only fields not in `Obs`

Spot-check: same `O`, different `H` → encodings must be identical.

### 5. Detect additional metadata

Fail if one codec exposes facts others cannot recover from the same `O`, e.g.:

- C with oracle `p` or true causes
- per-codec-only fields not in shared `F`
- different quantization / precision across codecs
- order markers counted as *content* equality for E (order may differ; facts must not)

### 6. Report violations

```markdown
## Codec audit

**Verdict:** PASS | FAIL

### Setup
- seeds / N observations
- codecs audited
- schema `F` fields

### Violations
| Severity | Codec | Issue | Evidence |
|---|---|---|---|
| critical | C | oracle p in input | `p=0.73` in encode_C |
| critical | A | H leakage | encoding changes when only H changes |
| major | B | extra field | `strength` not in F/O |
| minor | D | quantization drift | A_val_q bin edges differ |

### Info-equality
- parse∘encode == F: yes/no per codec
- cross-codec fact match: yes/no

### Methodology impact
- Does fixing require EXPERIMENT_PLAN change? yes/no
```

Severity: **critical** = validity broken (leak / unequal info); **major** = contract gap; **minor** = hygiene.

Prefer running existing `obs_codecs/` info-equality tests if present; extend them rather than one-off notebooks.
