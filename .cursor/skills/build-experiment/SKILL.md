---
name: build-experiment
description: >-
  Implement experiment components against EXPERIMENT_PLAN.md without silently
  changing methodology. Use when adding or modifying world/, codecs/, data/,
  model/, train/, eval/, or probes/; when the user asks to build, implement,
  or extend the experiment; or when running a minimal end-to-end slice.
disable-model-invocation: true
---

# Build Experiment

Implement code so it **conforms to** `EXPERIMENT_PLAN.md`. Do not redefine the experiment in code.

## Workflow

Copy and track:

```text
Build progress:
- [ ] 1. Read EXPERIMENT_PLAN.md (relevant sections only)
- [ ] 2. Inspect relevant modules
- [ ] 3. Identify invariant interfaces
- [ ] 4. Implement
- [ ] 5. Add tests
- [ ] 6. Run minimal experiment
- [ ] 7. Report files changed
- [ ] 8. Report whether methodology changed
```

### 1. Read EXPERIMENT_PLAN

Read only the sections that govern the requested change (e.g. §1 info contract, §2 world, §3 codecs, §5 matching, §7 eval).

Treat the plan as the scientific specification. Project rules `01-experiment-integrity` and `02-reproducibility` apply.

### 2. Inspect relevant modules

Map the request to layout:

```text
world/    codecs/    data/    model/    train/    eval/    probes/    configs/
```

Inspect existing interfaces, configs, and tests before writing new code.

### 3. Identify invariant interfaces

Before coding, list what must **not** change unless the user explicitly accepts a methodological change:

- `O_t = Obs(S_t)`; codecs never read `H_t`
- codec definitions A–E (incl. ambient time / A-bag; no oracle `p` in C)
- info-equality: `parse_k ∘ encode_k = F` on shared quantized schema
- matched-compute and matched-information regimes
- shared architecture / tokenizer / splits across codecs in a comparison
- seed and difficulty handling (§2.1)

If the request would break an invariant → **stop**, explain risk, name the affected comparison, propose the smallest valid alternative.

### 4. Implement

- Smallest change that satisfies the plan
- Typed Python 3.12; dataclasses; explicit RNG objects; no hidden globals
- Do not mix simulation, encoding, and training in one function
- Config diffs across comparable runs: only `codec`, `match_regime`, `seed` (shared locked `difficulty`)

### 5. Add tests

Prefer pytest. Minimum:

- determinism: same world seed → same episode
- invariants touched by the change (e.g. info-equality, no `H_t` in codec inputs)
- one smoke test on the new path

### 6. Run minimal experiment

Run the smallest command that exercises the change (unit/smoke, or one seeded episode / tiny train step). Do not launch the full matrix unless asked.

### 7–8. Report

End with:

```markdown
## Files changed
- path — why

## Methodology changed?
- **No** — implementation only; plan invariants preserved
- **Yes** — what changed, which comparison is affected, whether EXPERIMENT_PLAN.md was updated
```

If methodology changed and the plan was not updated, say so explicitly and offer to update the plan.
