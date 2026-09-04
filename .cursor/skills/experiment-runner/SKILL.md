---
name: experiment-runner
description: >-
  Run a single reproducible experiment end-to-end: validate config/dataset/seed,
  train, checkpoint, evaluate, and save metrics plus metadata. Use when launching
  training, re-running a seeded experiment, checking match regimes, or when the
  user asks to run, replay, or reproduce a codec-world-models experiment.
disable-model-invocation: true
---

# Experiment Runner

Run **one** comparable experiment unit (one codec × match regime × seed), not the full matrix unless asked.

Follow `EXPERIMENT_PLAN.md` and project rules `01-experiment-integrity`, `02-reproducibility`.

## Workflow

```text
Run progress:
- [ ] 1. Validate config
- [ ] 2. Validate dataset
- [ ] 3. Validate seed
- [ ] 4. Train
- [ ] 5. Checkpoint
- [ ] 6. Evaluate
- [ ] 7. Save metrics
- [ ] 8. Save metadata
```

### 1. Validate config

Confirm before any training:

- `codec`, `match_regime`, `seed` set; locked `difficulty` shared with comparison peers
- architecture / tokenizer / splits identical across codecs in the same comparison
- no codec-specific auxiliary losses or hidden-state inputs
- match regime is either compute-matched or info-matched (not an ad-hoc hybrid)

Refuse to start if config would break a fair comparison — stop and report.

### 2. Validate dataset

- dataset version / checksum present and matches config
- train / val / OOD splits fixed and shared across codecs
- world seed → episode reproducibility smoke check (same seed → same trajectory)

### 3. Validate seed

Record and fix:

- dataset seed
- model initialization seed
- training / RNG seed
- any codec-side shuffle seed (e.g. A-bag, B-shuffle)

No uncontrolled global randomness.

### 4. Train

- single entrypoint from `train/`
- log optimizer steps, tokens, FLOPs estimate, wall-clock as training proceeds
- do not change architecture mid-run per codec

### 5. Checkpoint

- save model weights + exact config used
- include git commit hash in checkpoint sidecar or run dir
- enough to resume or re-evaluate without retraining

### 6. Evaluate

- state/obs/intervention metrics from the plan (not text aesthetics)
- same eval suite and seeds for all codecs in the comparison
- probes / RSA only if requested; default = primary endpoints for this run

### 7. Save metrics

Persist machine-readable metrics (e.g. JSON/CSV) with:

- primary task scores
- token count, optimizer steps, FLOPs estimate, param count
- match regime identifier

### 8. Save metadata

Every run directory must include:

- git commit
- full config
- random seeds (all of the above)
- dataset version/checksum
- codec + matching regime
- model parameter count
- wall-clock

## Report template

```markdown
## Experiment run

**Status:** success | failed | aborted (methodology)

| Field | Value |
|---|---|
| codec | |
| match_regime | |
| seed | |
| difficulty | |
| git | |
| dataset checksum | |
| steps / tokens / FLOPs | |
| metrics path | |
| checkpoint path | |

### Notes
- reproducibility smoke: pass/fail
- methodology unchanged: yes/no
```

Prefer existing `train/` / `eval/` entrypoints; extend them rather than ad-hoc scripts.
