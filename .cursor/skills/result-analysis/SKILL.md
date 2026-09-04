---
name: result-analysis
description: >-
  Analyze experiment results across codecs and seeds: summaries, metrics,
  confidence intervals, seed dispersion, comparisons, plots, and anomalous runs.
  Use when reviewing run outputs, comparing A/B/C/D/E, drafting result writeups,
  or when the user asks to analyze, plot, or interpret codec-world-models metrics.
disable-model-invocation: true
---

# Result Analysis

Analyze completed runs without changing methodology or re-training unless asked.

Follow `EXPERIMENT_PLAN.md` §5 (matching), §8 (allowed conclusions), and multi-seed reporting (median + IQR / bootstrap).

## Workflow

```text
Analysis progress:
- [ ] 1. Summary
- [ ] 2. Metrics
- [ ] 3. Confidence intervals
- [ ] 4. Seed dispersion
- [ ] 5. Comparison
- [ ] 6. Plots
- [ ] 7. Anomalous runs
```

### 1. Summary

Collect run dirs / metric files for the requested slice:

- codecs, match regimes, seeds, difficulty
- git commit(s), dataset checksum(s)
- flag mixed commits or checksum mismatches before interpreting gaps

### 2. Metrics

Load primary endpoints (state/obs/intervention space). Keep compute diagnostics separate:

- task scores (e.g. regime, intervention, calibration)
- tokens, steps, FLOPs, params, wall-clock

Never average across different `match_regime` or unlocked `difficulty` without splitting.

### 3. Confidence intervals

Per codec × regime × difficulty (and per task):

- median over seeds
- IQR and/or bootstrap CI (state the method and \(n_{\text{seeds}}\))

Do not treat a single seed as a result.

### 4. Seed dispersion

Report seed-to-seed spread:

- range / IQR / std of each primary metric
- note when dispersion ≥ codec gap (comparison unreliable)

### 5. Comparison

Pairwise / ordered contrasts only under matched conditions:

- same match regime, difficulty, eval suite, data checksum family
- apply claim hierarchy from the plan (A-ordered vs B; E / A-bag; D as structure control)
- separate compute-matched vs info-matched verdicts if they disagree

Allowed conclusions only per §8 — no ontology shortcuts.

### 6. Plots

Prefer simple, labeled figures:

- metric vs codec (with seed jitter or error bars)
- metric vs `noise_scale` / difficulty when available
- optional: tokens/FLOPs vs score for cost-aware reading

Save under a clear path (e.g. `results/figures/`); do not bury one-off notebooks as the only artifact.

### 7. Anomalous runs

Flag runs that are:

- failed / incomplete / missing metadata
- extreme outliers vs other seeds of the same cell
- non-reproducible checksum / seed mismatch
- accidental methodology drift (different arch, split, or codec info)

State whether to exclude (with reason) or keep and discuss.

## Report template

```markdown
## Result analysis

### Slice
- codecs / regimes / seeds / difficulty
- git / dataset checksum consistency: ok | mixed

### Metrics (median [IQR] or CI)
| Task | A | B | D | E | … |
|---|---|---|---|---|---|

### Seed dispersion
- …

### Comparison
- direction under info-matched:
- direction under compute-matched:
- allowed conclusion (§8):

### Plots
- path — what it shows

### Anomalous runs
| Run | Issue | Action |
|---|---|---|

### Methodology changed?
- No (analysis only)
```

Prefer scripts in `eval/` / `probes/` over ad-hoc one-off analysis when extending the pipeline.
