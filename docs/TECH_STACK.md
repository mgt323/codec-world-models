# Tech Stack — Codec World Models

Canonical tooling for this repository. Prefer these versions and libraries; do not add alternate stacks without an explicit methodological / engineering decision.

## Runtime

| Component | Choice |
|---|---|
| Language | **Python 3.13.15** |
| Package manager | **uv** |
| Deep learning | **PyTorch ROCm** (`torch` + `triton-rocm` from the `rocm7.2` wheel index on Linux x86_64; HIP is exposed as `torch.device("cuda")`) |
| Array / numerics | **numpy** (`scipy` is an optional `analysis` extra for offline linear algebra) |
| Tabular I/O | **pandas** + **pyarrow** (Parquet) |
| Array storage | **h5py** (raw `State` trajectories) |
| Data models | stdlib **dataclasses** |
| Plotting | **matplotlib** (optional `analysis` extra; not used by training) |
| Tests | **pytest** |

`zarr` and `pydantic` are not dependencies. Under `INVARIANTS.md` §I15, Stage-1 training on ROCm GPU uses uniform **`bf16_mixed`** autocast across all codecs (keeping FP32 master weights, FP32 AdamW moments, FP32 `LayerNorm`, and FP32 cross-entropy loss), while CPU training and frozen `eval/` / `probes/` scoring stay in **`fp32`**.

## Project files

- `.python-version` — pins the interpreter for uv/pyenv
- `pyproject.toml` — dependencies, the explicit `pytorch-rocm` index, and tool config (uv)
- `uv.lock` — locked resolution, including the ROCm wheel URLs

## Conventions

- All **comments**, **identifiers** (functions, variables, modules, classes), **docstrings**, **logs**, **CLI output**, **metric field names**, and **user-facing reports** must be in **English**.
- Typed APIs; explicit RNG objects; see `.cursor/rules/03-python-conventions.mdc`.
- Scientific contracts: `EXPERIMENT_PLAN.md`, `PROGRAM_SPEC.md`, `INVARIANTS.md`.

## Install (uv)

The locked environment is Linux x86_64. On that platform `torch` and `triton-rocm` come from `https://download.pytorch.org/whl/rocm7.2`. The host ROCm stack for the RX 9070 XT (`gfx1201`) is `/opt/rocm` 7.2. Set `HSA_OVERRIDE_GFX_VERSION=12.0.1` only if `torch.cuda.get_arch_list()` does not include `gfx1201`.

```bash
uv sync --group dev
uv run pytest
```

`uv sync --group analysis` adds `scipy` and `matplotlib` for offline probe and plotting work. They are not required to train or score.
