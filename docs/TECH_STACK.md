# Tech Stack — Codec World Models

Canonical tooling for this repository. Prefer these versions and libraries; do not add alternate stacks without an explicit methodological / engineering decision.

## Runtime

| Component | Choice |
|---|---|
| Language | **Python 3.13.15** |
| Package manager | **uv** |
| Deep learning | **PyTorch** |
| Array / numerics | **numpy**, **scipy** |
| Tabular analysis | **pandas** |
| Plotting | **matplotlib** |
| Data models | **pydantic** and/or stdlib **dataclasses** |
| Array storage | **zarr** |
| Tests | **pytest** |

## Project files

- `.python-version` — pins the interpreter for uv/pyenv
- `pyproject.toml` — dependencies and tool config (uv)

## Conventions

- All **comments**, **identifiers** (functions, variables, modules, classes), **docstrings**, **logs**, **CLI output**, **metric field names**, and **user-facing reports** must be in **English**.
- Typed APIs; explicit RNG objects; see `.cursor/rules/03-python-conventions.mdc`.
- Scientific contracts: `EXPERIMENT_PLAN.md`, `PROGRAM_SPEC.md`, `INVARIANTS.md`.

## Install (uv)

```bash
uv sync
uv run pytest
```
