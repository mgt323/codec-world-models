"""Package-wide static guard: obs_codecs must not import latent / oracle types.

PROGRAM_SPEC §4.1: ``obs_codecs/`` ↛ ``LatentState`` (and related H_t channels).
One AST scan covers every module in the package — including future codecs D/E —
so per-codec ad-hoc import checks are not the sole defense.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_OBS_CODECS = _ROOT / "obs_codecs"

# Names that would open a LatentState / oracle side channel into codecs.
_FORBIDDEN_FROM_WORLD_SCHEMA = frozenset(
    {
        "LatentState",
        "CausalRegime",
        "oracle_regime_posterior",
    }
)


def _python_modules(package_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in package_dir.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def _forbidden_imports_in_file(path: Path) -> list[str]:
    """Return human-readable hits for forbidden world.schema imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "world.schema":
            continue

        if node.names and node.names[0].name == "*":
            hits.append(f"line {node.lineno}: from world.schema import *")
            continue

        for alias in node.names:
            if alias.name in _FORBIDDEN_FROM_WORLD_SCHEMA:
                as_clause = f" as {alias.asname}" if alias.asname else ""
                hits.append(
                    f"line {node.lineno}: from world.schema import "
                    f"{alias.name}{as_clause}"
                )

    return hits


def test_obs_codecs_modules_do_not_import_latent_or_oracle_types() -> None:
    """Every obs_codecs/*.py: no LatentState / CausalRegime / oracle_regime_posterior."""
    modules = _python_modules(_OBS_CODECS)
    assert modules, f"expected Python modules under {_OBS_CODECS}"

    offenders: list[str] = []
    for path in modules:
        for hit in _forbidden_imports_in_file(path):
            rel = path.relative_to(_ROOT).as_posix()
            offenders.append(f"{rel}: {hit}")

    assert not offenders, (
        "obs_codecs must not import LatentState, CausalRegime, or "
        "oracle_regime_posterior from world.schema:\n  "
        + "\n  ".join(offenders)
    )


def test_package_wide_scan_covers_all_current_codec_modules() -> None:
    """Sanity: scan includes encode_* and package helpers (won't silently shrink)."""
    scanned = {p.name for p in _python_modules(_OBS_CODECS)}
    expected = {
        "__init__.py",
        "diagnostics.py",
        "encode_a.py",
        "encode_b.py",
        "encode_c.py",
        "encode_d.py",
        "parity_audit.py",
        "transforms_e.py",
    }
    assert expected <= scanned
