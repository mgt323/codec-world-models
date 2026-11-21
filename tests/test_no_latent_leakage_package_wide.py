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


def _leakage_import_hits(path: Path) -> list[str]:
    """Return hits for latent-type imports and any ``eval`` import."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "eval" or alias.name.startswith("eval."):
                    hits.append(f"line {node.lineno}: import {alias.name}")
                elif alias.name == "world.schema":
                    hits.append(f"line {node.lineno}: import {alias.name}")
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        module = node.module or ""
        if module == "eval" or module.startswith("eval."):
            hits.append(f"line {node.lineno}: from {module} import ...")
            continue
        if module not in {"world.schema", "world"}:
            continue
        if node.names and node.names[0].name == "*":
            hits.append(f"line {node.lineno}: from {module} import *")
            continue
        for alias in node.names:
            if alias.name in _FORBIDDEN_FROM_WORLD_SCHEMA:
                as_clause = f" as {alias.asname}" if alias.asname else ""
                hits.append(
                    f"line {node.lineno}: from {module} import "
                    f"{alias.name}{as_clause}"
                )

    return hits


def test_obs_codecs_modules_do_not_import_latent_or_oracle_types() -> None:
    """Every obs_codecs/*.py: no LatentState / CausalRegime / oracle_regime_posterior."""
    modules = _python_modules(_OBS_CODECS)
    assert modules, f"expected Python modules under {_OBS_CODECS}"

    offenders: list[str] = []
    for path in modules:
        for hit in _leakage_import_hits(path):
            rel = path.relative_to(_ROOT).as_posix()
            offenders.append(f"{rel}: {hit}")

    assert not offenders, (
        "obs_codecs must not import LatentState, CausalRegime, "
        "oracle_regime_posterior, or eval:\n  "
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
        "vocab.py",
    }
    assert expected <= scanned


def test_train_and_model_do_not_import_latent_state_or_eval() -> None:
    """train/ and model/ must not import LatentState or the eval package."""
    offenders: list[str] = []
    for package in ("train", "model"):
        modules = _python_modules(_ROOT / package)
        assert modules, f"expected Python modules under {package}/"
        for path in modules:
            for hit in _leakage_import_hits(path):
                rel = path.relative_to(_ROOT).as_posix()
                offenders.append(f"{rel}: {hit}")
    assert not offenders, (
        "train/ and model/ must not import LatentState, CausalRegime, "
        "oracle_regime_posterior, or eval:\n  " + "\n  ".join(offenders)
    )


def test_leakage_scanner_flags_latent_and_eval_imports(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from world.schema import SCHEMA_VERSION, LatentState\n"
        "from eval.score import EVAL_SCHEMA_VERSION\n",
        encoding="utf-8",
    )
    hits = _leakage_import_hits(bad)
    assert any("LatentState" in hit for hit in hits)
    assert any("eval.score" in hit for hit in hits)

    allowed = tmp_path / "allowed.py"
    allowed.write_text(
        "from world.schema import SCHEMA_VERSION\n",
        encoding="utf-8",
    )
    assert _leakage_import_hits(allowed) == []
