"""Schema checks for metrics_eval.json and the tier manifest."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from eval.metrics import METRIC_KEYS
from eval.score import DECODER_MODES, EVAL_SCHEMA_VERSION

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "verify_tier_artifacts.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_tier_artifacts", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_doc() -> dict[str, object]:
    split = {key: 0.0 for key in METRIC_KEYS}
    return {
        "splits": {
            name: dict(split)
            for name in ("val", "ood_partial_obs", "intervention", "counterfactual")
        },
        "scoring_checkpoint": "final",
        "git_commit": "abc123",
        "git_dirty": False,
        "eval_schema_version": EVAL_SCHEMA_VERSION,
        "decoder_mode": DECODER_MODES[0],
    }


def test_eval_document_accepts_metric_keys_and_eval_v2_fields() -> None:
    script = _load_script()
    assert script.eval_document_problems("cell", _valid_doc()) == []
    assert script.manifest_eval_contract_problems(
        {
            "eval_schema_version": EVAL_SCHEMA_VERSION,
            "decoder_mode": DECODER_MODES[0],
        }
    ) == []


def test_eval_document_rejects_missing_metric_and_schema_fields() -> None:
    script = _load_script()
    doc = _valid_doc()
    splits = doc["splits"]
    assert isinstance(splits, dict)
    del splits["val"]["parse_fail_rate"]
    doc["eval_schema_version"] = "eval_v1"
    doc["decoder_mode"] = "oracle"
    problems = script.eval_document_problems("cell", doc)
    assert any("parse_fail_rate" in problem for problem in problems)
    assert any("eval_schema_version" in problem for problem in problems)
    assert any("decoder_mode" in problem for problem in problems)
    manifest_problems = script.manifest_eval_contract_problems({})
    assert any("eval_schema_version" in problem for problem in manifest_problems)
    assert any("decoder_mode" in problem for problem in manifest_problems)
