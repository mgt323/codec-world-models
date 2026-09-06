"""End-to-end dataset generation: seed ranges + fresh-process checksum identity."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from configs.dataset_v0 import (
    BASE_SEED_OOD_NOISE_RESERVED,
    BASE_SEED_OOD_PARTIAL_OBS,
    BASE_SEED_TRAIN,
    BASE_SEED_VAL,
    N_EPISODES_DRY_RUN,
    N_EPISODES_OOD_PARTIAL_OBS_V0,
    N_EPISODES_TRAIN_V0,
    N_EPISODES_VAL_V0,
    OOD_PARTIAL_OBS_DIFFICULTY,
    active_split_specs,
    assert_episode_seed_ranges_disjoint,
    n_episodes_by_split_for_scale,
)
from world.simulate import DEFAULT_DIFFICULTY_V0

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "generate_dataset.py"


def test_active_specs_use_locked_difficulties() -> None:
    active = {s.name: s for s in active_split_specs()}
    assert set(active) == {"train", "val", "ood_partial_obs"}
    assert active["train"].difficulty == DEFAULT_DIFFICULTY_V0
    assert active["val"].difficulty == DEFAULT_DIFFICULTY_V0
    assert active["ood_partial_obs"].difficulty == OOD_PARTIAL_OBS_DIFFICULTY
    assert OOD_PARTIAL_OBS_DIFFICULTY.partial_obs_rate == 0.5
    assert OOD_PARTIAL_OBS_DIFFICULTY.noise_scale == DEFAULT_DIFFICULTY_V0.noise_scale


def test_base_seeds_documented_and_disjoint_at_dry_run_scale() -> None:
    assert BASE_SEED_TRAIN == 1_000_000
    assert BASE_SEED_VAL == 2_000_000
    assert BASE_SEED_OOD_PARTIAL_OBS == 3_000_000
    assert BASE_SEED_OOD_NOISE_RESERVED == 4_000_000
    assert_episode_seed_ranges_disjoint(active_split_specs(), N_EPISODES_DRY_RUN)
    assert_episode_seed_ranges_disjoint(
        active_split_specs(), n_episodes_by_split_for_scale("v0")
    )
    assert N_EPISODES_TRAIN_V0 == 50_000
    assert N_EPISODES_VAL_V0 == 50_000
    assert N_EPISODES_OOD_PARTIAL_OBS_V0 == 5_000


def test_seed_range_overlap_detected() -> None:
    from configs.dataset_v0 import SplitSpec

    specs = (
        SplitSpec("a", "active", DEFAULT_DIFFICULTY_V0, base_seed=100),
        SplitSpec("b", "active", DEFAULT_DIFFICULTY_V0, base_seed=150),
    )
    with pytest.raises(ValueError, match="overlap"):
        assert_episode_seed_ranges_disjoint(specs, n_episodes=100)


def test_generate_dataset_script_fresh_process_identical_checksums(
    tmp_path: Path,
) -> None:
    """Re-run the script in two fresh processes → identical manifest sha256s."""
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    n_episodes = 6
    n_steps = 4

    for out in (out_a, out_b):
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT),
                "--out-dir",
                str(out),
                "--n-episodes",
                str(n_episodes),
                "--n-steps",
                str(n_steps),
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
        )
        assert result.returncode == 0, (
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    manifest_a = json.loads((out_a / "seed_checksums.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((out_b / "seed_checksums.json").read_text(encoding="utf-8"))

    assert set(manifest_a["splits"]) == {"train", "val", "ood_partial_obs"}
    assert "ood_noise" in manifest_a["deferred_splits"]
    assert manifest_a["n_episodes_by_split"] == {
        "train": n_episodes,
        "val": n_episodes,
        "ood_partial_obs": n_episodes,
    }
    assert manifest_a["n_steps"] == n_steps

    for split_name in manifest_a["splits"]:
        assert (
            manifest_a["splits"][split_name]["sha256"]
            == manifest_b["splits"][split_name]["sha256"]
        )
        assert (out_a / f"{split_name}.h5").is_file()
        assert (out_b / f"{split_name}.h5").is_file()

    # Full manifest equality (same inputs → same recorded metadata + checksums).
    assert manifest_a == manifest_b
