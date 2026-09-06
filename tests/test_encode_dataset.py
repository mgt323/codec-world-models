"""Tests for HDF5 episode round-trip and offline codec encoding."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from data.encode_episodes import (
    ALL_VARIANT_STEMS,
    EncodingValidationError,
    CodecVariant,
    avg_tokens_per_string,
    build_encoded_frame,
    codec_variants,
    encode_dataset,
    encode_split_variant,
)
from data.generate_episodes import generate_split
from data.hdf5_episodes import read_episodes_hdf5, write_episodes_hdf5
from obs_codecs.encode_a import encode_A
from obs_codecs.transforms_e import derive_transform_seed, shuffle_b
from world.schema import observe
from world.simulate import DEFAULT_DIFFICULTY_V0

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "encode_dataset.py"


def _tiny_episodes(n_episodes: int = 4, n_steps: int = 3, base_seed: int = 50_000):
    return list(
        generate_split(
            "train",
            n_episodes,
            DEFAULT_DIFFICULTY_V0,
            base_seed,
            n_steps,
            intervention_fraction=0.0,
        )
    )


def test_hdf5_write_read_roundtrip(tmp_path: Path) -> None:
    episodes = _tiny_episodes()
    path = tmp_path / "train.h5"
    write_episodes_hdf5(path, episodes, split_name="train")
    loaded = read_episodes_hdf5(path)
    assert len(loaded) == len(episodes)
    for (seed_a, states_a), (seed_b, states_b) in zip(episodes, loaded, strict=True):
        assert seed_a == seed_b
        assert states_a == states_b


def test_encode_split_variant_writes_parquet_and_validates(tmp_path: Path) -> None:
    episodes = _tiny_episodes(n_episodes=6, n_steps=4)
    out_dir = tmp_path / "encoded"
    reports = []
    for variant in codec_variants():
        report = encode_split_variant(
            episodes,
            split_name="train",
            variant=variant,
            out_dir=out_dir,
            sample_size=1000,
            validation_seed=1,
        )
        reports.append(report)
        assert report.ok
        assert report.validation_pass_rate == 1.0
        assert report.record_count == 6 * 4
        path = Path(report.path)
        assert path.is_file()
        frame = pd.read_parquet(path)
        assert list(frame.columns) == ["episode_seed", "timestep", "encoded_string"]
        assert len(frame) == report.record_count

    assert {r.variant for r in reports} == set(ALL_VARIANT_STEMS)


def test_e_shuffle_uses_derive_transform_seed(tmp_path: Path) -> None:
    episodes = _tiny_episodes(n_episodes=2, n_steps=2)
    shuffle_variant = next(v for v in codec_variants() if v.stem == "B_shuffle")
    frame = build_encoded_frame(episodes, shuffle_variant)
    for _, row in frame.iterrows():
        ep = int(row["episode_seed"])
        t = int(row["timestep"])
        # Locate matching state
        states = next(s for seed, s in episodes if seed == ep)
        obs = observe(states[t])
        from obs_codecs.encode_b import encode_B

        expected = shuffle_b(
            encode_B(obs), derive_transform_seed(ep, t, "shuffle_b")
        )
        assert row["encoded_string"] == expected


def test_validation_stops_on_mismatch(tmp_path: Path) -> None:
    episodes = _tiny_episodes(n_episodes=3, n_steps=2)

    def bad_parse(_text: str):
        raise ValueError("forced failure")

    bad = CodecVariant("A_bad", lambda obs, _e, _t: encode_A(obs), bad_parse)
    with pytest.raises(EncodingValidationError) as exc_info:
        encode_split_variant(
            episodes,
            split_name="train",
            variant=bad,
            out_dir=tmp_path / "encoded",
            sample_size=10,
            validation_seed=0,
        )
    assert exc_info.value.report.validation_pass_rate == 0.0


def test_encode_dataset_writes_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    episodes = _tiny_episodes(n_episodes=3, n_steps=2)
    write_episodes_hdf5(raw / "train.h5", episodes, split_name="train")
    out = tmp_path / "encoded"
    reports = encode_dataset(
        raw,
        out,
        split_names=["train"],
        sample_size=20,
        validation_seed=7,
    )
    assert len(reports) == len(ALL_VARIANT_STEMS)
    manifest = json.loads((out / "encoding_report.json").read_text(encoding="utf-8"))
    assert manifest["storage_format"] == "parquet"
    assert len(manifest["reports"]) == len(ALL_VARIANT_STEMS)
    assert all(r["validation_pass_rate"] == 1.0 for r in manifest["reports"])


def test_encode_dataset_script_smoke(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    episodes = _tiny_episodes(n_episodes=2, n_steps=2)
    write_episodes_hdf5(raw / "train.h5", episodes, split_name="train")
    out = tmp_path / "encoded"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--raw-dir",
            str(raw),
            "--out-dir",
            str(out),
            "--splits",
            "train",
            "--validation-sample-size",
            "5",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (out / "train_A.parquet").is_file()
    assert (out / "train_B_shuffle.parquet").is_file()
    assert (out / "encoding_report.json").is_file()


def test_avg_tokens_per_string_matches_tokenize() -> None:
    texts = ["a = 1", "hello → world"]
    assert avg_tokens_per_string(texts) > 0
