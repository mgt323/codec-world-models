"""Tests for train-only vocab construction and OOV reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from obs_codecs.diagnostics import TOKENIZATION_SCHEME_VERSION, tokenize
from obs_codecs.vocab import (
    B_FAMILY_VARIANTS,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    UNK_TOKEN,
    artifact_from_token_to_id,
    build_and_save_vocabs,
    build_vocab,
    compute_oov_stats,
    encode_string_ids,
    is_number_token,
    load_vocab,
    lookup_id,
    save_vocab,
)


def _write_parquet(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "episode_seed": list(range(len(texts))),
            "timestep": [0] * len(texts),
            "encoded_string": texts,
        }
    ).to_parquet(path)


def test_build_vocab_reserves_specials_and_sorts_content(tmp_path: Path) -> None:
    path = tmp_path / "train_A.parquet"
    _write_parquet(path, ["Entity A: value=0.41", "Entity B: value=0.38"])
    vocab = build_vocab([path])
    assert [vocab[t] for t in SPECIAL_TOKENS] == [PAD_ID, BOS_ID, EOS_ID, UNK_ID]
    content = sorted(t for t in vocab if t not in SPECIAL_TOKENS)
    assert [vocab[t] for t in content] == list(range(len(SPECIAL_TOKENS), len(vocab)))
    assert "0" in vocab and "4" in vocab and "1" in vocab and "." in vocab
    assert "0.41" not in vocab
    assert "Entity" in vocab


def test_lookup_and_encode_use_unk(tmp_path: Path) -> None:
    path = tmp_path / "train_A.parquet"
    _write_parquet(path, ["value=0.41"])
    vocab = build_vocab([path])
    assert lookup_id("0", vocab) == vocab["0"]
    assert lookup_id("zzz_missing", vocab) == UNK_ID
    ids = encode_string_ids("value=9.99", vocab)
    # digits 9 may be OOV if only 0,4,1 seen — map via unk
    assert vocab["value"] in ids
    assert vocab["="] in ids


def test_b_family_shares_vocab_including_pipe_and_arrow(tmp_path: Path) -> None:
    encoded = tmp_path / "encoded"
    out = tmp_path / "vocabs"
    _write_parquet(
        encoded / "train_A.parquet",
        ["Entity A: value=0.1, status=present."],
    )
    _write_parquet(
        encoded / "train_B.parquet",
        ["(observe A=0.41) \u2192 (miss B)"],
    )
    _write_parquet(
        encoded / "train_B_bag.parquet",
        ["(observe A=0.41) | (miss B)"],
    )
    _write_parquet(
        encoded / "train_B_shuffle.parquet",
        ["(miss B) \u2192 (observe A=0.41)"],
    )
    _write_parquet(
        encoded / "train_B_reverse.parquet",
        ["(miss B) \u2192 (observe A=0.41)"],
    )
    _write_parquet(encoded / "train_C.parquet", ["obs: A=0.41"])
    _write_parquet(encoded / "train_D.parquet", ["A.x=0.41; A.obs=1"])
    for split in ("val", "ood_partial_obs"):
        for stem in ("A", "B", "C", "D", "B_shuffle", "B_reverse", "B_bag"):
            _write_parquet(
                encoded / f"{split}_{stem}.parquet",
                ["(observe A=0.99)"],
            )

    report = build_and_save_vocabs(encoded, out, share_b_family=True)
    assert report.shared_b_family
    b = load_vocab(out / "vocab_B.json")
    assert "\u2192" in b.token_to_id
    assert "|" in b.token_to_id
    for stem in B_FAMILY_VARIANTS:
        other = load_vocab(out / f"vocab_{stem}.json")
        assert other.token_to_id == b.token_to_id
        assert other.metadata["shared_from_group"] == "B"
        assert other.metadata["tokenizer_scheme_version"] == TOKENIZATION_SCHEME_VERSION
    assert (out / "vocab_report.json").is_file()
    assert report.vocab_sizes["B"] == b.vocab_size


def test_oov_stats_count_unseen_idents(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    _write_parquet(train, ["value=0.10"])
    _write_parquet(val, ["value=0.10", "missing_ident=0.99"])
    vocab = build_vocab([train])
    stats = compute_oov_stats(val, vocab, split="val", variant="A")
    assert "missing_ident" not in vocab
    assert stats.n_oov_tokens >= 1
    assert stats.n_oov_types >= 1


def test_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    _write_parquet(path, ["A=0.5"])
    vocab = build_vocab([path])
    artifact = artifact_from_token_to_id(
        vocab,
        metadata={
            "source_split": "train",
            "tokenizer_scheme_version": TOKENIZATION_SCHEME_VERSION,
        },
    )
    out = save_vocab(artifact, tmp_path / "vocab_A.json")
    loaded = load_vocab(out)
    assert loaded.token_to_id == artifact.token_to_id
    assert loaded.id_to_token == artifact.id_to_token
    assert loaded.metadata["tokenizer_scheme_version"] == "regex_v4"
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert UNK_TOKEN in raw["token_to_id"]


def test_is_number_token_matches_regex_v4_pieces() -> None:
    assert is_number_token("0")
    assert is_number_token(".")
    assert is_number_token("-")
    assert not is_number_token("0.41")
    assert not is_number_token("A")
    assert tokenize("A=0.4123456789") == [
        "A",
        "=",
        "0",
        ".",
        "4",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]


def test_build_vocab_cli_smoke(tmp_path: Path) -> None:
    import subprocess
    import sys

    encoded = tmp_path / "encoded"
    out = tmp_path / "vocabs"
    for stem, text in (
        ("A", "Entity A: value=0.1."),
        ("B", "(observe A=0.1) \u2192 (miss B)"),
        ("B_bag", "(observe A=0.1) | (miss B)"),
        ("B_shuffle", "(miss B) \u2192 (observe A=0.1)"),
        ("B_reverse", "(miss B) \u2192 (observe A=0.1)"),
        ("C", "obs: A=0.1"),
        ("D", "A.x=0.1; A.obs=1"),
    ):
        _write_parquet(encoded / f"train_{stem}.parquet", [text])
        _write_parquet(encoded / f"val_{stem}.parquet", [text])
        _write_parquet(encoded / f"ood_partial_obs_{stem}.parquet", [text])

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_vocab.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--encoded-dir",
            str(encoded),
            "--out-dir",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Wrote vocabs" in completed.stdout
    assert (out / "vocab_A.json").is_file()
    assert (out / "vocab_B_bag.json").is_file()
