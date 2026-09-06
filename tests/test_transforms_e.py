"""Tests for Codec B structure-destroyed E-transforms (I11)."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from obs_codecs.diagnostics import compute_codec_diagnostics, tokenize
from obs_codecs.encode_b import (
    ARROW_SEP,
    BAG_SEP,
    encode_B,
    observation_from_B,
    observation_from_B_unordered,
    split_b_events,
)
from obs_codecs.transforms_e import (
    bag_b,
    derive_transform_seed,
    reverse_b,
    shuffle_b,
)
from world.parity_fixtures import sample_diverse_parity_states
from world.schema import facts_from_observation, observe

# Seeds whose Fisher–Yates permutation of 3 indices is not identity
# (verified against random.Random; seed 5/17 are identity for n=3).
_SHUFFLE_SEEDS = (0, 1, 7, 19)

# Locked blake2b golden for cross-process / schema stability checks.
_GOLDEN_EPISODE = 42
_GOLDEN_TIMESTEP = 7
_GOLDEN_SHUFFLE = 11_585_691_077_974_233_195  # derive_transform_seed(42, 7, "shuffle_b")

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sample_observations():
    return [observe(s) for s in sample_diverse_parity_states(target_n=500)]


def test_derive_transform_seed_deterministic_and_matches_blake2b_encoding() -> None:
    """Same inputs → identical seed; encoding matches the locked blake2b recipe."""
    a = derive_transform_seed(_GOLDEN_EPISODE, _GOLDEN_TIMESTEP, "shuffle_b")
    b = derive_transform_seed(_GOLDEN_EPISODE, _GOLDEN_TIMESTEP, "shuffle_b")
    assert a == b == _GOLDEN_SHUFFLE
    assert 0 <= a < (1 << 64)


def test_derive_transform_seed_identical_in_fresh_process() -> None:
    """Fresh interpreter must reproduce the same uint64 (not Python hash())."""
    code = (
        "from obs_codecs.transforms_e import derive_transform_seed; "
        f"print(derive_transform_seed({_GOLDEN_EPISODE}, {_GOLDEN_TIMESTEP}, 'shuffle_b'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert int(result.stdout.strip()) == _GOLDEN_SHUFFLE


def test_derive_transform_seed_variant_independence() -> None:
    """shuffle_b and bag_b must not share a remapping at the same (ep, t)."""
    ep, t = 1000, 3
    shuffle_seed = derive_transform_seed(ep, t, "shuffle_b")
    bag_seed = derive_transform_seed(ep, t, "bag_b")
    assert shuffle_seed != bag_seed


def test_derive_transform_seed_varies_with_timestep() -> None:
    """Timestep must enter the hash (guards accidental drop from payload)."""
    ep = 1000
    s0 = derive_transform_seed(ep, 0, "shuffle_b")
    s1 = derive_transform_seed(ep, 1, "shuffle_b")
    s2 = derive_transform_seed(ep, 2, "shuffle_b")
    assert len({s0, s1, s2}) == 3


def test_shuffle_b_roundtrip_info_equality_via_unordered_parser(sample_observations) -> None:
    """shuffle_b destroys order only — unordered parse recovers the same Observation."""
    for obs in sample_observations:
        natural = encode_B(obs)
        expected_facts = facts_from_observation(obs)
        for seed in _SHUFFLE_SEEDS:
            transformed = shuffle_b(natural, seed)
            recovered = observation_from_B_unordered(transformed)
            assert recovered == obs
            assert facts_from_observation(recovered) == expected_facts
            # Position parser still works on arrow-joined shuffle (roles may be
            # wrong by index) — do NOT require observation_from_B here.


def test_shuffle_b_roundtrip_under_derive_transform_seed(sample_observations) -> None:
    """Info-equality still holds when seeds come from derive_transform_seed."""
    episode_seed = 9_001
    for t, obs in enumerate(sample_observations[:100]):
        natural = encode_B(obs)
        seed = derive_transform_seed(episode_seed, t, "shuffle_b")
        transformed = shuffle_b(natural, seed)
        recovered = observation_from_B_unordered(transformed)
        assert recovered == obs
        assert facts_from_observation(recovered) == facts_from_observation(obs)


def test_reverse_b_roundtrip_info_equality_via_unordered_parser(sample_observations) -> None:
    """reverse_b destroys order only — unordered parse recovers the same Observation."""
    for obs in sample_observations:
        natural = encode_B(obs)
        transformed = reverse_b(natural)
        recovered = observation_from_B_unordered(transformed)
        assert recovered == obs
        assert facts_from_observation(recovered) == facts_from_observation(obs)


def test_bag_b_roundtrip_info_equality_via_unordered_parser_only(
    sample_observations,
) -> None:
    """bag_b drops arrows; ONLY observation_from_B_unordered recovers facts.

    Position-based observation_from_B is expected to FAIL on bag output —
    that is not a regression; bag removes the sequential marker the
    index parser depends on.
    """
    for obs in sample_observations:
        natural = encode_B(obs)
        for seed in _SHUFFLE_SEEDS:
            bagged = bag_b(natural, seed)
            assert ARROW_SEP not in bagged
            assert BAG_SEP in bagged

            recovered = observation_from_B_unordered(bagged)
            assert recovered == obs
            assert facts_from_observation(recovered) == facts_from_observation(obs)

            with pytest.raises(ValueError):
                observation_from_B(bagged)


def test_bag_b_roundtrip_under_derive_transform_seed(sample_observations) -> None:
    """Info-equality under derive_transform_seed for bag_b (independent variant)."""
    episode_seed = 9_001
    for t, obs in enumerate(sample_observations[:100]):
        natural = encode_B(obs)
        seed = derive_transform_seed(episode_seed, t, "bag_b")
        bagged = bag_b(natural, seed)
        recovered = observation_from_B_unordered(bagged)
        assert recovered == obs
        assert facts_from_observation(recovered) == facts_from_observation(obs)
        with pytest.raises(ValueError):
            observation_from_B(bagged)


def test_shuffle_b_and_bag_b_are_deterministic(sample_observations) -> None:
    obs = sample_observations[0]
    text = encode_B(obs)
    for seed in _SHUFFLE_SEEDS:
        assert shuffle_b(text, seed) == shuffle_b(text, seed)
        assert bag_b(text, seed) == bag_b(text, seed)


def test_reverse_b_is_deterministic(sample_observations) -> None:
    text = encode_B(sample_observations[0])
    assert reverse_b(text) == reverse_b(text)


def test_shuffle_b_actually_changes_order(sample_observations) -> None:
    """Guard against no-op shuffle for n>1 events (B always emits 3)."""
    for obs in sample_observations[:50]:
        text = encode_B(obs)
        assert len(split_b_events(text)) > 1
        for seed in _SHUFFLE_SEEDS:
            assert shuffle_b(text, seed) != text


def test_reverse_b_actually_changes_order(sample_observations) -> None:
    for obs in sample_observations[:50]:
        text = encode_B(obs)
        assert len(split_b_events(text)) > 1
        assert reverse_b(text) != text


def test_shuffle_and_reverse_preserve_token_multiset(sample_observations) -> None:
    """I11: token inventory unchanged; only order differs."""
    for obs in sample_observations[:100]:
        natural = encode_B(obs)
        base = Counter(tokenize(natural))
        for seed in _SHUFFLE_SEEDS:
            assert Counter(tokenize(shuffle_b(natural, seed))) == base
        assert Counter(tokenize(reverse_b(natural))) == base


def test_bag_b_preserves_content_tokens_accounting_for_separator(
    sample_observations,
) -> None:
    """bag_b swaps ARROW tokens for BAG_SEP pipes; event content tokens match."""
    for obs in sample_observations[:100]:
        natural = encode_B(obs)
        bagged = bag_b(natural, seed=0)
        base = Counter(tokenize(natural))
        got = Counter(tokenize(bagged))

        n_arrows = base["→"]
        assert n_arrows >= 1
        # Arrow glyphs removed; one "|" per former arrow join (n_events - 1).
        expected = base.copy()
        del expected["→"]
        expected["|"] = expected.get("|", 0) + n_arrows
        assert got == expected


def test_transforms_e_no_global_random_side_effects(sample_observations) -> None:
    """Transforms must not touch the module-global random state."""
    import random as py_random

    text = encode_B(sample_observations[0])
    py_random.seed(12345)
    before = [py_random.random() for _ in range(5)]
    py_random.seed(12345)
    _ = shuffle_b(text, seed=99)
    _ = bag_b(text, seed=99)
    _ = reverse_b(text)
    after = [py_random.random() for _ in range(5)]
    assert after == before


def test_b_vs_e_diagnostics_vocab_near_identical(sample_observations) -> None:
    """Surface stats: shuffle/reverse match B vocab; bag swaps → for |."""
    seed = 0

    def enc_shuffle(obs):
        return shuffle_b(encode_B(obs), seed)

    def enc_reverse(obs):
        return reverse_b(encode_B(obs))

    def enc_bag(obs):
        return bag_b(encode_B(obs), seed)

    report_b = compute_codec_diagnostics(sample_observations, encode=encode_B)
    report_s = compute_codec_diagnostics(sample_observations, encode=enc_shuffle)
    report_r = compute_codec_diagnostics(sample_observations, encode=enc_reverse)
    report_bag = compute_codec_diagnostics(sample_observations, encode=enc_bag)

    assert report_s.vocab_size == report_b.vocab_size
    assert report_r.vocab_size == report_b.vocab_size
    assert report_s.avg_tokens_per_observation == report_b.avg_tokens_per_observation
    assert report_r.avg_tokens_per_observation == report_b.avg_tokens_per_observation

    # bag: same token count (→ replaced 1:1 by |); vocab size equal (swap).
    assert report_bag.avg_tokens_per_observation == report_b.avg_tokens_per_observation
    assert report_bag.vocab_size == report_b.vocab_size
    assert "→" in report_b.token_histogram
    assert "→" not in report_bag.token_histogram
    assert "|" in report_bag.token_histogram
    assert "|" not in report_b.token_histogram
