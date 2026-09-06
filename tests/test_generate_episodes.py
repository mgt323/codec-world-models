"""Tests for data.generate_episodes (determinism, regime balance, no codecs)."""

from __future__ import annotations

import ast
import pickle
from collections import Counter
from pathlib import Path

import pytest

from data.generate_episodes import (
    ACTIVE_SPLIT_NAMES,
    DEFERRED_SPLIT_NAMES,
    REGIME_BALANCE_MIN_EPISODES,
    REGIME_BALANCE_TOLERANCE,
    episode_seed,
    generate_counterfactual_pair,
    generate_intervention_episode,
    generate_split,
)
from world.schema import CausalRegime, Intervention, InterventionTarget, State
from world.simulate import DEFAULT_DIFFICULTY_V0, simulate

_ROOT = Path(__file__).resolve().parents[1]
_GENERATE_EPISODES = _ROOT / "data" / "generate_episodes.py"


def _materialize(
    *,
    split_name: str = "train",
    n_episodes: int = 8,
    base_seed: int = 1000,
    n_steps: int = 5,
    intervention_fraction: float = 0.0,
) -> list[tuple[int, list[State]]]:
    return list(
        generate_split(
            split_name,
            n_episodes,
            DEFAULT_DIFFICULTY_V0,
            base_seed,
            n_steps,
            intervention_fraction=intervention_fraction,
        )
    )


def test_episode_seed_formula() -> None:
    assert episode_seed(10_000, 0) == 10_000
    assert episode_seed(10_000, 7) == 10_007


def test_generate_split_determinism_field_by_field() -> None:
    """Same args twice → identical episode list (State equality, not just seeds)."""
    kwargs = dict(
        split_name="train",
        n_episodes=12,
        base_seed=42_000,
        n_steps=8,
    )
    a = _materialize(**kwargs)
    b = _materialize(**kwargs)

    assert len(a) == len(b) == 12
    for (seed_a, states_a), (seed_b, states_b) in zip(a, b, strict=True):
        assert seed_a == seed_b
        assert len(states_a) == len(states_b)
        for sa, sb in zip(states_a, states_b, strict=True):
            assert sa == sb
            assert sa.t == sb.t
            assert sa.a == sb.a
            assert sa.b == sb.b
            assert sa.a_observed == sb.a_observed
            assert sa.b_observed == sb.b_observed
            assert sa.noise_bin == sb.noise_bin
            assert sa.n_samples == sb.n_samples
            assert sa.source == sb.source
            assert sa.latent == sb.latent
            assert sa.active_intervention == sb.active_intervention

    assert pickle.dumps(a) == pickle.dumps(b)


def test_episode_seeds_follow_base_seed_plus_index() -> None:
    base_seed = 500
    n_episodes = 5
    out = _materialize(n_episodes=n_episodes, base_seed=base_seed, n_steps=3)
    for i, (seed, _) in enumerate(out):
        assert seed == base_seed + i


def test_regime_distribution_sanity_at_1000_train() -> None:
    """At n=1000 under train difficulty, regimes stay within ±10% of uniform."""
    n_episodes = REGIME_BALANCE_MIN_EPISODES
    assert n_episodes == 1000

    episodes = _materialize(
        split_name="train",
        n_episodes=n_episodes,
        base_seed=7,
        n_steps=4,
    )
    assert len(episodes) == n_episodes

    regimes = tuple(CausalRegime)
    expected = 1.0 / len(regimes)
    counts: Counter[CausalRegime] = Counter(
        states[0].latent.regime for _, states in episodes
    )

    for regime in regimes:
        share = counts[regime] / n_episodes
        assert abs(share - expected) <= REGIME_BALANCE_TOLERANCE, (
            f"{regime.value} share={share:.3f} outside "
            f"[{expected - REGIME_BALANCE_TOLERANCE:.3f}, "
            f"{expected + REGIME_BALANCE_TOLERANCE:.3f}]; counts={dict(counts)}"
        )


def test_generate_episodes_does_not_import_obs_codecs() -> None:
    """Static guard: data/generate_episodes.py must not import obs_codecs."""
    tree = ast.parse(
        _GENERATE_EPISODES.read_text(encoding="utf-8"),
        filename=str(_GENERATE_EPISODES),
    )
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "obs_codecs" or alias.name.startswith("obs_codecs."):
                    hits.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "obs_codecs" or mod.startswith("obs_codecs."):
                hits.append(f"line {node.lineno}: from {mod} import ...")

    assert not hits, (
        "generate_episodes.py must not import obs_codecs "
        "(raw State generation only):\n  " + "\n  ".join(hits)
    )


@pytest.mark.parametrize("split_name", sorted(DEFERRED_SPLIT_NAMES))
def test_deferred_splits_rejected(split_name: str) -> None:
    with pytest.raises(ValueError, match="deferred"):
        _materialize(split_name=split_name, n_episodes=1, n_steps=2)


@pytest.mark.parametrize("split_name", sorted(ACTIVE_SPLIT_NAMES))
def test_active_splits_accepted(split_name: str) -> None:
    out = _materialize(split_name=split_name, n_episodes=2, n_steps=2)
    assert len(out) == 2


def test_intervention_fraction_nonzero_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Prompt B2"):
        _materialize(n_episodes=1, n_steps=2, intervention_fraction=0.1)


def test_generate_intervention_episode_matches_simulate() -> None:
    seed, n_steps = 99, 6
    intervention = Intervention(
        target=InterventionTarget.A, value=0.55, timestep=2
    )
    assert generate_intervention_episode(
        seed, n_steps, DEFAULT_DIFFICULTY_V0, intervention=None
    ) == simulate(seed, n_steps, difficulty=DEFAULT_DIFFICULTY_V0, interventions=None)
    assert generate_intervention_episode(
        seed, n_steps, DEFAULT_DIFFICULTY_V0, intervention=intervention
    ) == simulate(
        seed,
        n_steps,
        difficulty=DEFAULT_DIFFICULTY_V0,
        interventions=(intervention,),
    )


@pytest.mark.parametrize(
    "regime",
    [
        CausalRegime.A_CAUSES_B,
        CausalRegime.B_CAUSES_A,
        CausalRegime.COMMON_CAUSE,
        CausalRegime.SPURIOUS,
    ],
)
def test_counterfactual_pair_prefix_and_do_asymmetry(regime: CausalRegime) -> None:
    """Twins share noise; diverge only via do() per existing asymmetry suite.

    Reuses hardcoded seeds / settings from
    ``tests/test_simulate_intervention_asymmetry.py`` rather than re-deriving
    regime→seed mappings or do-calculus expectations here.
    """
    # Import fixture constants from the authoritative asymmetry module.
    from tests.test_simulate_intervention_asymmetry import (
        _DIFF,
        _DO_VALUE,
        _N_STEPS,
        _SEED_BY_REGIME,
        _T_DO,
        _obs_do_pair,
    )

    seed = _SEED_BY_REGIME[regime]
    do_a = Intervention(
        target=InterventionTarget.A, value=_DO_VALUE, timestep=_T_DO
    )
    do_b = Intervention(
        target=InterventionTarget.B, value=_DO_VALUE, timestep=_T_DO
    )

    obs_a, int_a = generate_counterfactual_pair(seed, _N_STEPS, _DIFF, do_a)
    obs_b, int_b = generate_counterfactual_pair(seed, _N_STEPS, _DIFF, do_b)

    # Same seed → observational legs identical; match direct simulate pair API.
    ref_obs, ref_do_a, ref_do_b = _obs_do_pair(seed)
    assert obs_a == obs_b == ref_obs
    assert int_a == ref_do_a
    assert int_b == ref_do_b
    assert obs_a[0].latent.regime is regime

    # Identical up to (not including) the intervention timestep.
    for t in range(_T_DO):
        assert obs_a[t] == int_a[t] == int_b[t]

    # Divergence at t_do follows the do-calculus asymmetry already locked in
    # test_simulate_intervention_asymmetry (positive / negative controls).
    o, a, b = obs_a[_T_DO], int_a[_T_DO], int_b[_T_DO]
    assert a.a == pytest.approx(_DO_VALUE)
    assert b.b == pytest.approx(_DO_VALUE)

    if regime is CausalRegime.A_CAUSES_B:
        assert a.b != pytest.approx(o.b)
        assert b.a == pytest.approx(o.a)
    elif regime is CausalRegime.B_CAUSES_A:
        assert b.a != pytest.approx(o.a)
        assert a.b == pytest.approx(o.b)
    else:
        # common_cause / spurious: do(A) leaves B; do(B) leaves A.
        assert a.b == pytest.approx(o.b)
        assert b.a == pytest.approx(o.a)
