"""Formalize explore scripts test2/test3: do-calculus asymmetry with fixed seeds.

Hardcoded seeds were found under Difficulty(noise_scale=0.15, partial_obs_rate=0.0)
and n_steps=8 — not scanned at test time (fast, non-flaky).
"""

from __future__ import annotations

import pytest

from world.schema import (
    CausalRegime,
    Difficulty,
    Intervention,
    InterventionTarget,
)
from world.simulate import simulate

# Exploration settings (match test2.py / test3.py).
_DIFF = Difficulty(noise_scale=0.15, partial_obs_rate=0.0)
_N_STEPS = 8
_T_DO = 3
_DO_VALUE = 0.9

# First seed that yields each regime under (_DIFF, n_steps=8).
_SEED_BY_REGIME: dict[CausalRegime, int] = {
    CausalRegime.COMMON_CAUSE: 11,
    CausalRegime.A_CAUSES_B: 1,
    CausalRegime.B_CAUSES_A: 4,
    CausalRegime.SPURIOUS: 0,
}


def _obs_do_pair(seed: int) -> tuple:
    obs = simulate(seed, _N_STEPS, difficulty=_DIFF, interventions=None)
    do_a = simulate(
        seed,
        _N_STEPS,
        difficulty=_DIFF,
        interventions=(
            Intervention(
                target=InterventionTarget.A, value=_DO_VALUE, timestep=_T_DO
            ),
        ),
    )
    do_b = simulate(
        seed,
        _N_STEPS,
        difficulty=_DIFF,
        interventions=(
            Intervention(
                target=InterventionTarget.B, value=_DO_VALUE, timestep=_T_DO
            ),
        ),
    )
    return obs, do_a, do_b


@pytest.mark.parametrize(
    "regime",
    [CausalRegime.A_CAUSES_B, CausalRegime.B_CAUSES_A],
)
def test_causal_parent_intervention_changes_child_not_vice_versa(
    regime: CausalRegime,
) -> None:
    """Positive control: intervene on parent → child moves; on child → parent fixed."""
    seed = _SEED_BY_REGIME[regime]
    obs, do_a, do_b = _obs_do_pair(seed)
    assert obs[0].latent.regime is regime

    # Prefix identity (same seed / RNG) before intervention timestep.
    for t in range(_T_DO):
        assert obs[t] == do_a[t] == do_b[t]

    o, a, b = obs[_T_DO], do_a[_T_DO], do_b[_T_DO]
    assert a.a == pytest.approx(_DO_VALUE)
    assert b.b == pytest.approx(_DO_VALUE)

    if regime is CausalRegime.A_CAUSES_B:
        # Parent A → child B: do(A) changes B; do(B) does not change A.
        assert a.b != pytest.approx(o.b)
        assert b.a == pytest.approx(o.a)
    else:
        # Parent B → child A: do(B) changes A; do(A) does not change B.
        assert b.a != pytest.approx(o.a)
        assert a.b == pytest.approx(o.b)


@pytest.mark.parametrize(
    "regime",
    [CausalRegime.COMMON_CAUSE, CausalRegime.SPURIOUS],
)
def test_common_cause_and_spurious_do_a_never_changes_b(
    regime: CausalRegime,
) -> None:
    """Negative control (test2.py): do(A) must not move B; do(B) must not move A."""
    seed = _SEED_BY_REGIME[regime]
    obs, do_a, do_b = _obs_do_pair(seed)
    assert obs[0].latent.regime is regime

    for t in range(_T_DO):
        assert obs[t] == do_a[t] == do_b[t]

    o, a, b = obs[_T_DO], do_a[_T_DO], do_b[_T_DO]
    assert a.a == pytest.approx(_DO_VALUE)
    assert b.b == pytest.approx(_DO_VALUE)
    assert a.b == pytest.approx(o.b)
    assert b.a == pytest.approx(o.a)


def test_hardcoded_seeds_still_map_to_declared_regimes() -> None:
    """Guard: if simulate RNG/regime sampling changes, fail loudly here."""
    for regime, seed in _SEED_BY_REGIME.items():
        states = simulate(seed, _N_STEPS, difficulty=_DIFF, interventions=None)
        assert states[0].latent.regime is regime, (
            f"seed={seed} no longer yields {regime.value}; re-discover and update "
            f"_SEED_BY_REGIME"
        )
