"""Tier B2 additive diagnostics: probe baselines and eval diagnostics.

Locked metrics keep their definitions; these tests pin the new keys and prove
they decompose (never replace) the locked numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.generate_episodes import generate_counterfactual_pair
from eval.diagnostics import (
    DIAGNOSTICS_VERSION,
    WINSOR_UPPER_QUANTILE,
    counterfactual_step_diagnostics,
    intervention_baselines,
    intervention_diagnostics,
    oracle_scored_value,
    squared_error_summary,
)
from eval.score import (
    intervention_accuracy,
    make_intervention_episodes,
)
from model.architecture import build_model
from model.init import initialize_model
from obs_codecs.diagnostics import tokenize
from obs_codecs.encode_a import encode_A
from obs_codecs.encode_b import encode_B
from obs_codecs.vocab import SPECIAL_TOKENS, VocabArtifact, save_vocab
from probes.baselines import last_observed_b, oracle_next_b_mean, prefix_mean_observed_b
from probes.linear import heldout_ridge_r2, split_by_group
from scripts.run_probes import (
    NO_OBSERVED_B_FALLBACK,
    _fixture_baselines,
    _flatten_states,
    _probe_codec,
    _untrained_model,
)
from world.schema import (
    CausalRegime,
    Intervention,
    InterventionTarget,
    State,
    observe,
)
from world.simulate import DEFAULT_DIFFICULTY_V0, simulate

_REGIMES = tuple(CausalRegime)


def _episodes(n: int, *, n_steps: int = 30, base: int = 100) -> list[tuple[int, list[State]]]:
    return [
        (base + i, simulate(base + i, n_steps, difficulty=DEFAULT_DIFFICULTY_V0))
        for i in range(n)
    ]


def _vocab_for(episodes, encode) -> VocabArtifact:
    content = sorted(
        {tok for _seed, states in episodes for s in states for tok in tokenize(encode(observe(s)))}
    )
    id_to_token = list(SPECIAL_TOKENS) + content
    return VocabArtifact(
        token_to_id={tok: idx for idx, tok in enumerate(id_to_token)},
        id_to_token=id_to_token,
        metadata={},
    )


# --------------------------------------------------------------------------
# probes/baselines.py
# --------------------------------------------------------------------------


def test_oracle_next_b_mean_matches_simulated_conditional_mean() -> None:
    # Per-episode mean of the true b over 30 steps estimates E[b | regime, hidden_c].
    residual: dict[CausalRegime, list[float]] = {regime: [] for regime in _REGIMES}
    for _seed, states in _episodes(400, base=900):
        latent = states[0].latent
        episode_mean = float(np.mean([s.b for s in states]))
        residual[latent.regime].append(
            episode_mean - oracle_next_b_mean(latent.regime, latent.hidden_c)
        )
    for regime, values in residual.items():
        assert len(values) > 30
        assert abs(float(np.mean(values))) < 0.02, regime


def test_last_observed_and_prefix_mean_respect_sensor_mask() -> None:
    episode = next(
        states
        for _seed, states in _episodes(200)
        if any(not s.b_observed for s in states[:10]) and any(s.b_observed for s in states[:10])
    )
    for end in range(1, 11):
        prefix = episode[:end]
        observed = [s.b for s in prefix if s.b_observed]
        if observed:
            assert last_observed_b(prefix, fallback=-9.0) == observed[-1]
            assert prefix_mean_observed_b(prefix, fallback=-9.0) == pytest.approx(
                float(np.mean(observed))
            )
        else:
            assert last_observed_b(prefix, fallback=-9.0) == -9.0
            assert prefix_mean_observed_b(prefix, fallback=-9.0) == -9.0


def test_fixture_baselines_order_and_hold_out_rows() -> None:
    samples = _flatten_states(_episodes(120), n_states=120 * 30)
    train_mask, test_mask = split_by_group([s.episode_index for s in samples])
    baselines = _fixture_baselines(samples, test_mask)
    has_next = np.asarray([s.next_b is not None for s in samples])
    assert baselines["n_test_next_states"] == int((test_mask & has_next).sum())
    assert baselines["no_observed_b_fallback"] == NO_OBSERVED_B_FALLBACK
    persistence = baselines["persistence_next_b_r2"]
    running = baselines["running_mean_next_b_r2"]
    oracle = baselines["oracle_next_b_r2"]
    assert persistence is not None and running is not None and oracle is not None
    # b_{t+1} is conditionally i.i.d. given (regime, hidden_c): integrating the
    # prefix beats persistence, and the latent conditional mean is the ceiling.
    assert persistence < running < oracle <= 1.0


def test_fixture_baselines_are_none_without_enough_test_rows() -> None:
    samples = _flatten_states(_episodes(2, n_steps=1), n_states=2)
    _train, test_mask = split_by_group([s.episode_index for s in samples])
    baselines = _fixture_baselines(samples, test_mask)
    assert baselines["persistence_next_b_r2"] is None
    assert baselines["oracle_next_b_r2"] is None


def test_common_cause_probe_isolates_the_regime_that_emits_hidden_c() -> None:
    samples = _flatten_states(_episodes(80), n_states=80 * 30)
    train_mask, test_mask = split_by_group([s.episode_index for s in samples])
    rng = np.random.default_rng(0)
    hidden_c = np.asarray([s.hidden_c for s in samples])
    common = np.asarray([s.regime_index == _REGIMES.index(CausalRegime.COMMON_CAUSE) for s in samples])
    features = rng.normal(size=(len(samples), 12))
    # hidden_c is readable only where it drives the observables.
    features[:, 0] = np.where(common, hidden_c, rng.uniform(size=len(samples)))
    report = _probe_codec(features, samples, train_mask, test_mask)
    assert report["hidden_c_r2_common_cause"] is not None
    assert report["hidden_c_r2_common_cause"] > 0.9
    assert report["hidden_c_r2_common_cause"] > report["hidden_c_r2"] + 0.5
    assert report["n_common_cause_train_states"] == int((train_mask & common).sum())
    assert report["n_common_cause_test_states"] == int((test_mask & common).sum())


def test_locked_probe_keys_do_not_depend_on_the_additive_subset() -> None:
    samples = _flatten_states(_episodes(40), n_states=40 * 30)
    train_mask, test_mask = split_by_group([s.episode_index for s in samples])
    features = np.random.default_rng(1).normal(size=(len(samples), 8))
    report = _probe_codec(features, samples, train_mask, test_mask)
    hidden_c = np.asarray([s.hidden_c for s in samples])
    assert report["hidden_c_r2"] == heldout_ridge_r2(
        features[train_mask], hidden_c[train_mask], features[test_mask], hidden_c[test_mask]
    )
    assert {"hidden_c_r2", "regime_accuracy", "next_b_r2"} <= set(report)


def test_common_cause_probe_is_none_when_subset_too_small() -> None:
    episodes = [
        (seed, states)
        for seed, states in _episodes(60, n_steps=3)
        if states[0].latent.regime is not CausalRegime.COMMON_CAUSE
    ][:8]
    samples = _flatten_states(episodes, n_states=len(episodes) * 3)
    train_mask, test_mask = split_by_group([s.episode_index for s in samples])
    features = np.random.default_rng(2).normal(size=(len(samples), 4))
    report = _probe_codec(features, samples, train_mask, test_mask)
    assert report["hidden_c_r2_common_cause"] is None
    assert report["n_common_cause_train_states"] == 0


def test_untrained_model_is_the_seeded_frozen_init(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_init_seed": 1_000_003}), encoding="utf-8"
    )
    first = _untrained_model(tmp_path, "B", torch.device("cpu"))
    second = _untrained_model(tmp_path, "B", torch.device("cpu"))
    other_codec = _untrained_model(tmp_path, "A", torch.device("cpu"))
    reference = initialize_model(build_model(), model_init_seed=1_000_003, codec_name="B")
    for name, tensor in reference.state_dict().items():
        assert torch.equal(first.state_dict()[name], tensor)
        assert torch.equal(second.state_dict()[name], tensor)
    assert not torch.equal(first.state_dict()["tok_emb.weight"], other_codec.state_dict()["tok_emb.weight"])
    assert not first.training
    assert all(not p.requires_grad for p in first.parameters())


def test_run_probes_writes_additive_blocks_and_keeps_locked_keys(
    tmp_path: Path,
) -> None:
    from scripts.run_probes import main as probes_main

    n_states = 60
    fixture = _episodes(2, base=7_000_000)
    vocab_dir = tmp_path / "vocabs"
    save_vocab(_vocab_for(fixture, encode_A), vocab_dir / "vocab_A.json")
    run_dir = tmp_path / "runs" / "compute_matched" / "A" / "rep0"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"codec": "A", "model_init_seed": 1_000_000}), encoding="utf-8"
    )
    model = initialize_model(build_model(), model_init_seed=1_000_000, codec_name="A")
    torch.save(
        {"model_state_dict": model.state_dict(), "scoring": True, "step": 1},
        run_dir / "checkpoints" / "final.pt",
    )

    def run(out_dir: Path, *extra: str) -> dict:
        rc = probes_main(
            [
                "--runs-root", str(tmp_path / "runs"),
                "--raw-dir", str(tmp_path / "raw"),
                "--vocab-dir", str(vocab_dir),
                "--out-dir", str(out_dir),
                "--n-states", str(n_states),
                "--device", "cpu",
                *extra,
            ]
        )
        assert rc == 0
        return json.loads((out_dir / "probes.json").read_text(encoding="utf-8"))

    full = run(tmp_path / "with_untrained")
    lean = run(tmp_path / "without_untrained", "--skip-untrained")

    assert full["probe_protocol"]["additive_diagnostics"]["version"] == "probe_diagnostics_v1"
    assert set(full["fixture_baselines"]) >= {
        "persistence_next_b_r2",
        "running_mean_next_b_r2",
        "oracle_next_b_r2",
    }
    trained = full["probes"]["A"]
    untrained = full["untrained_probes"]["A"]
    for key in ("hidden_c_r2", "regime_accuracy", "next_b_r2", "hidden_c_r2_common_cause"):
        assert key in trained and key in untrained
    # The untrained pass must not perturb any trained score.
    assert lean["probes"] == full["probes"]
    assert lean["similarity"] == full["similarity"]
    assert "untrained_probes" not in lean
    assert "untrained_probes[codec]" not in lean["probe_protocol"]["additive_diagnostics"]["added"]


# --------------------------------------------------------------------------
# eval/diagnostics.py
# --------------------------------------------------------------------------


def test_squared_error_summary_is_robust_to_one_outlier() -> None:
    errors = [0.01] * 199 + [1.0e12]
    summary = squared_error_summary(errors)
    assert summary["next_state_n_errors"] == 200
    assert summary["next_state_mse_raw"] == pytest.approx(float(np.mean(errors)))
    assert summary["next_state_mse_median"] == pytest.approx(0.01)
    cap = float(np.percentile(np.asarray(errors), WINSOR_UPPER_QUANTILE * 100.0, method="linear"))
    assert summary["next_state_mse_winsorized"] == pytest.approx(
        float(np.minimum(np.asarray(errors), cap).mean())
    )
    assert summary["next_state_mse_winsorized"] == pytest.approx(0.01)
    assert summary["next_state_mse_raw"] > 1.0e9
    assert summary["next_state_squared_error_max"] == 1.0e12


def test_squared_error_summary_empty_and_invalid() -> None:
    empty = squared_error_summary([])
    assert empty["next_state_mse_median"] is None and empty["next_state_n_errors"] == 0
    with pytest.raises(ValueError):
        squared_error_summary([1.0], upper_quantile=0.0)


def _intervened_state(regime: CausalRegime, target: InterventionTarget) -> State:
    seed = next(
        s
        for s in range(2000)
        if simulate(s, 1, difficulty=DEFAULT_DIFFICULTY_V0)[0].latent.regime is regime
    )
    states = simulate(
        seed,
        4,
        difficulty=DEFAULT_DIFFICULTY_V0,
        interventions=[Intervention(target=target, value=0.8, timestep=2)],
    )
    return states[2]


@pytest.mark.parametrize(
    ("regime", "target", "expected"),
    [
        (CausalRegime.A_CAUSES_B, InterventionTarget.A, 0.8),  # child B follows do(A)
        (CausalRegime.A_CAUSES_B, InterventionTarget.B, 0.0),  # A is untouched noise
        (CausalRegime.B_CAUSES_A, InterventionTarget.B, 0.8),  # child A follows do(B)
        (CausalRegime.B_CAUSES_A, InterventionTarget.A, 0.0),
        (CausalRegime.SPURIOUS, InterventionTarget.A, 0.0),
        (CausalRegime.SPURIOUS, InterventionTarget.B, 0.0),
    ],
)
def test_oracle_scored_value_follows_the_causal_graph(
    regime: CausalRegime, target: InterventionTarget, expected: float
) -> None:
    assert oracle_scored_value(_intervened_state(regime, target)) == expected


def test_oracle_scored_value_common_cause_is_hidden_c_and_needs_intervention() -> None:
    state = _intervened_state(CausalRegime.COMMON_CAUSE, InterventionTarget.A)
    assert oracle_scored_value(state) == state.latent.hidden_c
    with pytest.raises(ValueError):
        oracle_scored_value(simulate(1, 1)[0])


def test_intervention_baselines_match_manual_counts() -> None:
    from world.schema import quantize_value

    episodes = make_intervention_episodes(32, base_seed=5_000_000, n_steps=30)
    result = intervention_baselines(episodes)
    steps = [s for _seed, states in episodes for s in states if s.active_intervention]
    assert result["baseline_n_steps"] == len(steps) == 32

    def truth(s: State) -> int:
        target = s.active_intervention.target  # type: ignore[union-attr]
        return quantize_value(s.b if target is InterventionTarget.A else s.a)

    zero = sum(quantize_value(0.0) == truth(s) for s in steps)
    oracle = sum(quantize_value(oracle_scored_value(s)) == truth(s) for s in steps)
    assert result["zero_baseline_acc"] == zero / 32
    assert result["oracle_acc"] == oracle / 32
    both = [s for s in steps if s.a_observed and s.b_observed]
    assert result["baseline_n_both_observed_steps"] == len(both)
    assert intervention_baselines([])["zero_baseline_acc"] is None


@pytest.mark.parametrize("decoder_mode", ["skeleton", "field_to_delimiter"])
def test_intervention_diagnostics_decompose_the_locked_accuracy(decoder_mode: str) -> None:
    episodes = make_intervention_episodes(8, base_seed=5_000_000, n_steps=6)
    vocab = _vocab_for(episodes, encode_B)
    model = initialize_model(build_model(), model_init_seed=1234, codec_name="B")
    device = torch.device("cpu")
    locked = intervention_accuracy(model, "B", episodes, vocab, device, decoder_mode=decoder_mode)
    extra = intervention_diagnostics(model, "B", episodes, vocab, device, decoder_mode=decoder_mode)
    n_steps = extra["intervention_n_steps"]
    assert n_steps == 8
    assert extra["intervention_hits"] / n_steps == locked["accuracy"]
    assert 1.0 - extra["intervention_n_parsed_steps"] / n_steps == pytest.approx(
        locked["parse_fail_rate"]
    )
    manual_both = sum(
        1
        for _seed, states in episodes
        for s in states
        if s.active_intervention and s.a_observed and s.b_observed
    )
    assert extra["intervention_n_both_observed_steps"] == manual_both
    if extra["intervention_n_parsed_steps"]:
        assert extra["intervention_accuracy_parsed"] == (
            extra["intervention_hits"] / extra["intervention_n_parsed_steps"]
        )
    else:
        assert extra["intervention_accuracy_parsed"] is None


def test_counterfactual_step_diagnostics_attribute_error_to_the_intervention_step() -> None:
    n_steps = 8
    pairs = []
    for index in range(6):
        seed = 6_000_000 + index
        intervention = Intervention(target=InterventionTarget.A, value=0.7, timestep=n_steps // 2)
        observational, intervened = generate_counterfactual_pair(
            seed, n_steps, DEFAULT_DIFFICULTY_V0, intervention
        )
        pairs.append(((seed, observational), (seed, intervened)))
    vocab = _vocab_for([p[0] for p in pairs] + [p[1] for p in pairs], encode_B)
    model = initialize_model(build_model(), model_init_seed=99, codec_name="B")
    result = counterfactual_step_diagnostics(model, "B", pairs, vocab, torch.device("cpu"))
    # Twins share every token before the intervention, so a causal model predicts
    # identical gaps there and the true gap is exactly zero.
    if result["cf_n_steps_before_intervention"]:
        assert result["cf_step_mse_before_intervention"] == pytest.approx(0.0, abs=1e-9)
    assert 0 <= result["cf_n_steps_at_intervention"] <= len(pairs)
    total = (
        result["cf_n_steps_before_intervention"]
        + result["cf_n_steps_at_intervention"]
        + result["cf_n_steps_after_intervention"]
    )
    assert total <= len(pairs) * n_steps
    if result["cf_n_steps_at_intervention"] == 0:
        assert result["cf_step_mse_at_intervention"] is None
    else:
        assert result["cf_step_mse_at_intervention"] >= 0.0


def test_counterfactual_step_buckets_with_controlled_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eval.diagnostics as diagnostics_module

    n_steps = 8
    intervention_t = n_steps // 2
    pairs = []
    for index in range(4):
        seed = 6_100_000 + index
        intervention = Intervention(
            target=InterventionTarget.A, value=0.7, timestep=intervention_t
        )
        observational, intervened = generate_counterfactual_pair(
            seed, n_steps, DEFAULT_DIFFICULTY_V0, intervention
        )
        pairs.append(((seed, observational), (seed, intervened)))

    def fake_predicted(model, codec, episode, vocab, device, *, decoder_mode):
        # Observational twin predicts 0.5 everywhere; the intervened twin predicts
        # 0.9 at the intervention step and 0.6 after it.
        is_intervened = any(s.active_intervention is not None for s in episode[1])
        values = {}
        for timestep in range(n_steps):
            if not is_intervened:
                values[timestep] = 0.5
            elif timestep == intervention_t:
                values[timestep] = 0.9
            elif timestep > intervention_t:
                values[timestep] = 0.6
            else:
                values[timestep] = 0.5
        return values

    monkeypatch.setattr(diagnostics_module, "_predicted_b_by_step", fake_predicted)
    result = counterfactual_step_diagnostics(None, "B", pairs, None, torch.device("cpu"))  # type: ignore[arg-type]

    expected_at: list[float] = []
    expected_after: list[float] = []
    n_before = 0
    for (_s, obs), (_s2, inter) in pairs:
        for state_o, state_i in zip(obs, inter, strict=True):
            if not (state_o.b_observed and state_i.b_observed):
                continue
            true_gap = abs(state_i.b - state_o.b)
            if state_i.t < intervention_t:
                n_before += 1
                assert true_gap == 0.0
            elif state_i.t == intervention_t:
                expected_at.append((0.4 - true_gap) ** 2)
            else:
                expected_after.append((0.1 - true_gap) ** 2)
    assert result["cf_n_steps_before_intervention"] == n_before
    assert result["cf_step_mse_before_intervention"] == 0.0
    assert result["cf_n_steps_at_intervention"] == len(expected_at) > 0
    assert result["cf_step_mse_at_intervention"] == pytest.approx(float(np.mean(expected_at)))
    assert result["cf_step_mse_after_intervention"] == pytest.approx(float(np.mean(expected_after)))
    assert result["cf_pred_gap_mean_at_intervention"] == pytest.approx(0.4)


def test_diagnostics_version_constant() -> None:
    assert DIAGNOSTICS_VERSION == "eval_diagnostics_v1"
