"""Additive evaluation diagnostics (Tier B2; recording only).

Nothing here redefines a locked metric. The locked ``splits`` block of
``metrics_eval.json`` (``METRIC_KEYS``) is produced by ``eval.score`` and is
left untouched; these diagnostics are written to a separate top-level
``diagnostics`` block, are not aggregated into the §5 summaries, and do not
change any parser, decoder, codec, split, or seed.

All diagnostics reuse the locked scoring helpers (teacher-forced argmax,
``skeleton`` / ``field_to_delimiter`` reconstruction, codec parsers) and the
same hit rule, so each one decomposes or contextualizes a locked number
instead of replacing it:

* intervention: the locked hit rule restricted to steps where both sensors are
  observed (a property of the world, identical for every codec), the hit rate
  over parsed steps, and codec-independent zero-predictor / graph-oracle hit
  rates on the same steps.
* counterfactual: step-level squared error of the paired ``|Δb|`` gap, split
  into the intervention step versus the steps before and after it.
* passive splits: median and winsorized squared next-state error beside the
  raw mean, which a single float-token error can dominate.

The oracle uses the simulator's latent ``(regime, hidden_c)`` and the active
intervention. It is an evaluation-side reference (I7), never a model input.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from eval import score as _score
from eval.score import Episode
from model.architecture import DecoderOnlyLM
from obs_codecs.vocab import VocabArtifact
from world.schema import (
    CausalRegime,
    InterventionTarget,
    State,
    observe,
    quantize_value,
)

DIAGNOSTICS_VERSION = "eval_diagnostics_v1"
# Upper quantile at which squared errors are clipped for the winsorized mean.
WINSOR_UPPER_QUANTILE = 0.95

Diagnostics = dict[str, float | int | None]


# ---------------------------------------------------------------------------
# Passive splits: outlier-robust next-state error
# ---------------------------------------------------------------------------


def squared_error_summary(
    errors: Sequence[float],
    *,
    upper_quantile: float = WINSOR_UPPER_QUANTILE,
) -> Diagnostics:
    """Raw mean, median, winsorized mean, max, and count of squared errors."""
    if not 0.0 < upper_quantile <= 1.0:
        raise ValueError(f"upper_quantile must be in (0, 1], got {upper_quantile}")
    if len(errors) == 0:
        return {
            "next_state_mse_raw": None,
            "next_state_mse_median": None,
            "next_state_mse_winsorized": None,
            "next_state_squared_error_max": None,
            "next_state_n_errors": 0,
        }
    array = np.asarray(errors, dtype=np.float64)
    cap = float(np.percentile(array, upper_quantile * 100.0, method="linear"))
    return {
        "next_state_mse_raw": float(array.mean()),
        "next_state_mse_median": float(np.median(array)),
        "next_state_mse_winsorized": float(np.minimum(array, cap).mean()),
        "next_state_squared_error_max": float(array.max()),
        "next_state_n_errors": int(array.size),
    }


def passive_split_diagnostics(
    model: DecoderOnlyLM,
    codec: str,
    episodes: Sequence[Episode],
    vocab: VocabArtifact,
    device: torch.device,
    *,
    decoder_mode: str = "skeleton",
) -> Diagnostics:
    """Robust next-state error summaries on the same episodes ``score_passive_split`` scores."""
    scored = _score._cap(episodes, _score.EVAL_MAX_EPISODES)
    errors, _parsed, _attempted = _score.next_state_errors(
        model, codec, scored, vocab, device, decoder_mode=decoder_mode
    )
    return squared_error_summary(errors)


# ---------------------------------------------------------------------------
# Intervention: symmetric and parse-conditioned accuracy, codec-free baselines
# ---------------------------------------------------------------------------


def _scored_value(state: State) -> float:
    """True value of the variable ``intervention_accuracy`` scores at this step."""
    intervention = state.active_intervention
    assert intervention is not None
    return state.b if intervention.target is InterventionTarget.A else state.a


def oracle_scored_value(state: State) -> float:
    """Noise-free graph prediction of the scored value at an intervention step.

    ``do(A=v)`` scores ``B`` and ``do(B=v)`` scores ``A``. The scored variable
    equals ``v`` when it is the intervened variable's causal child, equals
    ``hidden_c`` under ``COMMON_CAUSE`` (both sensors read the confounder), and
    has zero mean otherwise.
    """
    intervention = state.active_intervention
    if intervention is None:
        raise ValueError("oracle_scored_value needs a state with an active intervention")
    regime = state.latent.regime
    if regime is CausalRegime.COMMON_CAUSE:
        return float(state.latent.hidden_c)
    if regime is CausalRegime.A_CAUSES_B and intervention.target is InterventionTarget.A:
        return float(intervention.value)
    if regime is CausalRegime.B_CAUSES_A and intervention.target is InterventionTarget.B:
        return float(intervention.value)
    return 0.0


def _intervention_states(episodes: Sequence[Episode]) -> list[State]:
    states: list[State] = []
    for _seed, episode_states in _score._cap(episodes, _score.EVAL_SUITE_EPISODES):
        states.extend(s for s in episode_states if s.active_intervention is not None)
    return states


def _rate(hits: int, total: int) -> float | None:
    return None if total == 0 else hits / total


def intervention_baselines(episodes: Sequence[Episode]) -> Diagnostics:
    """Codec-independent hit rates of a constant-zero predictor and the graph oracle.

    Uses the locked bin rule (``quantize_value``) against the true scored value.
    The ``_both_observed`` variants restrict to steps with both sensors observed
    so they sit beside ``intervention_accuracy_both_observed``.
    """
    steps = _intervention_states(episodes)
    zero_hits = oracle_hits = 0
    both_total = zero_both = oracle_both = 0
    for state in steps:
        truth_bin = quantize_value(_scored_value(state))
        zero_hit = int(quantize_value(0.0) == truth_bin)
        oracle_hit = int(quantize_value(oracle_scored_value(state)) == truth_bin)
        zero_hits += zero_hit
        oracle_hits += oracle_hit
        if state.a_observed and state.b_observed:
            both_total += 1
            zero_both += zero_hit
            oracle_both += oracle_hit
    return {
        "zero_baseline_acc": _rate(zero_hits, len(steps)),
        "oracle_acc": _rate(oracle_hits, len(steps)),
        "zero_baseline_acc_both_observed": _rate(zero_both, both_total),
        "oracle_acc_both_observed": _rate(oracle_both, both_total),
        "baseline_n_steps": len(steps),
        "baseline_n_both_observed_steps": both_total,
    }


@torch.no_grad()
def intervention_diagnostics(
    model: DecoderOnlyLM,
    codec: str,
    episodes: Sequence[Episode],
    vocab: VocabArtifact,
    device: torch.device,
    *,
    decoder_mode: str = "skeleton",
) -> Diagnostics:
    """Locked-rule hit rates restricted to both-observed and to parsed steps.

    The hit rule and parse handling mirror ``eval.score.intervention_accuracy``:
    a step is a hit when the parsed scored value falls in the true value's bin;
    a step whose scored variable is unobserved, or that fails to parse, is a
    non-hit. ``intervention_accuracy_both_observed`` keeps parse failures in its
    denominator (it only removes the world-side partial-observation split), and
    ``intervention_accuracy_parsed`` drops parse failures from the denominator.
    """
    variant = _score.variant_for(codec)
    parser = _score._parser(codec)
    total = hits = parse_fails = 0
    both_total = both_hits = both_parse_fails = 0
    framed = _score._framed_token_spans(
        _score._cap(episodes, _score.EVAL_SUITE_EPISODES), variant, vocab
    )
    for seed, stream, spans in framed:
        predicted = _score._argmax_predictions(model, stream, device)
        for timestep, (start, end, state) in enumerate(spans):
            intervention = state.active_intervention
            if intervention is None:
                continue
            both_observed = state.a_observed and state.b_observed
            total += 1
            both_total += int(both_observed)
            hit, parse_failed = _intervention_step_outcome(
                state=state,
                parser=parser,
                variant=variant,
                vocab=vocab,
                seed=seed,
                timestep=timestep,
                predicted_ids=predicted[start:end],
                span_length=end - start,
                decoder_mode=decoder_mode,
            )
            hits += int(hit)
            parse_fails += int(parse_failed)
            if both_observed:
                both_hits += int(hit)
                both_parse_fails += int(parse_failed)
    parsed = total - parse_fails
    return {
        "intervention_accuracy_both_observed": _rate(both_hits, both_total),
        "intervention_accuracy_parsed": _rate(hits, parsed),
        "parse_fail_rate_both_observed": _rate(both_parse_fails, both_total),
        "intervention_hits": hits,
        "intervention_n_steps": total,
        "intervention_n_parsed_steps": parsed,
        "intervention_n_both_observed_steps": both_total,
    }


def _intervention_step_outcome(
    *,
    state: State,
    parser: Any,
    variant: Any,
    vocab: VocabArtifact,
    seed: int,
    timestep: int,
    predicted_ids: Sequence[int | None],
    span_length: int,
    decoder_mode: str,
) -> tuple[bool, bool]:
    """Return ``(hit, parse_failed)`` for one intervention step (locked hit rule)."""
    intervention = state.active_intervention
    assert intervention is not None
    pred_ids = [token_id for token_id in predicted_ids if token_id is not None]
    if len(pred_ids) != span_length:
        return False, True
    true_text = variant.encode_row(observe(state), seed, timestep)
    text = _score._reconstruct_text(
        true_text,
        _score._ids_to_tokens(pred_ids, vocab),
        decoder_mode=decoder_mode,
    )
    if text is None:
        return False, True
    try:
        recovered = parser(text)
    except (ValueError, KeyError, IndexError):
        return False, True
    if intervention.target is InterventionTarget.A:
        if recovered.b_value is None or not state.b_observed:
            return False, False
        return quantize_value(recovered.b_value) == quantize_value(state.b), False
    if recovered.a_value is None or not state.a_observed:
        return False, False
    return quantize_value(recovered.a_value) == quantize_value(state.a), False


# ---------------------------------------------------------------------------
# Counterfactual: attribute the pair error to the intervention step
# ---------------------------------------------------------------------------


@torch.no_grad()
def _predicted_b_by_step(
    model: DecoderOnlyLM,
    codec: str,
    episode: Episode,
    vocab: VocabArtifact,
    device: torch.device,
    *,
    decoder_mode: str,
) -> dict[int, float]:
    """Per-timestep recovered ``b`` (same parsing as ``_predicted_b_series``)."""
    variant = _score.variant_for(codec)
    parser = _score._parser(codec)
    seed, stream, spans = _score._framed_token_spans([episode], variant, vocab)[0]
    predicted = _score._argmax_predictions(model, stream, device)
    values: dict[int, float] = {}
    for timestep, (start, end, state) in enumerate(spans):
        pred_ids = [token_id for token_id in predicted[start:end] if token_id is not None]
        if len(pred_ids) != end - start:
            continue
        true_text = variant.encode_row(observe(state), seed, timestep)
        text = _score._reconstruct_text(
            true_text,
            _score._ids_to_tokens(pred_ids, vocab),
            decoder_mode=decoder_mode,
        )
        if text is None:
            continue
        try:
            recovered = parser(text)
        except (ValueError, KeyError, IndexError):
            continue
        if recovered.b_value is not None:
            values[timestep] = float(recovered.b_value)
    return values


def counterfactual_step_diagnostics(
    model: DecoderOnlyLM,
    codec: str,
    pairs: Sequence[tuple[Episode, Episode]],
    vocab: VocabArtifact,
    device: torch.device,
    *,
    decoder_mode: str = "skeleton",
) -> Diagnostics:
    """Step-level squared error of the paired ``|Δb|`` gap, by position vs the intervention.

    For every step where both twins have ``b`` observed and a recovered ``b``,
    the error is ``(|b̂_int - b̂_obs| - |b_int - b_obs|)^2``. Steps are bucketed
    into ``at`` (the intervention step of the intervened twin), ``before``, and
    ``after``. This decomposes where a codec's counterfactual error arises; it
    does not recompute the locked per-pair ``counterfactual_divergence_mse``.
    """
    buckets: dict[str, list[float]] = {"before": [], "at": [], "after": []}
    true_gaps_at: list[float] = []
    pred_gaps_at: list[float] = []
    for observational, intervened in list(pairs)[: _score.EVAL_SUITE_EPISODES]:
        intervention_steps = [
            s.t for s in intervened[1] if s.active_intervention is not None
        ]
        if not intervention_steps:
            continue
        intervention_t = intervention_steps[0]
        pred_obs = _predicted_b_by_step(
            model, codec, observational, vocab, device, decoder_mode=decoder_mode
        )
        pred_int = _predicted_b_by_step(
            model, codec, intervened, vocab, device, decoder_mode=decoder_mode
        )
        for timestep, (state_o, state_i) in enumerate(
            zip(observational[1], intervened[1], strict=False)
        ):
            if not (state_o.b_observed and state_i.b_observed):
                continue
            if timestep not in pred_obs or timestep not in pred_int:
                continue
            true_gap = abs(state_i.b - state_o.b)
            pred_gap = abs(pred_int[timestep] - pred_obs[timestep])
            error = (pred_gap - true_gap) ** 2
            if timestep < intervention_t:
                buckets["before"].append(error)
            elif timestep == intervention_t:
                buckets["at"].append(error)
                true_gaps_at.append(true_gap)
                pred_gaps_at.append(pred_gap)
            else:
                buckets["after"].append(error)

    def _mean(values: list[float]) -> float | None:
        return None if not values else float(np.mean(values))

    return {
        "cf_step_mse_before_intervention": _mean(buckets["before"]),
        "cf_step_mse_at_intervention": _mean(buckets["at"]),
        "cf_step_mse_after_intervention": _mean(buckets["after"]),
        "cf_true_gap_mean_at_intervention": _mean(true_gaps_at),
        "cf_pred_gap_mean_at_intervention": _mean(pred_gaps_at),
        "cf_n_steps_before_intervention": len(buckets["before"]),
        "cf_n_steps_at_intervention": len(buckets["at"]),
        "cf_n_steps_after_intervention": len(buckets["after"]),
    }


# ---------------------------------------------------------------------------
# Block assembly for metrics_eval.json
# ---------------------------------------------------------------------------


def compute_eval_diagnostics(
    model: DecoderOnlyLM,
    codec: str,
    vocab: VocabArtifact,
    device: torch.device,
    *,
    val_episodes: Sequence[Episode],
    ood_episodes: Sequence[Episode],
    intervention_episodes: Sequence[Episode],
    counterfactual_pairs: Sequence[tuple[Episode, Episode]],
    decoder_mode: str = "skeleton",
) -> dict[str, Any]:
    """The ``diagnostics`` block written next to the locked ``splits`` block."""
    return {
        "version": DIAGNOSTICS_VERSION,
        "kind": "recording_change_no_methodology_change",
        "locked_metrics_unchanged": True,
        "winsor_upper_quantile": WINSOR_UPPER_QUANTILE,
        "splits": {
            "val": passive_split_diagnostics(
                model, codec, val_episodes, vocab, device, decoder_mode=decoder_mode
            ),
            "ood_partial_obs": passive_split_diagnostics(
                model, codec, ood_episodes, vocab, device, decoder_mode=decoder_mode
            ),
            "intervention": {
                **intervention_diagnostics(
                    model,
                    codec,
                    intervention_episodes,
                    vocab,
                    device,
                    decoder_mode=decoder_mode,
                ),
                **intervention_baselines(intervention_episodes),
            },
            "counterfactual": counterfactual_step_diagnostics(
                model,
                codec,
                counterfactual_pairs,
                vocab,
                device,
                decoder_mode=decoder_mode,
            ),
        },
    }
