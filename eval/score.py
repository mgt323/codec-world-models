"""Trained-model evaluation in world-state space (EXPERIMENT_PLAN §7).

Stage-1 checkpoints stay frozen. Regime probabilities come from a linear
probe fit on train episodes and scored on the requested split (I12). The
probe is not part of the training loss.

Sample caps are identical for every codec. ``ood_regime_accuracy`` stays
null because that OOD axis is not unlocked.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from data.encode_episodes import CodecVariant, codec_variants
from eval.decode import reconstruct_with_skeleton
from eval.metrics import (
    brier_score,
    empty_metrics,
    expected_calibration_error,
    mean_or_none,
)
from model.architecture import CONTEXT_LENGTH_L, DecoderOnlyLM
from obs_codecs.diagnostics import tokenize
from obs_codecs.encode_a import observation_from_A
from obs_codecs.encode_b import observation_from_B, observation_from_B_unordered
from obs_codecs.encode_c import observation_from_C
from obs_codecs.encode_d import observation_from_D
from obs_codecs.vocab import BOS_ID, EOS_ID, VocabArtifact, encode_string_ids
from world.schema import CausalRegime, Intervention, InterventionTarget, Observation, State, observe
from world.simulate import DEFAULT_DIFFICULTY_V0

# Shared across codecs. Not a RunConfig field and not a matching axis.
EVAL_MAX_EPISODES = 64
PROBE_TRAIN_EPISODES = 128
EVAL_SUITE_EPISODES = 32
PROBE_L2 = 1e-2
INTERVENTION_BASE_SEED = 5_000_000
COUNTERFACTUAL_BASE_SEED = 6_000_000

_REGIMES: tuple[CausalRegime, ...] = tuple(CausalRegime)
Episode = tuple[int, list[State]]

_OBSERVATION_PARSERS: dict[str, Callable[[str], Observation]] = {
    "A": observation_from_A,
    "B": observation_from_B,
    "C": observation_from_C,
    "D": observation_from_D,
    "B_shuffle": observation_from_B_unordered,
    "B_reverse": observation_from_B_unordered,
    "B_bag": observation_from_B_unordered,
}


def variant_for(codec: str) -> CodecVariant:
    for variant in codec_variants():
        if variant.stem == codec:
            return variant
    raise ValueError(f"Unknown codec {codec!r}")


def _parser(codec: str) -> Callable[[str], Observation]:
    try:
        return _OBSERVATION_PARSERS[codec]
    except KeyError:
        raise ValueError(f"Unknown codec {codec!r}") from None


def _framed_token_spans(
    episodes: Sequence[Episode],
    variant: CodecVariant,
    vocab: VocabArtifact,
) -> list[tuple[list[int], list[tuple[int, int, State]]]]:
    """Return ``(stream, spans)`` per episode.

    ``spans`` entries are ``(start, end, state)`` into ``stream`` for each
    observation's tokens. The stream is ``<bos> + tokens + <eos>``.
    """
    packed: list[tuple[int, list[int], list[tuple[int, int, State]]]] = []
    for seed, states in episodes:
        stream = [BOS_ID]
        spans: list[tuple[int, int, State]] = []
        for timestep, state in enumerate(states):
            text = variant.encode_row(observe(state), int(seed), int(timestep))
            ids = encode_string_ids(text, vocab.token_to_id)
            start = len(stream)
            stream.extend(ids)
            spans.append((start, len(stream), state))
        stream.append(EOS_ID)
        packed.append((int(seed), stream, spans))
    return packed


@torch.no_grad()
def _argmax_predictions(
    model: DecoderOnlyLM,
    stream: Sequence[int],
    device: torch.device,
) -> list[int | None]:
    """Teacher-forced argmax of the token at each stream index."""
    predicted: list[int | None] = [None] * len(stream)
    if len(stream) < 2:
        return predicted
    offset = 0
    segment = 0
    while offset < len(stream):
        chunk = list(stream[offset : offset + CONTEXT_LENGTH_L])
        ids = torch.tensor([chunk], dtype=torch.long, device=device)
        segs = torch.full_like(ids, segment)
        logits = model(ids, segs)
        top_ids: list[int] = logits[0, : len(chunk)].argmax(dim=-1).cpu().tolist()
        for index, token_id in enumerate(top_ids):
            target = offset + index + 1
            if target >= len(stream):
                break
            predicted[target] = int(token_id)
        offset += CONTEXT_LENGTH_L
    return predicted


def _ids_to_tokens(ids: Sequence[int], vocab: VocabArtifact) -> list[str]:
    out: list[str] = []
    unk = vocab.id_to_token[3] if len(vocab.id_to_token) > 3 else "<unk>"
    for token_id in ids:
        if 0 <= token_id < len(vocab.id_to_token):
            out.append(vocab.id_to_token[token_id])
        else:
            out.append(unk)
    return out


def next_state_errors(
    model: DecoderOnlyLM,
    codec: str,
    episodes: Sequence[Episode],
    vocab: VocabArtifact,
    device: torch.device,
) -> tuple[list[float], int, int]:
    """Squared errors of recovered continuous values, plus parse counts.

    Compares the teacher-forced argmax reconstruction of observation ``t+1``
    with the true observation. A and B values contribute only when both the
    true state and the parse expose that sensor.
    """
    variant = variant_for(codec)
    parser = _parser(codec)
    errors: list[float] = []
    attempted = 0
    parsed = 0
    for seed, stream, spans in _framed_token_spans(episodes, variant, vocab):
        predicted = _argmax_predictions(model, stream, device)
        for index in range(len(spans) - 1):
            start, end, target = spans[index + 1]
            attempted += 1
            pred_ids = [token_id for token_id in predicted[start:end] if token_id is not None]
            if len(pred_ids) != end - start:
                continue
            true_text = variant.encode_row(observe(target), seed, index + 1)
            text = reconstruct_with_skeleton(true_text, _ids_to_tokens(pred_ids, vocab))
            if text is None:
                continue
            try:
                recovered = parser(text)
            except (ValueError, KeyError, IndexError):
                continue
            parsed += 1
            true = observe(target)
            errors.extend(_paired_squared_errors(true, recovered))
    return errors, parsed, attempted


def _paired_squared_errors(true: Observation, recovered: Observation) -> list[float]:
    errors: list[float] = []
    if true.a_observed and true.a_value is not None and recovered.a_value is not None:
        errors.append((recovered.a_value - true.a_value) ** 2)
    if true.b_observed and true.b_value is not None and recovered.b_value is not None:
        errors.append((recovered.b_value - true.b_value) ** 2)
    return errors


@torch.no_grad()
def episode_hidden(
    model: DecoderOnlyLM,
    codec: str,
    episode: Episode,
    vocab: VocabArtifact,
    device: torch.device,
) -> np.ndarray:
    """Hidden state at the final token of one framed episode (last ``L`` tokens)."""
    variant = variant_for(codec)
    _seed, stream, _spans = _framed_token_spans([episode], variant, vocab)[0]
    window = stream[-CONTEXT_LENGTH_L:]
    ids = torch.tensor([window], dtype=torch.long, device=device)
    segs = torch.zeros_like(ids)
    hidden = model.hidden_states(ids, segs)
    return hidden[0, -1].detach().float().cpu().numpy()


def _fit_regime_probe(
    features: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Ridge map from hidden states to 4 regime scores. Returns ``(D+1, C)``."""
    n_classes = len(_REGIMES)
    one_hot = np.eye(n_classes, dtype=np.float64)[labels]
    bias = np.ones((features.shape[0], 1), dtype=np.float64)
    design = np.concatenate([features.astype(np.float64), bias], axis=1)
    gram = design.T @ design
    penalty = PROBE_L2 * np.eye(gram.shape[0])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(gram + penalty, design.T @ one_hot)


def _probe_probabilities(features: np.ndarray, weight: np.ndarray) -> np.ndarray:
    bias = np.ones((features.shape[0], 1), dtype=np.float64)
    design = np.concatenate([features.astype(np.float64), bias], axis=1)
    logits = design @ weight
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def regime_probe_scores(
    model: DecoderOnlyLM,
    codec: str,
    train_episodes: Sequence[Episode],
    eval_episodes: Sequence[Episode],
    vocab: VocabArtifact,
    device: torch.device,
) -> dict[str, float] | None:
    """Fit a frozen linear regime probe on train episodes; score ``eval_episodes``."""
    if len(train_episodes) < 2 or len(eval_episodes) < 1:
        return None
    train_x = np.stack(
        [episode_hidden(model, codec, ep, vocab, device) for ep in train_episodes]
    )
    train_y = np.asarray(
        [_REGIMES.index(ep[1][0].latent.regime) for ep in train_episodes],
        dtype=np.int64,
    )
    if len(set(train_y.tolist())) < 2:
        return None
    weight = _fit_regime_probe(train_x, train_y)
    eval_x = np.stack(
        [episode_hidden(model, codec, ep, vocab, device) for ep in eval_episodes]
    )
    labels = np.asarray(
        [_REGIMES.index(ep[1][0].latent.regime) for ep in eval_episodes],
        dtype=np.int64,
    )
    probabilities = _probe_probabilities(eval_x, weight)
    predicted = probabilities.argmax(axis=1)
    return {
        "regime_accuracy": float(np.mean(predicted == labels)),
        "calibration_ece": expected_calibration_error(probabilities, labels),
        "calibration_brier": brier_score(probabilities, labels, len(_REGIMES)),
    }


def _cap(episodes: Sequence[Episode], limit: int) -> list[Episode]:
    return list(episodes[:limit])


def score_passive_split(
    model: DecoderOnlyLM,
    codec: str,
    episodes: Sequence[Episode],
    vocab: VocabArtifact,
    device: torch.device,
    *,
    train_episodes: Sequence[Episode],
    partial_obs: bool,
    transfer_model_codec: str | None = None,
    transfer_episodes: Sequence[Episode] | None = None,
) -> dict[str, float | None]:
    """Tasks 1, 2, 5, 6, and 8 for one passive split."""
    metrics = empty_metrics()
    scored = _cap(episodes, EVAL_MAX_EPISODES)
    errors, parsed, attempted = next_state_errors(model, codec, scored, vocab, device)
    metrics["next_state_mse"] = mean_or_none(errors)
    metrics["next_state_parse_rate"] = (
        float(parsed / attempted) if attempted else None
    )
    if partial_obs:
        metrics["partial_obs_next_state_mse"] = metrics["next_state_mse"]
    probe = regime_probe_scores(
        model,
        codec,
        _cap(train_episodes, PROBE_TRAIN_EPISODES),
        scored,
        vocab,
        device,
    )
    if probe is not None:
        metrics["regime_accuracy"] = probe["regime_accuracy"]
        metrics["calibration_ece"] = probe["calibration_ece"]
        metrics["calibration_brier"] = probe["calibration_brier"]
    if transfer_model_codec is not None and transfer_episodes is not None:
        # This checkpoint is scored on the other codec's strings, ids looked up
        # in this checkpoint's vocab (symbol alignment by surface token).
        transfer_errors, _parsed, _attempted = next_state_errors(
            model,
            transfer_model_codec,
            _cap(transfer_episodes, EVAL_MAX_EPISODES),
            vocab,
            device,
        )
        metrics["transfer_aligned_next_state_mse"] = mean_or_none(transfer_errors)
    return metrics


def intervention_accuracy(
    model: DecoderOnlyLM,
    codec: str,
    episodes: Sequence[Episode],
    vocab: VocabArtifact,
    device: torch.device,
) -> float | None:
    """Fraction of post-intervention steps whose affected value bin matches.

    ``do(A)`` scores B; ``do(B)`` scores A. Episodes with no intervention are
    skipped.
    """
    from world.schema import quantize_value

    variant = variant_for(codec)
    parser = _parser(codec)
    hits = 0
    total = 0
    framed = _framed_token_spans(_cap(episodes, EVAL_SUITE_EPISODES), variant, vocab)
    for seed, stream, spans in framed:
        predicted = _argmax_predictions(model, stream, device)
        for timestep, (start, end, state) in enumerate(spans):
            intervention = state.active_intervention
            if intervention is None:
                continue
            pred_ids = [token_id for token_id in predicted[start:end] if token_id is not None]
            if len(pred_ids) != end - start:
                total += 1
                continue
            true_text = variant.encode_row(observe(state), seed, timestep)
            text = reconstruct_with_skeleton(true_text, _ids_to_tokens(pred_ids, vocab))
            if text is None:
                total += 1
                continue
            total += 1
            try:
                recovered = parser(text)
            except (ValueError, KeyError, IndexError):
                continue
            if intervention.target is InterventionTarget.A:
                if recovered.b_value is None or not state.b_observed:
                    continue
                hits += int(quantize_value(recovered.b_value) == quantize_value(state.b))
            else:
                if recovered.a_value is None or not state.a_observed:
                    continue
                hits += int(quantize_value(recovered.a_value) == quantize_value(state.a))
    if total == 0:
        return None
    return hits / total


def counterfactual_divergence_mse(
    model: DecoderOnlyLM,
    codec: str,
    pairs: Sequence[tuple[Episode, Episode]],
    vocab: VocabArtifact,
    device: torch.device,
) -> float | None:
    """MSE between true and predicted mean |Δb| across counterfactual twins."""
    errors: list[float] = []
    for observational, intervened in list(pairs)[:EVAL_SUITE_EPISODES]:
        true_gap = _mean_abs_b_gap(observational[1], intervened[1])
        pred_obs = _predicted_b_series(model, codec, observational, vocab, device)
        pred_int = _predicted_b_series(model, codec, intervened, vocab, device)
        if true_gap is None or pred_obs is None or pred_int is None:
            continue
        errors.append((abs(pred_int - pred_obs) - true_gap) ** 2)
    return mean_or_none(errors)


def _mean_abs_b_gap(left: Sequence[State], right: Sequence[State]) -> float | None:
    gaps: list[float] = []
    for state_l, state_r in zip(left, right, strict=False):
        if state_l.b_observed and state_r.b_observed:
            gaps.append(abs(state_l.b - state_r.b))
    return mean_or_none(gaps)


def _predicted_b_series(
    model: DecoderOnlyLM,
    codec: str,
    episode: Episode,
    vocab: VocabArtifact,
    device: torch.device,
) -> float | None:
    variant = variant_for(codec)
    parser = _parser(codec)
    seed, stream, spans = _framed_token_spans([episode], variant, vocab)[0]
    predicted = _argmax_predictions(model, stream, device)
    values: list[float] = []
    for timestep, (start, end, state) in enumerate(spans):
        pred_ids = [token_id for token_id in predicted[start:end] if token_id is not None]
        if len(pred_ids) != end - start:
            continue
        true_text = variant.encode_row(observe(state), seed, timestep)
        text = reconstruct_with_skeleton(true_text, _ids_to_tokens(pred_ids, vocab))
        if text is None:
            continue
        try:
            recovered = parser(text)
        except (ValueError, KeyError, IndexError):
            continue
        if recovered.b_value is not None:
            values.append(recovered.b_value)
    return mean_or_none(values)


def make_intervention_episodes(
    n_episodes: int,
    *,
    base_seed: int,
    n_steps: int,
) -> list[Episode]:
    """Persistent-split generator: alternating ``do(A)`` / ``do(B)`` at mid-episode."""
    from data.generate_episodes import generate_intervention_episode

    episodes: list[Episode] = []
    for index in range(n_episodes):
        seed = base_seed + index
        target = InterventionTarget.A if index % 2 == 0 else InterventionTarget.B
        intervention = Intervention(target=target, value=0.85, timestep=n_steps // 2)
        states = generate_intervention_episode(
            seed, n_steps, DEFAULT_DIFFICULTY_V0, intervention
        )
        episodes.append((seed, states))
    return episodes


def make_counterfactual_pairs(
    n_pairs: int,
    *,
    base_seed: int,
    n_steps: int,
) -> list[tuple[Episode, Episode]]:
    from data.generate_episodes import generate_counterfactual_pair

    pairs: list[tuple[Episode, Episode]] = []
    intervention = Intervention(
        target=InterventionTarget.A, value=0.85, timestep=n_steps // 2
    )
    for index in range(n_pairs):
        seed = base_seed + index
        observational, intervened = generate_counterfactual_pair(
            seed, n_steps, DEFAULT_DIFFICULTY_V0, intervention
        )
        pairs.append(((seed, observational), (seed, intervened)))
    return pairs


def protocol_record() -> dict[str, Any]:
    return {
        "eval_max_episodes": EVAL_MAX_EPISODES,
        "probe_train_episodes": PROBE_TRAIN_EPISODES,
        "eval_suite_episodes": EVAL_SUITE_EPISODES,
        "precision": "fp32",
        "regime_readout": "frozen_linear_probe_fit_on_train",
        "next_state": "teacher_forced_argmax_parse",
        "ood_regime": "not_unlocked",
        "transfer": "A<->B symbol alignment by surface token; null for other codecs",
    }


def roundtrip_token_count(text: str) -> int:
    """Helper for tests that guard the tokenizer against silent drift."""
    return len(tokenize(text))
