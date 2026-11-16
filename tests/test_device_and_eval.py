"""Device resolution, checkpoint rejection, detokenize, and aggregation guards."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from eval.checkpoint import load_scoring_checkpoint
from eval.decode import reconstruct_field_to_delimiter, reconstruct_with_skeleton
from eval.metrics import METRIC_KEYS, median_iqr
from model.architecture import build_model
from model.init import initialize_model
from obs_codecs.diagnostics import tokenize
from obs_codecs.encode_a import encode_A, observation_from_A
from obs_codecs.encode_b import encode_B, observation_from_B
from obs_codecs.encode_d import encode_D, observation_from_D
from train.config import build_run_config, get_budget_profile
from train.device import resolve_device
from train.loop import run_training
from train.matrix import aggregate_scientific_metrics, expand_matrix
from obs_codecs.vocab import VocabArtifact
from world.schema import Observation, observe
from world.simulate import simulate


def _obs() -> Observation:
    return observe(simulate(7, 2)[0])


def test_skeleton_roundtrips_codec_strings() -> None:
    obs = _obs()
    for encode, parse in (
        (encode_A, observation_from_A),
        (encode_B, observation_from_B),
        (encode_D, observation_from_D),
    ):
        text = encode(obs)
        rebuilt = reconstruct_with_skeleton(text, tokenize(text))
        assert rebuilt == text
        recovered = parse(rebuilt or "")
        assert recovered.noise_bin is obs.noise_bin
        assert recovered.n_samples == obs.n_samples
        assert recovered.source is obs.source


def test_resolve_device_cpu_and_require_gpu() -> None:
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(RuntimeError, match="require_gpu"):
        resolve_device("cpu", require_gpu=True)


def test_run_training_resolves_rocm_string_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vocab = VocabArtifact(
        token_to_id={"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "a": 4},
        id_to_token=["<pad>", "<bos>", "<eos>", "<unk>", "a"],
        metadata={},
    )
    monkeypatch.setattr("train.loop.load_codec_vocab", lambda *args, **kwargs: vocab)
    monkeypatch.setattr("train.device.gpu_available", lambda: False)
    config = build_run_config(
        codec="A",
        match_regime="compute_matched",
        replication_id=0,
        budget_profile="tier0",
    )
    with pytest.raises(RuntimeError, match="requires a ROCm/HIP GPU"):
        run_training(
            config,
            output_root=tmp_path,
            device="rocm",
            episode_pool=[[4] * 128],
            episode_limit=1,
            max_optimizer_steps=1,
            batch_size=1,
        )


def test_full_v0_min_replications_gate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_replications"):
        expand_matrix(
            profile=get_budget_profile("full_v0"),
            codecs=["A"],
            match_regimes=["compute_matched"],
            n_replications=1,
            output_root=tmp_path,
            min_replications=3,
        )


def test_median_iqr_uses_linear_percentile() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    stats = median_iqr(values)
    q1, median, q3 = np.percentile(np.asarray(values), [25, 50, 75], method="linear")
    assert stats["q1"] == float(q1)
    assert stats["median"] == float(median)
    assert stats["q3"] == float(q3)
    assert set(METRIC_KEYS) >= {
        "next_state_mse",
        "regime_accuracy",
        "intervention_accuracy",
        "calibration_ece",
    }


def test_aggregation_refuses_pilot_profile(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "tier0"
    runs.mkdir(parents=True)
    with pytest.raises(ValueError, match="plumbing"):
        aggregate_scientific_metrics(
            runs,
            profile_name="tier0",
            output_root=tmp_path,
            git_commit="abc",
            git_dirty=False,
        )


def test_load_scoring_checkpoint_rejects_non_final(tmp_path: Path) -> None:
    model = build_model()
    initialize_model(model, model_init_seed=1, codec_name="A")
    path = tmp_path / "step_1_non_scoring.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "scoring": False, "step": 1},
        path,
    )
    with pytest.raises(ValueError, match="non-final"):
        load_scoring_checkpoint(path, torch.device("cpu"))


def test_missing_parquet_does_not_fall_back_to_synthetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vocab = VocabArtifact(
        token_to_id={"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "a": 4},
        id_to_token=["<pad>", "<bos>", "<eos>", "<unk>", "a"],
        metadata={},
    )
    monkeypatch.setattr("train.loop.load_codec_vocab", lambda *args, **kwargs: vocab)
    config = replace(
        build_run_config(
            codec="A",
            match_regime="compute_matched",
            replication_id=0,
            budget_profile="tier0",
        ),
        dataset_split_train=str(tmp_path / "missing_{codec}.parquet"),
    )
    with pytest.raises(FileNotFoundError, match="encoded train split"):
        run_training(config, output_root=tmp_path, episode_limit=1, max_optimizer_steps=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/HIP GPU required")
def test_segment_mask_forward_backward_matches_cpu() -> None:
    torch.manual_seed(0)
    model = build_model()
    initialize_model(model, model_init_seed=11, codec_name="A")
    cpu = model
    gpu = build_model()
    gpu.load_state_dict(cpu.state_dict())
    gpu.cuda()
    ids = torch.randint(1, 40, (2, 32))
    segs = torch.zeros(2, 32, dtype=torch.long)
    segs[:, 16:] = 1
    segs[:, 28:] = -1
    logits_cpu = cpu(ids, segs)
    logits_gpu = gpu(ids.cuda(), segs.cuda())
    assert torch.isfinite(logits_gpu).all()
    assert torch.allclose(logits_cpu, logits_gpu.cpu(), atol=1e-4, rtol=1e-4)
    logits_cpu.sum().backward()
    logits_gpu.sum().backward()
    for left, right in zip(cpu.parameters(), gpu.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        assert torch.isfinite(right.grad).all()
        assert torch.allclose(left.grad, right.grad.cpu(), atol=1e-4, rtol=1e-4)


def test_aggregate_scientific_metrics_writes_median_iqr_summaries(
    tmp_path: Path,
) -> None:
    import json

    runs_root = tmp_path / "runs" / "full_v0"
    for rep, (a_mse, b_mse) in enumerate([(0.4, 0.2), (0.6, 0.1), (0.8, 0.3)]):
        for codec, mse in (("A", a_mse), ("B", b_mse)):
            rep_dir = runs_root / "compute_matched" / codec / f"rep{rep}"
            rep_dir.mkdir(parents=True)
            doc = {
                "scoring_checkpoint": "final",
                "splits": {
                    "val": {
                        "next_state_mse": mse,
                        "ood_regime_accuracy": None,
                    }
                },
            }
            (rep_dir / "metrics_eval.json").write_text(
                json.dumps(doc), encoding="utf-8"
            )

    written = aggregate_scientific_metrics(
        runs_root,
        profile_name="full_v0",
        output_root=tmp_path,
        git_commit="deadbeef",
        git_dirty=True,
    )
    assert len(written) == 1
    summary_path = tmp_path / "runs" / "aggregates" / "compute_matched" / "val" / "summary.json"
    assert written == [summary_path]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["aggregation"] == "median_iqr"
    assert summary["percentile_method"] == "linear"
    assert summary["git_commit"] == "deadbeef"
    assert summary["git_dirty"] is True
    assert summary["n_replications"] == 3
    assert "ood_regime_accuracy" not in summary["metrics"]
    assert summary["metrics"]["next_state_mse"]["A"]["median"] == pytest.approx(0.6)
    assert summary["metrics"]["next_state_mse"]["B"]["median"] == pytest.approx(0.2)
    assert summary["metrics"]["next_state_mse"]["A"]["iqr"] == pytest.approx(0.2)


def test_argmax_predictions_multi_window_exact_shift() -> None:
    from eval.score import _argmax_predictions
    from model.architecture import CONTEXT_LENGTH_L

    model = build_model()
    initialize_model(model, model_init_seed=5, codec_name="A")
    model.eval()
    stream = [4 + (i % 10) for i in range(CONTEXT_LENGTH_L + 15)]
    preds = _argmax_predictions(model, stream, torch.device("cpu"))
    assert len(preds) == len(stream)
    assert preds[0] is None
    assert all(p is not None for p in preds[1:])

    # Boundary position CONTEXT_LENGTH_L must come from the end of window 0.
    with torch.no_grad():
        w0 = torch.tensor([stream[:CONTEXT_LENGTH_L]], dtype=torch.long)
        logits0 = model(w0, torch.zeros_like(w0))
        expected_boundary = int(logits0[0, CONTEXT_LENGTH_L - 1].argmax().item())
    assert preds[CONTEXT_LENGTH_L] == expected_boundary



def test_generate_eval_splits_and_evaluate_runs_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from data.hdf5_episodes import write_episodes_hdf5_streaming
    from obs_codecs.vocab import SPECIAL_TOKENS, save_vocab
    from scripts.evaluate_runs import main as evaluate_main
    from scripts.generate_eval_splits import main as generate_eval_main
    from train.matrix import build_manifest, write_manifest_atomic

    n_steps = 4
    raw_dir = tmp_path / "raw"
    train_eps = [(100 + i, simulate(100 + i, n_steps)) for i in range(8)]
    val_eps = [(200 + i, simulate(200 + i, n_steps)) for i in range(2)]
    ood_eps = [(300 + i, simulate(300 + i, n_steps)) for i in range(2)]
    write_episodes_hdf5_streaming(
        raw_dir / "train.h5", iter(train_eps), n_episodes=8, n_steps=n_steps, split_name="train"
    )
    write_episodes_hdf5_streaming(
        raw_dir / "val.h5", iter(val_eps), n_episodes=2, n_steps=n_steps, split_name="val"
    )
    write_episodes_hdf5_streaming(
        raw_dir / "ood_partial_obs.h5",
        iter(ood_eps),
        n_episodes=2,
        n_steps=n_steps,
        split_name="ood_partial_obs",
    )

    eval_dir = tmp_path / "eval_splits"
    assert (
        generate_eval_main(
            ["--out-dir", str(eval_dir), "--n-episodes", "4", "--n-steps", str(n_steps)]
        )
        == 0
    )
    assert (eval_dir / "intervention.h5").is_file()
    assert (eval_dir / "counterfactual_observational.h5").is_file()
    assert (eval_dir / "counterfactual_intervened.h5").is_file()

    vocab_dir = tmp_path / "vocabs"
    content_tokens = sorted(
        {tok for _seed, states in train_eps for s in states for tok in tokenize(encode_A(observe(s)))}
    )
    id_to_token = list(SPECIAL_TOKENS) + content_tokens
    vocab_a = VocabArtifact(
        token_to_id={tok: idx for idx, tok in enumerate(id_to_token)},
        id_to_token=id_to_token,
        metadata={"codec": "A"},
    )
    save_vocab(vocab_a, vocab_dir / "vocab_A.json")

    profile = get_budget_profile("tier0")
    cells = expand_matrix(
        profile=profile,
        codecs=["A"],
        match_regimes=["compute_matched"],
        n_replications=1,
        output_root=tmp_path,
        runs_subdir="tier0",
    )
    cell = cells[0]
    (cell.run_dir / "checkpoints").mkdir(parents=True)
    cell.config.to_json(cell.run_dir / "config.json")
    model = build_model()
    initialize_model(model, model_init_seed=cell.config.model_init_seed, codec_name="A")
    torch.save(
        {"model_state_dict": model.state_dict(), "scoring": True, "step": 1},
        cell.run_dir / "checkpoints" / "final.pt",
    )
    runs_root = tmp_path / "runs" / "tier0"
    manifest = build_manifest(
        profile=profile,
        cells=cells,
        codecs=["A"],
        match_regimes=["compute_matched"],
        n_replications=1,
        git_commit="abc123",
        smoke_max_steps=1,
    )
    write_manifest_atomic(runs_root / "manifest.json", manifest)

    monkeypatch.setattr("eval.score.EVAL_MAX_EPISODES", 2)
    monkeypatch.setattr("eval.score.PROBE_TRAIN_EPISODES", 8)
    monkeypatch.setattr("eval.score.EVAL_SUITE_EPISODES", 2)
    monkeypatch.setattr("scripts.evaluate_runs.EVAL_MAX_EPISODES", 2)
    monkeypatch.setattr("scripts.evaluate_runs.PROBE_TRAIN_EPISODES", 8)
    monkeypatch.setattr("scripts.evaluate_runs.EVAL_SUITE_EPISODES", 2)

    rc = evaluate_main(
        [
            "--runs-root",
            str(runs_root),
            "--raw-dir",
            str(raw_dir),
            "--eval-dir",
            str(eval_dir),
            "--vocab-dir",
            str(vocab_dir),
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    metrics_doc = json.loads((cell.run_dir / "metrics_eval.json").read_text(encoding="utf-8"))
    assert metrics_doc["scoring_checkpoint"] == "final"
    assert metrics_doc["git_commit"]
    assert set(metrics_doc["splits"]) == {
        "val",
        "ood_partial_obs",
        "intervention",
        "counterfactual",
    }
    for split_metrics in metrics_doc["splits"].values():
        assert set(METRIC_KEYS).issubset(split_metrics.keys())
    updated_manifest = json.loads((runs_root / "manifest.json").read_text(encoding="utf-8"))
    assert updated_manifest["eval_status"] == "scored"
    assert updated_manifest["eval_schema_version"]
    assert metrics_doc["eval_schema_version"] == updated_manifest["eval_schema_version"]
    assert metrics_doc["splits"]["intervention"]["parse_fail_rate"] is not None
    assert 0.0 <= metrics_doc["splits"]["intervention"]["parse_fail_rate"] <= 1.0


def test_intervention_values_match_training_token_length_distribution() -> None:
    from collections import Counter

    from data.generate_episodes import generate_split
    from eval.score import (
        INTERVENTION_BASE_SEED,
        make_counterfactual_pairs,
        make_intervention_episodes,
    )
    from obs_codecs.encode_b import _format_value
    from world.simulate import DEFAULT_DIFFICULTY_V0

    n_episodes = 400
    int_eps_a = make_intervention_episodes(
        n_episodes, base_seed=INTERVENTION_BASE_SEED, n_steps=10
    )
    int_eps_b = make_intervention_episodes(
        n_episodes, base_seed=INTERVENTION_BASE_SEED, n_steps=10
    )
    assert int_eps_a == int_eps_b  # Deterministic from seed

    cf_pairs = make_counterfactual_pairs(64, base_seed=6_000_000, n_steps=10)
    cf_vals = [
        states[5].active_intervention.value
        for _, (_seed, states) in cf_pairs
        if states[5].active_intervention is not None
    ]
    assert len(set(cf_vals)) > 10  # Not a hardcoded constant like 0.85

    int_lengths: list[int] = []
    for _seed, states in int_eps_a:
        active = states[5].active_intervention
        assert active is not None
        int_lengths.append(len(tokenize(_format_value(active.value))))

    train_eps = list(
        generate_split(
            "train",
            n_episodes,
            DEFAULT_DIFFICULTY_V0,
            base_seed=10_000,
            n_steps=10,
        )
    )
    train_lengths: list[int] = []
    for _seed, states in train_eps:
        for st in states:
            train_lengths.append(len(tokenize(_format_value(st.a))))
            train_lengths.append(len(tokenize(_format_value(st.b))))

    int_hist = Counter(int_lengths)
    train_hist = Counter(train_lengths)
    all_keys = set(int_hist) | set(train_hist)
    tv_distance = 0.5 * sum(
        abs(int_hist[k] / len(int_lengths) - train_hist[k] / len(train_lengths))
        for k in all_keys
    )
    mean_int = sum(int_lengths) / len(int_lengths)
    mean_train = sum(train_lengths) / len(train_lengths)
    assert abs(mean_int - mean_train) < 0.25
    assert tv_distance < 0.10
    # Guard against short constant artifacts like 0.85 (4 tokens)
    assert sum(1 for k in int_lengths if k >= 11) / len(int_lengths) > 0.95


def test_intervention_accuracy_reports_parse_fail_rate_and_optional_decoder() -> None:
    from eval.score import intervention_accuracy, make_intervention_episodes
    from obs_codecs.vocab import SPECIAL_TOKENS

    int_eps = make_intervention_episodes(4, base_seed=5_000_000, n_steps=6)
    content_tokens = sorted(
        {
            tok
            for _seed, states in int_eps
            for s in states
            for tok in tokenize(encode_B(observe(s)))
        }
    )
    id_to_token = list(SPECIAL_TOKENS) + content_tokens
    vocab = VocabArtifact(
        token_to_id={tok: idx for idx, tok in enumerate(id_to_token)},
        id_to_token=id_to_token,
        metadata={"codec": "B"},
    )
    model = build_model()
    initialize_model(model, model_init_seed=1234, codec_name="B")
    res = intervention_accuracy(model, "B", int_eps, vocab, torch.device("cpu"))
    assert isinstance(res, dict)
    assert set(res.keys()) == {"accuracy", "parse_fail_rate"}
    assert res["accuracy"] is not None
    assert res["parse_fail_rate"] is not None
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["parse_fail_rate"] <= 1.0

    # Field-to-delimiter optional decoder handles numeric overflow/early-delimiter
    # while leaving structural errors (e.g. co-vary vs miss) broken.
    true_text = "(observe A=0.123456789) → (co-vary B=0.1234567891) → (meta n=3 source=sensor_both noise=medium)"
    toks = tokenize(true_text)
    # 1) Overflow digit into ')' after A's 9-digit value
    toks_overflow = list(toks)
    a_close_idx = toks_overflow.index(")")
    toks_overflow[a_close_idx] = "7"
    assert reconstruct_with_skeleton(true_text, toks_overflow) is not None
    with pytest.raises(ValueError):
        observation_from_B(reconstruct_with_skeleton(true_text, toks_overflow) or "")
    rebuilt_overflow = reconstruct_field_to_delimiter(true_text, toks_overflow)
    assert rebuilt_overflow is not None
    parsed_overflow = observation_from_B(rebuilt_overflow)
    assert parsed_overflow.a_value == pytest.approx(0.123456789)

    # 2) Early delimiter inside B's numeric span
    toks_early = list(toks)
    b_start = toks_early.index("B") + 2  # after B, =
    toks_early[b_start + 5] = ")"
    rebuilt_early = reconstruct_field_to_delimiter(true_text, toks_early)
    assert rebuilt_early is not None
    parsed_early = observation_from_B(rebuilt_early)
    assert parsed_early.b_value == pytest.approx(0.123)


