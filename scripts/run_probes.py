"""Frozen linear probes and CKA / SVCCA / Procrustes on a shared world fixture.

Models are loaded from ``checkpoints/final.pt`` with gradients disabled.
The fixture is the leading states of ``--raw-dir/val.h5`` (or a deterministic
simulator draw when that file is absent). Every codec sees the same states.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.hdf5_episodes import read_episodes_hdf5  # noqa: E402
from eval.checkpoint import load_scoring_checkpoint  # noqa: E402
from eval.score import episode_hidden  # noqa: E402
from probes.rsa import linear_cka, orthogonal_procrustes, svcca_mean_correlation  # noqa: E402
from train.data import load_codec_vocab  # noqa: E402
from train.device import resolve_device  # noqa: E402
from train.loop import git_commit, git_dirty  # noqa: E402
from world.schema import CausalRegime  # noqa: E402
from world.simulate import DEFAULT_DIFFICULTY_V0, simulate  # noqa: E402

DEFAULT_CODECS = ("A", "B", "C", "D", "B_shuffle", "B_reverse", "B_bag")
_REGIMES = tuple(CausalRegime)


def _states_from_raw(raw_dir: Path, n_states: int) -> list[tuple[int, list]]:
    val = raw_dir / "val.h5"
    if val.is_file():
        episodes = read_episodes_hdf5(val, max_episodes=max(1, n_states))
        return episodes
    episodes = []
    for index in range(max(1, (n_states + 29) // 30)):
        seed = 7_000_000 + index
        episodes.append((seed, simulate(seed, 30, difficulty=DEFAULT_DIFFICULTY_V0)))
    return episodes


def _flatten_states(episodes: list[tuple[int, list]], n_states: int) -> list[tuple[int, list]]:
    """One-state episodes so each hidden is tied to one world state."""
    flat: list[tuple[int, list]] = []
    for seed, states in episodes:
        for state in states:
            flat.append((seed, [state]))
            if len(flat) >= n_states:
                return flat
    return flat


def _ridge_r2(features: np.ndarray, target: np.ndarray) -> float:
    bias = np.ones((features.shape[0], 1), dtype=np.float64)
    design = np.concatenate([features.astype(np.float64), bias], axis=1)
    weight, *_rest = np.linalg.lstsq(design, target.astype(np.float64), rcond=None)
    predicted = design @ weight
    residual = float(np.mean((predicted - target) ** 2))
    variance = float(np.var(target))
    if variance <= 0:
        return 0.0
    return float(1.0 - residual / variance)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=_ROOT / "artifacts" / "probes")
    parser.add_argument("--vocab-dir", type=Path, default=_ROOT / "artifacts" / "vocabs")
    parser.add_argument("--n-states", type=int, default=10_000)
    parser.add_argument("--device", choices=("auto", "rocm", "cuda", "cpu"), default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--match-regime", default="compute_matched")
    parser.add_argument("--replication", type=int, default=0)
    args = parser.parse_args(argv)
    device = resolve_device(args.device, require_gpu=args.require_gpu)
    fixture = _flatten_states(_states_from_raw(args.raw_dir, args.n_states), args.n_states)
    if len(fixture) < 2:
        raise SystemExit("probe fixture needs at least 2 states")
    activations: dict[str, np.ndarray] = {}
    probe_report: dict[str, dict[str, float]] = {}
    labels = np.asarray(
        [_REGIMES.index(episode[1][0].latent.regime) for episode in fixture],
        dtype=np.int64,
    )
    hidden_c = np.asarray(
        [episode[1][0].latent.hidden_c for episode in fixture], dtype=np.float64
    )
    next_b = np.asarray(
        [episode[1][0].b for episode in fixture], dtype=np.float64
    )
    for codec in DEFAULT_CODECS:
        run_dir = (
            args.runs_root / args.match_regime / codec / f"rep{args.replication}"
        )
        ckpt = run_dir / "checkpoints" / "final.pt"
        if not ckpt.is_file():
            print(f"skip {codec}: missing {ckpt}")
            continue
        vocab = load_codec_vocab(codec, args.vocab_dir)
        model, _payload = load_scoring_checkpoint(ckpt, device)
        rows = [
            episode_hidden(model, codec, episode, vocab, device) for episode in fixture
        ]
        features = np.stack(rows)
        activations[codec] = features
        # Regime accuracy of a probe fit on the first half, scored on the second.
        split = max(1, len(fixture) // 2)
        from eval.score import _fit_regime_probe, _probe_probabilities

        weight = _fit_regime_probe(features[:split], labels[:split])
        probabilities = _probe_probabilities(features[split:], weight)
        accuracy = float(np.mean(probabilities.argmax(axis=1) == labels[split:]))
        probe_report[codec] = {
            "hidden_c_r2": _ridge_r2(features, hidden_c),
            "regime_accuracy": accuracy,
            "next_b_r2": _ridge_r2(features, next_b),
            "n_states": int(features.shape[0]),
        }
        print(f"probe {codec} states={features.shape[0]}")
    codecs = list(activations)
    similarity: dict[str, dict[str, dict[str, float]]] = {
        "cka": {},
        "svcca": {},
        "procrustes": {},
    }
    for left in codecs:
        similarity["cka"][left] = {}
        similarity["svcca"][left] = {}
        similarity["procrustes"][left] = {}
        for right in codecs:
            similarity["cka"][left][right] = linear_cka(
                activations[left], activations[right]
            )
            similarity["svcca"][left][right] = svcca_mean_correlation(
                activations[left], activations[right]
            )
            similarity["procrustes"][left][right] = orthogonal_procrustes(
                activations[left], activations[right]
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "match_regime": args.match_regime,
        "replication": args.replication,
        "n_states": len(fixture),
        "frozen": True,
        "probes": probe_report,
        "similarity": similarity,
    }
    path = args.out_dir / "probes.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
