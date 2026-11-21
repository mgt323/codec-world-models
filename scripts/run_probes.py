"""Frozen ridge probes and CKA / SVCCA / Procrustes on a shared world fixture.

Models are loaded from ``checkpoints/final.pt`` with gradients disabled.
The fixture is the leading episodes of ``--raw-dir/val.h5`` (or a deterministic
simulator draw when that file is absent). Every codec sees the same states.

Probe protocol (identical for every codec):

* One sample per ``(episode, t)``: the model reads the framed prefix
  ``states[0..t]`` (original timesteps preserved) and the hidden state at the
  last token of the final layer is the representation.
* Targets: ``hidden_c`` and the regime at time ``t``, and ``next_b`` =
  ``state_{t+1}.b`` (only for ``t`` < last step of the episode).
* Train/test split is by *episode* (first half train, second half test), so
  prefixes of one episode never straddle the split. Ridge probes are fit on the
  train rows only; ``hidden_c_r2``, ``next_b_r2`` and ``regime_accuracy`` are
  scored on the held-out test rows.

Single-cell mode writes ``--out-dir/probes.json``. Matrix mode (``--matrix``)
loops over regimes and replications and writes
``--out-dir/{regime}/rep{r}/probes.json``, the layout ``eval/report.py`` reads.

Provenance: ``git_commit`` / ``git_dirty`` are inherited from the training
``manifest.json`` in ``--runs-root`` (so ``eval/report.py`` can cross-check them
against the manifest); the commit that executed the probes is recorded
separately as ``probe_git_commit`` / ``probe_git_dirty``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.hdf5_episodes import episodes_sha256, read_episodes_hdf5  # noqa: E402
from eval.checkpoint import load_scoring_checkpoint  # noqa: E402
from eval.score import _fit_regime_probe, _probe_probabilities, episode_hidden  # noqa: E402
from probes.linear import (  # noqa: E402
    PROBE_RIDGE_ALPHA,
    PROBE_TRAIN_FRACTION,
    heldout_ridge_r2,
    split_by_group,
)
from probes.rsa import linear_cka, orthogonal_procrustes, svcca_mean_correlation  # noqa: E402
from train.data import file_sha256, load_codec_vocab  # noqa: E402
from train.device import resolve_device  # noqa: E402
from train.loop import git_commit, git_dirty  # noqa: E402
from train.matrix import MANIFEST_NAME, load_manifest  # noqa: E402
from world.schema import CausalRegime, State  # noqa: E402
from world.simulate import DEFAULT_DIFFICULTY_V0, simulate  # noqa: E402

DEFAULT_CODECS = ("A", "B", "C", "D", "B_shuffle", "B_reverse", "B_bag")
DEFAULT_MATCH_REGIMES = ("compute_matched", "info_matched")
STEPS_PER_EPISODE = 30
FALLBACK_SEED_BASE = 7_000_000
_REGIMES = tuple(CausalRegime)


@dataclass(frozen=True, slots=True)
class ProbeSample:
    """One probe row: a framed prefix and the targets read off its last state."""

    episode_index: int
    seed: int
    t: int
    prefix: list[State]
    hidden_c: float
    regime_index: int
    next_b: float | None

    @property
    def episode(self) -> tuple[int, list[State]]:
        return (self.seed, self.prefix)


def _states_from_raw(raw_dir: Path, n_states: int) -> list[tuple[int, list[State]]]:
    """Leading episodes covering ``n_states`` states (``ceil(n_states / 30)``)."""
    n_episodes = max(1, math.ceil(n_states / STEPS_PER_EPISODE))
    val = raw_dir / "val.h5"
    if val.is_file():
        return read_episodes_hdf5(val, max_episodes=n_episodes)
    episodes: list[tuple[int, list[State]]] = []
    for index in range(n_episodes):
        seed = FALLBACK_SEED_BASE + index
        episodes.append((seed, simulate(seed, STEPS_PER_EPISODE, difficulty=DEFAULT_DIFFICULTY_V0)))
    return episodes


def _flatten_states(
    episodes: list[tuple[int, list[State]]], n_states: int
) -> list[ProbeSample]:
    """Expand episodes into prefix samples, preserving each state's ``t``.

    Sample ``(episode, t)`` encodes ``states[0..t]``, so codec-specific
    per-timestep transforms (E variants) see the same timesteps as in training.
    """
    samples: list[ProbeSample] = []
    for episode_index, (seed, states) in enumerate(episodes):
        for position, state in enumerate(states):
            if state.t != position:
                raise ValueError(
                    f"episode seed={seed}: state.t={state.t} at position {position}; "
                    "probe prefixes require contiguous timesteps starting at 0"
                )
            next_b = states[position + 1].b if position + 1 < len(states) else None
            samples.append(
                ProbeSample(
                    episode_index=episode_index,
                    seed=int(seed),
                    t=int(state.t),
                    prefix=list(states[: position + 1]),
                    hidden_c=float(state.latent.hidden_c),
                    regime_index=_REGIMES.index(state.latent.regime),
                    next_b=None if next_b is None else float(next_b),
                )
            )
            if len(samples) >= n_states:
                return samples
    return samples


def _training_provenance(runs_root: Path) -> dict[str, object]:
    """Training commit/dirty from ``runs_root/manifest.json`` (else current HEAD)."""
    manifest = load_manifest(runs_root / MANIFEST_NAME)
    if manifest is not None and manifest.get("git_commit"):
        return {
            "git_commit": str(manifest["git_commit"]),
            "git_dirty": bool(manifest.get("git_dirty", False)),
            "git_commit_source": "manifest",
            "budget_profile": manifest.get("budget_profile"),
            "n_replications": manifest.get("n_replications"),
        }
    return {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "git_commit_source": "current_head_no_manifest",
        "budget_profile": None,
        "n_replications": None,
    }


def _probe_codec(
    features: np.ndarray,
    samples: list[ProbeSample],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray([s.regime_index for s in samples], dtype=np.int64)
    hidden_c = np.asarray([s.hidden_c for s in samples], dtype=np.float64)
    has_next = np.asarray([s.next_b is not None for s in samples], dtype=bool)
    next_b = np.asarray(
        [0.0 if s.next_b is None else s.next_b for s in samples], dtype=np.float64
    )
    weight = _fit_regime_probe(features[train_mask], labels[train_mask])
    probabilities = _probe_probabilities(features[test_mask], weight)
    accuracy = float(np.mean(probabilities.argmax(axis=1) == labels[test_mask]))
    train_next = train_mask & has_next
    test_next = test_mask & has_next
    return {
        "hidden_c_r2": heldout_ridge_r2(
            features[train_mask],
            hidden_c[train_mask],
            features[test_mask],
            hidden_c[test_mask],
        ),
        "regime_accuracy": accuracy,
        "next_b_r2": heldout_ridge_r2(
            features[train_next], next_b[train_next], features[test_next], next_b[test_next]
        ),
        "n_states": int(features.shape[0]),
        "n_train_states": int(train_mask.sum()),
        "n_test_states": int(test_mask.sum()),
    }


def _run_cell(
    *,
    args: argparse.Namespace,
    device: torch.device,
    samples: list[ProbeSample],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    match_regime: str,
    replication: int,
    out_dir: Path,
    training: dict[str, object],
    fixture_info: dict[str, object],
) -> Path:
    activations: dict[str, np.ndarray] = {}
    probe_report: dict[str, dict[str, float]] = {}
    for codec in DEFAULT_CODECS:
        run_dir = args.runs_root / match_regime / codec / f"rep{replication}"
        ckpt = run_dir / "checkpoints" / "final.pt"
        if not ckpt.is_file():
            print(f"skip {codec}: missing {ckpt}")
            continue
        vocab = load_codec_vocab(codec, args.vocab_dir)
        model, _payload = load_scoring_checkpoint(ckpt, device)
        features = np.stack(
            [episode_hidden(model, codec, s.episode, vocab, device) for s in samples]
        )
        activations[codec] = features
        probe_report[codec] = _probe_codec(features, samples, train_mask, test_mask)
        print(f"probe {match_regime}/rep{replication} {codec} states={features.shape[0]}")
    if not activations:
        raise SystemExit(f"no final checkpoints found for {match_regime}/rep{replication}")

    codecs = list(activations)
    similarity: dict[str, dict[str, dict[str, float]]] = {
        "cka": {},
        "svcca": {},
        "procrustes": {},
    }
    for left in codecs:
        for name in similarity:
            similarity[name][left] = {}
        for right in codecs:
            similarity["cka"][left][right] = linear_cka(activations[left], activations[right])
            similarity["svcca"][left][right] = svcca_mean_correlation(
                activations[left], activations[right]
            )
            similarity["procrustes"][left][right] = orthogonal_procrustes(
                activations[left], activations[right]
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "git_commit": training["git_commit"],
        "git_dirty": training["git_dirty"],
        "git_commit_source": training["git_commit_source"],
        "probe_git_commit": git_commit(),
        "probe_git_dirty": git_dirty(),
        "match_regime": match_regime,
        "replication": replication,
        "n_states": len(samples),
        "frozen": True,
        "probe_protocol": {
            "representation": "final_layer_hidden_at_last_token_of_framed_prefix",
            "ridge_alpha": PROBE_RIDGE_ALPHA,
            "feature_standardization": "train_rows_only",
            "split": "by_episode",
            "train_fraction": PROBE_TRAIN_FRACTION,
            "n_train_states": int(train_mask.sum()),
            "n_test_states": int(test_mask.sum()),
            "next_b_target": "state_{t+1}.b from prefix 0..t",
            "regime_probe": "eval.score._fit_regime_probe (ridge to one-hot)",
            **fixture_info,
        },
        "probes": probe_report,
        "similarity": similarity,
    }
    path = out_dir / "probes.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Single mode: directory for probes.json (default artifacts/probes). "
        "Matrix mode: probes root (default artifacts/probes/<budget-profile>).",
    )
    parser.add_argument("--vocab-dir", type=Path, default=_ROOT / "artifacts" / "vocabs")
    parser.add_argument("--n-states", type=int, default=10_000)
    parser.add_argument("--device", choices=("auto", "rocm", "cuda", "cpu"), default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--match-regime", default="compute_matched")
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Probe every (--match-regimes x replications) cell into out-dir/{regime}/rep{r}/.",
    )
    parser.add_argument(
        "--match-regimes",
        nargs="+",
        default=list(DEFAULT_MATCH_REGIMES),
        help="Matrix mode: match regimes to probe.",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=None,
        help="Matrix mode: number of replications (default: manifest n_replications).",
    )
    parser.add_argument(
        "--budget-profile",
        default=None,
        help="Optional; must equal the manifest budget_profile. Sets the matrix default out-dir.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    training = _training_provenance(args.runs_root)
    manifest_profile = training["budget_profile"]
    if (
        args.budget_profile is not None
        and manifest_profile is not None
        and args.budget_profile != manifest_profile
    ):
        raise SystemExit(
            f"--budget-profile {args.budget_profile!r} != manifest budget_profile "
            f"{manifest_profile!r}"
        )
    profile = args.budget_profile or manifest_profile

    if args.matrix:
        n_replications = args.replications
        if n_replications is None and training["n_replications"] is not None:
            n_replications = int(training["n_replications"])  # type: ignore[arg-type]
        if n_replications is None or n_replications < 1:
            raise SystemExit("--matrix needs --replications (no manifest n_replications found)")
        if args.out_dir is not None:
            out_root = args.out_dir
        elif profile is not None:
            out_root = _ROOT / "artifacts" / "probes" / str(profile)
        else:
            raise SystemExit("--matrix needs --out-dir or --budget-profile / manifest profile")
        cells = [
            (regime, rep, out_root / regime / f"rep{rep}")
            for regime in args.match_regimes
            for rep in range(n_replications)
        ]
    else:
        out_dir = args.out_dir if args.out_dir is not None else _ROOT / "artifacts" / "probes"
        cells = [(args.match_regime, args.replication, out_dir)]

    device = resolve_device(args.device, require_gpu=args.require_gpu)
    episodes = _states_from_raw(args.raw_dir, args.n_states)
    samples = _flatten_states(episodes, args.n_states)
    if len(samples) < 2:
        raise SystemExit("probe fixture needs at least 2 states")
    try:
        train_mask, test_mask = split_by_group([s.episode_index for s in samples])
    except ValueError as exc:
        raise SystemExit(f"probe fixture too small: {exc}") from exc
    val_path = args.raw_dir / "val.h5"
    fixture_info: dict[str, object] = {
        "fixture_source": "val.h5" if val_path.is_file() else "simulator_fallback",
        "fixture_val_h5_sha256": file_sha256(val_path),
        "fixture_episodes_sha256": episodes_sha256(episodes),
        "fixture_n_episodes": len(episodes),
        "fixture_fallback_seed_base": FALLBACK_SEED_BASE,
    }
    for regime, replication, out_dir in cells:
        _run_cell(
            args=args,
            device=device,
            samples=samples,
            train_mask=train_mask,
            test_mask=test_mask,
            match_regime=regime,
            replication=replication,
            out_dir=out_dir,
            training=training,
            fixture_info=fixture_info,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
