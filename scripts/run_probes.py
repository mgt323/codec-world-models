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

Additive diagnostics (Tier B2, recording only; the locked keys above keep their
definitions and values):

* ``probes[codec].hidden_c_r2_common_cause``: the same ``PROBE_RIDGE_ALPHA`` ridge
  fit and scored on ``COMMON_CAUSE`` rows only (``hidden_c`` is irrelevant to the
  observables in the other three regimes, which dilutes the pooled score).
* ``fixture_baselines``: codec-independent held-out ``R^2`` of trivial and oracle
  predictors of ``state_{t+1}.b`` on exactly the ``next_b_r2`` test rows
  (``persistence_next_b_r2``, ``running_mean_next_b_r2``, ``oracle_next_b_r2``).
* ``untrained_probes[codec]``: the three locked probes plus the common-cause score
  on the *untrained* initialization of the same run (same ``model_init_seed`` and
  codec), giving each trained score an initialization floor. Disable with
  ``--skip-untrained``.

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
from model.architecture import build_model  # noqa: E402
from model.init import initialize_model  # noqa: E402
from probes.baselines import (  # noqa: E402
    last_observed_b,
    oracle_next_b_mean,
    prefix_mean_observed_b,
)
from probes.linear import (  # noqa: E402
    PROBE_RIDGE_ALPHA,
    PROBE_TRAIN_FRACTION,
    heldout_ridge_r2,
    r2_score,
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
_COMMON_CAUSE_INDEX = _REGIMES.index(CausalRegime.COMMON_CAUSE)
# Minimum rows per split for the common-cause subset ridge fit / held-out score.
_MIN_SUBSET_ROWS = 2
ADDITIVE_DIAGNOSTICS_VERSION = "probe_diagnostics_v1"
# Baseline prediction for prefixes in which no ``b`` has been observed yet.
NO_OBSERVED_B_FALLBACK = 0.0


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
) -> dict[str, float | int | None]:
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
    # Additive (Tier B2): same ridge, same split, restricted to COMMON_CAUSE rows.
    common_cause = labels == _COMMON_CAUSE_INDEX
    train_cc = train_mask & common_cause
    test_cc = test_mask & common_cause
    hidden_c_r2_common_cause: float | None = None
    if int(train_cc.sum()) >= _MIN_SUBSET_ROWS and int(test_cc.sum()) >= _MIN_SUBSET_ROWS:
        hidden_c_r2_common_cause = heldout_ridge_r2(
            features[train_cc], hidden_c[train_cc], features[test_cc], hidden_c[test_cc]
        )
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
        "hidden_c_r2_common_cause": hidden_c_r2_common_cause,
        "n_common_cause_train_states": int(train_cc.sum()),
        "n_common_cause_test_states": int(test_cc.sum()),
        "n_states": int(features.shape[0]),
        "n_train_states": int(train_mask.sum()),
        "n_test_states": int(test_mask.sum()),
    }


def _fixture_baselines(
    samples: list[ProbeSample],
    test_mask: np.ndarray,
) -> dict[str, float | int | None]:
    """Codec-independent next-``b`` reference scores on the ``next_b_r2`` test rows.

    ``R^2`` is taken against the test-row mean, exactly as for ``next_b_r2``.
    Prefixes with no observed ``b`` yet predict the constant
    ``NO_OBSERVED_B_FALLBACK`` (no train or test target enters a prediction).
    """
    has_next = np.asarray([s.next_b is not None for s in samples], dtype=bool)
    test_next = test_mask & has_next
    if int(test_next.sum()) < _MIN_SUBSET_ROWS:
        return {
            "persistence_next_b_r2": None,
            "running_mean_next_b_r2": None,
            "oracle_next_b_r2": None,
            "n_test_next_states": int(test_next.sum()),
        }
    next_b = np.asarray(
        [0.0 if s.next_b is None else s.next_b for s in samples], dtype=np.float64
    )
    persistence = np.asarray(
        [last_observed_b(s.prefix, fallback=NO_OBSERVED_B_FALLBACK) for s in samples],
        dtype=np.float64,
    )
    running_mean = np.asarray(
        [prefix_mean_observed_b(s.prefix, fallback=NO_OBSERVED_B_FALLBACK) for s in samples],
        dtype=np.float64,
    )
    oracle = np.asarray(
        [
            oracle_next_b_mean(_REGIMES[s.regime_index], s.hidden_c)
            for s in samples
        ],
        dtype=np.float64,
    )
    target = next_b[test_next]
    return {
        "persistence_next_b_r2": r2_score(target, persistence[test_next]),
        "running_mean_next_b_r2": r2_score(target, running_mean[test_next]),
        "oracle_next_b_r2": r2_score(target, oracle[test_next]),
        "n_test_next_states": int(test_next.sum()),
        "no_observed_b_fallback": NO_OBSERVED_B_FALLBACK,
    }


def _untrained_model(run_dir: Path, codec: str, device: torch.device):
    """Frozen untrained initialization of ``run_dir``'s run (same seed and codec)."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model = build_model()
    initialize_model(model, model_init_seed=int(config["model_init_seed"]), codec_name=codec)
    model.to(device)
    model.float()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


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
    fixture_baselines: dict[str, float | int | None],
) -> Path:
    activations: dict[str, np.ndarray] = {}
    probe_report: dict[str, dict[str, float | int | None]] = {}
    untrained_report: dict[str, dict[str, float | int | None]] = {}
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
        if not args.skip_untrained:
            untrained = _untrained_model(run_dir, codec, device)
            untrained_features = np.stack(
                [episode_hidden(untrained, codec, s.episode, vocab, device) for s in samples]
            )
            untrained_report[codec] = _probe_codec(
                untrained_features, samples, train_mask, test_mask
            )
            print(f"probe-untrained {match_regime}/rep{replication} {codec}")
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
            "additive_diagnostics": {
                "version": ADDITIVE_DIAGNOSTICS_VERSION,
                "locked_keys_unchanged": ["hidden_c_r2", "regime_accuracy", "next_b_r2"],
                "added": [
                    "probes[codec].hidden_c_r2_common_cause",
                    "fixture_baselines",
                    *([] if args.skip_untrained else ["untrained_probes[codec]"]),
                ],
                "kind": "recording_change_no_methodology_change",
            },
            **fixture_info,
        },
        "probes": probe_report,
        "fixture_baselines": fixture_baselines,
        **({} if args.skip_untrained else {"untrained_probes": untrained_report}),
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
    parser.add_argument(
        "--skip-untrained",
        action="store_true",
        help="Skip the untrained-initialization probe scores (additive diagnostic).",
    )
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
    fixture_baselines = _fixture_baselines(samples, test_mask)
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
            fixture_baselines=fixture_baselines,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
