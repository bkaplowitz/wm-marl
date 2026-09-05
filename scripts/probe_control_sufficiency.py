#!/usr/bin/env python3
"""Probe frozen local representations using episode-held-out ridge regression.

Reads an existing checkpoint, config, and replay; never creates an environment,
optimizer, actor update, or checkpoint. Complete replay episodes are re-encoded
causally from their reset with current frozen weights (full episode burn-in),
without using the stale latent entries saved during collection. Targets are
real discounted team returns and current observable coordinates/availability.

Example (run only on an explicitly assigned device):
  CUDA_VISIBLE_DEVICES=0 python scripts/probe_control_sufficiency.py \
      --run /workspace/majepa_control_probe_20260904 --platform cuda \
      --output /workspace/majepa_control_probe_20260904/probe.json

Linear decodability is a diagnostic, not proof of representation sufficiency.
Return prediction includes behavior-policy and partial-observability uncertainty.
Raw observations are an identity-decoding upper bound for observable targets.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np


@dataclass
class Episode:
    identity: str
    source: str
    start: int
    data: dict[str, np.ndarray]


def complete_episode_slices(first, last):
    """Discard incomplete chunk prefixes/tails and never cross a reset."""
    first, last = np.asarray(first, bool), np.asarray(last, bool)
    if first.ndim != 1 or last.shape != first.shape:
        raise ValueError("Episode boundaries must be synchronized team vectors [T]")
    start = None
    for index in range(len(first)):
        if first[index]:
            start = index
        if last[index] and start is not None:
            if index > start:
                yield start, index + 1
            start = None


def short_returns(reward, terminal, horizons, gamma):
    """At observation t predict rewards t+1:t+H, with zero terminal tails."""
    reward = np.asarray(reward, np.float64)
    terminal = np.asarray(terminal, bool)
    if reward.ndim != 1 or terminal.shape != reward.shape:
        raise ValueError("Returns need one team reward and terminal flag per state")
    result = np.full((len(reward), len(horizons)), np.nan, np.float32)
    for root in range(len(reward)):
        for column, horizon in enumerate(horizons):
            total, ended = 0.0, bool(terminal[root])
            for offset in range(1, horizon + 1):
                if ended:
                    break
                target = root + offset
                if target >= len(reward):
                    total = np.nan  # Unknown tail of a nonterminal truncation.
                    break
                total += gamma ** (offset - 1) * reward[target]
                ended = bool(terminal[target])
            result[root, column] = total
    return result


def split_episodes(identities, seed):
    """Assign whole episodes to 60/20/20 train/validation/test partitions."""
    unique = np.unique(identities)
    if len(unique) < 6:
        raise ValueError("Need at least six complete episodes for separated splits")
    unique = np.random.default_rng(seed).permutation(unique)
    train_end = max(2, int(0.6 * len(unique)))
    validation_end = min(len(unique) - 1, max(train_end + 1, int(0.8 * len(unique))))
    groups = (
        unique[:train_end],
        unique[train_end:validation_end],
        unique[validation_end:],
    )
    return tuple(np.isin(identities, group) for group in groups)


def ridge_probe(features, target, partitions, alphas, max_train_rows, seed):
    """Choose regularization per target on validation, then evaluate test once.

    Standardization and target centering use training data only. The dual solve
    handles the 10k-dimensional actor tensor without a 10k-square matrix; no
    random projection discards potentially control-relevant dimensions.
    """
    x = np.asarray(features, np.float64)
    y = np.asarray(target, np.float64)
    if y.ndim == 1:
        y = y[:, None]
    valid = np.isfinite(y).all(axis=-1) & np.isfinite(x).all(axis=-1)
    train, validation, test = [np.flatnonzero(mask & valid) for mask in partitions]
    if min(len(train), len(validation), len(test)) < 2:
        raise ValueError("Each probe split needs at least two valid samples")
    if len(train) > max_train_rows:
        train = np.sort(
            np.random.default_rng(seed).choice(train, max_train_rows, replace=False)
        )
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
    keep = scale > 1e-7
    x = (x[:, keep] - mean[keep]) / scale[keep]
    center = y[train].mean(axis=0)
    train_y = y[train] - center
    train_x = x[train]
    count = len(train)
    # The objective is mean squared error + alpha * ||weight||^2.
    if train_x.shape[1] <= count:
        gram = train_x.T @ train_x / count
        eigenvalues, vectors = np.linalg.eigh(gram)
        right = vectors.T @ train_x.T @ train_y / count
        validation_basis = x[validation] @ vectors
        test_basis = x[test] @ vectors
    else:
        gram = train_x @ train_x.T / count
        eigenvalues, vectors = np.linalg.eigh(gram)
        right = vectors.T @ train_y
        validation_basis = x[validation] @ train_x.T @ vectors / count
        test_basis = x[test] @ train_x.T @ vectors / count
    eigenvalues = np.maximum(eigenvalues, 0.0)
    best_error = np.full(y.shape[1], np.inf)
    chosen = np.zeros(y.shape[1], np.float64)
    prediction = np.zeros((len(test), y.shape[1]), np.float64)
    for alpha in alphas:
        weight = right / (eigenvalues[:, None] + alpha)
        validation_prediction = validation_basis @ weight + center
        error = np.square(validation_prediction - y[validation]).mean(axis=0)
        improve = error < best_error
        best_error = np.where(improve, error, best_error)
        chosen = np.where(improve, alpha, chosen)
        prediction[:, improve] = (test_basis @ weight + center)[:, improve]
    squared = np.square(prediction - y[test])
    baseline = np.square(center - y[test]).mean(axis=0)
    variance = np.var(y[test], axis=0)
    return {
        "input_dimensions": int(features.shape[-1]),
        "nonconstant_dimensions": int(keep.sum()),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "alpha": chosen.tolist(),
        "rmse": np.sqrt(squared.mean(axis=0)).tolist(),
        "mae": np.abs(prediction - y[test]).mean(axis=0).tolist(),
        "r2": [
            float(1.0 - err / var) if var > 1e-12 else None
            for err, var in zip(squared.mean(axis=0), variance, strict=True)
        ],
        "mean_baseline_rmse": np.sqrt(baseline).tolist(),
        "skill_vs_train_mean": [
            float(1.0 - err / base) if base > 1e-12 else None
            for err, base in zip(squared.mean(axis=0), baseline, strict=True)
        ],
        "validation_rmse": np.sqrt(best_error).tolist(),
    }


def _team_flag(value):
    value = np.asarray(value, bool)
    if value.ndim == 2:
        if not np.all(value == value[:, :1]):
            raise ValueError("Replay has unsynchronized team episode boundaries")
        value = value[:, 0]
    return value


def load_episodes(directory, max_chunks, max_episodes, seed):
    required = (
        "observation",
        "action",
        "action_mask",
        "reward",
        "is_first",
        "is_last",
        "is_terminal",
    )
    files = sorted(directory.glob("*.npz"))[-max_chunks:]
    if not files:
        raise FileNotFoundError(f"No replay NPZ files in {directory}")
    found, seen = [], set()
    source_files = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            missing = set(required) - set(archive.files)
            if missing:
                raise ValueError(
                    f"{path} misses required replay fields {sorted(missing)}"
                )
            keys = (
                *required,
                *(
                    key
                    for key in (
                        "agent_present",
                        "agent_alive",
                        "controllable_alive",
                        "stepid",
                    )
                    if key in archive
                ),
            )
            data = {key: archive[key] for key in keys}
        if data["observation"].ndim != 3:
            raise ValueError("Only vector SMAC replay [T,A,O] is supported")
        for key in ("is_first", "is_last", "is_terminal"):
            data[key] = _team_flag(data[key])
        for start, end in complete_episode_slices(data["is_first"], data["is_last"]):
            identity = (
                np.asarray(data["stepid"][start]).tobytes().hex()
                if "stepid" in data
                else f"{path.name}:{start}"
            )
            if identity in seen:
                continue
            seen.add(identity)
            found.append(
                Episode(
                    identity,
                    path.name,
                    start,
                    {
                        key: value[start:end]
                        for key, value in data.items()
                        if key != "stepid"
                    },
                )
            )
        source_files.append({"path": str(path), "bytes": path.stat().st_size})
    if len(found) < 6:
        raise ValueError(
            f"Only {len(found)} complete unique episodes found; supply more chunks"
        )
    selected = np.random.default_rng(seed).permutation(len(found))[:max_episodes]
    episodes = [found[int(index)] for index in selected]
    return episodes, {
        "files": source_files,
        "complete_episodes_available": len(found),
        "partial_chunk_episodes": "discarded, never spliced or used across reset",
    }


def make_extractor(config, observation_dim, action_count):
    """Instantiate only checkpoint-compatible encoder/local dynamics modules."""
    import elements
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from majepa.models.visual import Encoder
    from majepa.world_model.transformer import ParallelTransformerDynamics

    # Elements restores tuple-valued module fields from YAML lists exactly as
    # the training entry point does (for example Encoder.mults).
    config = elements.Config(config)
    encoder = Encoder(
        {"observation": elements.Space(np.float32, (observation_dim,))},
        **config["agent"]["enc"]["simple"],
        name="enc",
    )
    dynamics = ParallelTransformerDynamics(
        {"action": elements.Space(np.int32, (), 0, action_count)},
        encoder.calculate_encoder_output_dim(),
        **config["agent"]["dyn"]["parallel_transformer"],
        name="dyn",
    )

    def forward(observation, previous_action, first, active):
        agents = observation.shape[0]
        _, _, tokens = encoder({}, {"observation": observation}, first, training=False)
        _, _, features, posterior = dynamics.observe(
            dynamics.initial(agents),
            tokens,
            {"action": previous_action},
            first,
            training=False,
            active=active,
        )
        probability = jax.nn.softmax(posterior.astype(jnp.float32), axis=-1)
        probability = (
            1.0 - dynamics.unimix
        ) * probability + dynamics.unimix / probability.shape[-1]
        return {
            "encoder": tokens.astype(jnp.float32),
            "posterior_sample": jnp.concatenate(
                [
                    features["deter"].astype(jnp.float32),
                    features["stoch"].reshape((*first.shape, -1)).astype(jnp.float32),
                ],
                axis=-1,
            ),
            "posterior_mean": jnp.concatenate(
                [
                    features["deter"].astype(jnp.float32),
                    probability.reshape((*first.shape, -1)),
                ],
                axis=-1,
            ),
        }

    return nj.pure(forward), nj.init(forward)


def load_frozen_params(checkpoint, initialize, example):
    import jax

    if checkpoint.is_dir():
        if not (checkpoint / "done").is_file():
            raise ValueError(f"Checkpoint is not completed: {checkpoint}")
        checkpoint = checkpoint / "agent.pkl"
    with checkpoint.open("rb") as stream:
        payload = pickle.load(stream)
    loaded = payload["params"]
    expected = jax.eval_shape(lambda *args: initialize({}, *args, seed=0), *example)
    missing = sorted(set(expected) - set(loaded))
    mismatched = [
        key
        for key in expected
        if key in loaded and expected[key].shape != loaded[key].shape
    ]
    if missing or mismatched:
        raise ValueError(
            f"Incompatible frozen local checkpoint: missing={missing[:6]}, shape={mismatched[:6]}"
        )
    params = jax.device_put({key: np.asarray(loaded[key]) for key in expected})
    return params, {
        "file": str(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "local_parameters": len(params),
        "counters": payload.get("counters", {}),
    }


def extract_samples(
    episodes, config, checkpoint, samples_per_episode, seed, horizons, gamma
):
    import jax
    import jax.numpy as jnp

    agents, observation_dim = episodes[0].data["observation"].shape[1:]
    action_count = episodes[0].data["action_mask"].shape[-1]
    if agents != int(config["agent"]["num_agents"]):
        raise ValueError("Replay roster disagrees with checkpoint config")
    maximum = max(len(episode.data["observation"]) for episode in episodes)
    padded_length = 64 * ((maximum + 63) // 64)
    forward, initialize = make_extractor(config, observation_dim, action_count)
    example = (
        jnp.zeros((agents, 1, observation_dim), jnp.float32),
        jnp.zeros((agents, 1), jnp.int32),
        jnp.ones((agents, 1), bool),
        jnp.ones((agents, 1), bool),
    )
    params, provenance = load_frozen_params(checkpoint, initialize, example)
    extract = jax.jit(forward)
    rng = np.random.default_rng(seed)
    collected = []
    for index, episode in enumerate(episodes):
        data = episode.data
        length = len(data["observation"])
        rewards = data["reward"].astype(np.float32)
        if rewards.ndim != 2 or not np.allclose(rewards, rewards[:, :1]):
            raise ValueError("Probe assumes broadcast shared SMAC team rewards")
        returns = short_returns(
            rewards.mean(axis=-1), data["is_terminal"], horizons, gamma
        )
        # Only decision states are probed, with all agents of a sampled timestep
        # remaining in the same episode partition. After-death team rewards
        # still belong to the return targets of preceding decisions.
        alive = data.get("controllable_alive", data["action_mask"][..., 1:].any(-1))
        candidates = np.flatnonzero(
            ~data["is_last"] & np.isfinite(returns).all(-1) & alive.any(-1)
        )
        chosen = np.sort(
            rng.choice(
                candidates, min(samples_per_episode, len(candidates)), replace=False
            )
        )
        if not len(chosen):
            continue

        def pad(value, fill=0):
            return np.pad(
                value,
                ((0, padded_length - length), *((0, 0) for _ in value.shape[1:])),
                constant_values=fill,
            )

        observation = pad(data["observation"]).transpose(1, 0, 2)
        action = np.concatenate(
            [np.zeros_like(data["action"][:1]), data["action"][:-1]], axis=0
        )
        previous_action = pad(action).T
        first = np.broadcast_to(
            pad(data["is_first"], True)[None], (agents, padded_length)
        )
        active = pad(data.get("agent_alive", np.ones((length, agents), bool))).T.astype(
            bool
        )
        if "agent_present" in data:
            active &= pad(data["agent_present"]).T.astype(bool)
        inputs = jax.device_put((observation, previous_action, first, active))
        _, outputs = extract(params, *inputs, seed=seed + index)
        outputs = jax.device_get(outputs)
        decision = alive[chosen].reshape(-1).astype(bool)

        def selected(value):
            return (
                np.asarray(value)[:, chosen]
                .transpose(1, 0, 2)
                .reshape(len(chosen) * agents, -1)[decision]
            )

        item = {name: selected(value) for name, value in outputs.items()}
        item.update(
            {
                "raw": data["observation"][chosen].reshape(-1, observation_dim)[
                    decision
                ],
                "observable": data["observation"][chosen].reshape(-1, observation_dim)[
                    decision
                ],
                "availability": data["action_mask"][chosen]
                .reshape(-1, action_count)[decision]
                .astype(np.float32),
                "return": np.repeat(returns[chosen], agents, axis=0)[decision],
                "episode": np.full(int(decision.sum()), episode.identity),
            }
        )
        collected.append(item)
        print(
            f"Extracted episode {index + 1}/{len(episodes)}: {length} states, {decision.sum()} local decisions",
            flush=True,
        )
    if not collected:
        raise ValueError("No valid decision samples found")
    return {
        name: np.concatenate([item[name] for item in collected])
        for name in collected[0]
    }, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--external", type=Path)
    parser.add_argument("--max-chunks", type=int, default=64)
    parser.add_argument("--max-episodes", type=int, default=96)
    parser.add_argument("--samples-per-episode", type=int, default=8)
    parser.add_argument("--max-train-rows", type=int, default=2048)
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 15])
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    )
    args = parser.parse_args(argv)
    if (
        min(
            args.max_chunks,
            args.max_episodes,
            args.samples_per_episode,
            args.max_train_rows,
            *args.horizons,
        )
        < 1
    ):
        parser.error("Sample limits and horizons must be positive")
    if min(args.alphas) <= 0:
        parser.error("Ridge penalties must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output {args.output}")
    run = args.run.expanduser().resolve()
    replay = (args.replay or run / "replay").expanduser().resolve()
    checkpoint = (
        args.checkpoint or run / "ckpt" / (run / "ckpt/latest").read_text().strip()
    )
    checkpoint = checkpoint.expanduser().resolve()
    # Load/validate replay before allocating any device or checkpoint parameters.
    episodes, replay_provenance = load_episodes(
        replay, args.max_chunks, args.max_episodes, args.seed
    )
    from ruamel.yaml import YAML

    config_path = run / "config.yaml"
    config_text = config_path.read_text()
    config = YAML(typ="safe").load(config_text)
    gamma = (
        args.gamma
        if args.gamma is not None
        else 1.0 - 1.0 / float(config["agent"]["horizon"])
    )
    if not 0 < gamma <= 1:
        parser.error("Gamma must be in (0,1]")
    repository = Path(__file__).resolve().parents[1]
    external = args.external or repository / "external/dreamerv3"
    sys.path[:0] = [str(repository / "src"), str(external)]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import embodied.jax.nets as nets

    jax.config.update("jax_platforms", args.platform)
    nets.COMPUTE_DTYPE = getattr(jnp, config["jax"].get("compute_dtype", "bfloat16"))
    samples, checkpoint_provenance = extract_samples(
        episodes,
        config,
        checkpoint,
        args.samples_per_episode,
        args.seed,
        args.horizons,
        gamma,
    )
    partitions = split_episodes(samples["episode"], args.seed)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run),
        "task": config["task"],
        "checkpoint": checkpoint_provenance,
        "replay": replay_provenance,
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "gamma": gamma,
        "return_horizons": args.horizons,
        "seed": args.seed,
        "split": {
            name: {
                "episodes": np.unique(samples["episode"][mask]).tolist(),
                "rows": int(mask.sum()),
            }
            for name, mask in zip(
                ("train", "validation", "test"), partitions, strict=True
            )
        },
        "feature_protocol": "Fresh frozen encoder and causal local posterior from raw complete episodes; no saved latent entries, future inputs, optimizer, or actor updates. Posterior mean shares the sampled causal history; only current categorical posterior is averaged.",
        "limits": "Held-out linear decodability, not proof of control sufficiency. Returns depend on the replay behavior policy. Observable indices are not semantic health/cooldown labels without a recorded SMAC observation schema. Partial chunk episodes are discarded.",
        "probes": {},
    }
    target_names = ("return", "observable", "availability")
    all_targets = np.concatenate([samples[name] for name in target_names], axis=-1)
    for feature in ("raw", "encoder", "posterior_sample", "posterior_mean"):
        result["probes"][feature] = {}
        print(f"Fitting {feature} -> returns, observables, availability", flush=True)
        combined = ridge_probe(
            samples[feature],
            all_targets,
            partitions,
            args.alphas,
            args.max_train_rows,
            args.seed,
        )
        offset = 0
        for target in target_names:
            width = samples[target].shape[-1]
            result["probes"][feature][target] = {
                key: value[offset : offset + width]
                if isinstance(value, list)
                else value
                for key, value in combined.items()
            }
            offset += width
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
