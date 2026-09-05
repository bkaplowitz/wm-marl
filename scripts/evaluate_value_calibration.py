#!/usr/bin/env python3
"""Pair frozen central values with fresh real, decentralized-policy SMAC returns.

Run in the same maintained Python/SMAC runtime as training, on an assigned device:
  python scripts/evaluate_value_calibration.py --config RUN/config.yaml \
      --checkpoint RUN/ckpt/CHECKPOINT --output NEW_DIRECTORY --episodes 32

No training, reporting, optimizer step, or parameter synchronization occurs after
checkpoint loading. The canonical runtime and Driver execute the actor; a separate
read-only critic call consumes its returned local state *after* action selection.
The default eval_sample policy samples the learned policy without collection
unimix. Greedy eval is a separate, explicitly labeled protocol.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


RAW_FIELDS = (
    "observation",
    "reward",
    "action",
    "action_mask",
    "agent_present",
    "agent_alive",
    "controllable_alive",
    "is_first",
    "is_last",
    "is_terminal",
)
VALUE_FIELDS = ("calibration/live", "calibration/slow")


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_provenance(root):
    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    files = sorted((root / "src" / "majepa").rglob("*.py"))
    files += [root / "src" / "majepa" / "configs.yaml", Path(__file__).resolve()]
    hashes = {str(path.relative_to(root)): digest_file(path) for path in files}
    return {
        "root": str(root),
        "commit": git("rev-parse", "HEAD"),
        "status": git("status", "--short"),
        "files_sha256": hashes,
    }


def runtime_provenance(embodied):
    root = Path(embodied.__file__).resolve().parent
    files = (
        "__init__.py",
        "jax/agent.py",
        "jax/internal.py",
        "jax/utils.py",
        "jax/transform.py",
        "core/driver.py",
        "core/wrappers.py",
    )
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "embodied_root": str(root),
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "files_sha256": {name: digest_file(root / name) for name in files},
    }


def parameter_digest(agent):
    """Hash every loaded tensor, including optimizer state; called only twice."""
    import jax

    digest = hashlib.sha256()
    for key, value in sorted(agent.params.items()):
        array = np.asarray(jax.device_get(value))
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def counters(agent):
    return {
        name: int(getattr(agent, name).value)
        for name in ("n_updates", "n_batches", "n_actions")
    }


def critic_forward(model, features, present, alive):
    """Input [B,A,...] is the exact online posterior, never a teacher re-encode."""
    local = {
        key: model.team.fold_sequence(value[:, None]) for key, value in features.items()
    }
    context = {
        "present": present[:, None],
        "controllable_alive": alive[:, None],
    }
    return {
        name: model.team.unfold_sequence(
            model.critic(local, 2, slow=slow, context=context).pred()
        )[:, 0]
        for name, slow in (("live", False), ("slow", True))
    }


class FrozenCentralCritic:
    """Canonical central heads with creation/writes forbidden and separate RNG."""

    def __init__(self, agent, seed):
        import jax
        import jax.numpy as jnp
        import ninjax as nj

        if not agent.model.ctde_enabled:
            raise ValueError("Calibration requires the maintained central critic")
        self.agent = agent
        self.seed = int(seed)
        self.calls = 0
        # SlowModel reads its source as well as the slow model, but neither its
        # counter nor any optimizer/actor/world weights are needed here.
        self.params = {
            key: value
            for key, value in agent.params.items()
            if key.startswith(("ctde_val/", "slowctde_val/"))
        }
        for prefix in ("ctde_val/", "slowctde_val/"):
            if not any(key.startswith(prefix) for key in self.params):
                raise ValueError(f"Checkpoint lacks {prefix} parameters")
        pure = nj.pure(partial(critic_forward, agent.model))

        def predict(params, features, present, alive, seed):
            _, values = pure(
                params,
                features,
                present,
                alive,
                seed=seed,
                create=False,
                modify=False,
            )
            return values

        self.predict = jax.jit(predict)
        # Outer Agent returns each batch axis split into a list of device arrays.
        self.stack = jax.jit(
            lambda xs: jax.tree.map(
                jnp.stack, xs, is_leaf=lambda value: isinstance(value, list)
            )
        )

    def __call__(self, carry, observation):
        import jax

        dynamic = carry[1]
        features = self.stack({key: dynamic[key] for key in ("deter", "stoch")})
        rng = np.random.default_rng([self.seed, self.calls])
        seed = rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)
        values = self.predict(
            self.params,
            features,
            jax.device_put(observation["agent_present"]),
            jax.device_put(observation["controllable_alive"]),
            jax.device_put(seed),
        )
        self.calls += 1
        return {
            key: np.asarray(value, np.float32)
            for key, value in jax.device_get(values).items()
        }


def instrument_policy(agent, diagnostic, policy_mode):
    """Add outputs only after the unchanged actor call and verify RNG isolation."""

    def policy(carry, observation):
        carry, actions, outputs = agent.policy(carry, observation, mode=policy_mode)
        count = int(agent.n_actions.value)
        values = diagnostic(carry, observation)
        if int(agent.n_actions.value) != count:
            raise RuntimeError("The critic diagnostic modified the actor RNG counter")
        expected = observation["agent_present"].shape
        if set(values) != {"live", "slow"}:
            raise ValueError(
                "Both live and slow frozen critic predictions are required"
            )
        for name, value in values.items():
            if np.shape(value) != expected or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name} central values: expected {expected}")
        return (
            carry,
            actions,
            {
                **outputs,
                **{f"calibration/{key}": value for key, value in values.items()},
            },
        )

    return policy


def validate_episode(data):
    first, last, terminal = (
        np.asarray(data[key], bool) for key in ("is_first", "is_last", "is_terminal")
    )
    if first.ndim != 1 or last.shape != first.shape or terminal.shape != first.shape:
        raise ValueError("Episode flags must be synchronized team vectors [T]")
    if (
        len(first) < 2
        or not first[0]
        or first[1:].any()
        or not last[-1]
        or last[:-1].any()
        or terminal[:-1].any()
    ):
        raise ValueError(
            "Expected one complete episode with no internal reset/terminal"
        )
    reward = np.asarray(data["reward"], np.float64)
    if (
        reward.ndim != 2
        or len(reward) != len(first)
        or not np.isfinite(reward).all()
        or not np.allclose(reward, reward[:, :1], rtol=0, atol=1e-6)
    ):
        raise ValueError("Expected finite broadcast SMAC team rewards [T,A]")
    if not np.allclose(reward[0], 0.0):
        raise ValueError("Reset reward must be zero")
    return reward[:, 0], bool(terminal[-1])


def collect_episodes(
    agent,
    make_env,
    diagnostic,
    *,
    episodes,
    envs=1,
    worker_offset=100_000,
    policy_mode="eval_sample",
    max_driver_steps=1_000_000,
    environment_records=None,
):
    """Use the same Driver/canonical actor as maintained evaluation, fixed quotas."""
    import embodied
    from majepa.evaluation import _preserve_policy_state

    if episodes < 1 or envs < 1 or envs > episodes:
        raise ValueError("Require episodes >= envs >= 1")
    if policy_mode not in {"eval", "eval_sample"}:
        raise ValueError("Use eval or eval_sample")
    quota = [episodes // envs + int(index < episodes % envs) for index in range(envs)]
    completed = [0] * envs
    pending = [[] for _ in range(envs)]
    records = []

    def on_step(transition, worker):
        if completed[worker] >= quota[worker]:
            return
        if bool(transition["is_first"]):
            if pending[worker]:
                raise ValueError(
                    "Environment reset before completing the previous episode"
                )
        elif not pending[worker]:
            raise ValueError("Environment episode did not begin with is_first")
        pending[worker].append(
            {
                key: np.array(value, copy=True)
                for key, value in transition.items()
                if key in (*RAW_FIELDS, *VALUE_FIELDS) or key.startswith("log/")
            }
        )
        if bool(transition["is_last"]):
            rows = pending[worker]
            data = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
            validate_episode(data)
            records.append(
                {
                    "worker": worker,
                    "worker_index": worker_offset + worker,
                    "worker_episode": completed[worker],
                    "data": data,
                }
            )
            completed[worker] += 1
            pending[worker] = []

    # Serial stepping in the maintained Driver fixes batch and callback ordering.
    driver = embodied.Driver(
        [partial(make_env, worker_offset + worker) for worker in range(envs)],
        parallel=False,
    )
    driver.on_step(on_step)
    try:
        with _preserve_policy_state(agent):
            driver.reset(agent.init_policy)
            policy = instrument_policy(agent, diagnostic, policy_mode)
            for _ in range(max_driver_steps):
                if sum(completed) == episodes:
                    break
                driver(policy, steps=envs)
            if sum(completed) != episodes:
                raise RuntimeError(
                    f"Step budget exhausted: {sum(completed)}/{episodes} complete episodes"
                )
        if environment_records is not None:
            environment_records.extend(
                environment_provenance(env) for env in driver.envs
            )
    finally:
        driver.close()
    return records


def episode_targets(data, gamma, horizons=(5, 15)):
    """V(s_t) predicts r[t+1] onward; never truncate at focal unit death."""
    reward, terminated = validate_episode(data)
    length, agents = data["agent_present"].shape
    target = np.full(length, np.nan, np.float64)
    if terminated:
        target[-1] = 0.0
        for step in range(length - 2, -1, -1):
            target[step] = reward[step + 1] + gamma * target[step + 1]
    # A timeout labeled nonterminal is not an observed full Monte Carlo return.
    # Current SMAC adapter labels every episode end terminal; record that fact.
    result = {"monte_carlo": np.broadcast_to(target[:, None], (length, agents))}
    for horizon in horizons:
        if int(horizon) != horizon or horizon < 1:
            raise ValueError("Calibration horizons must be positive integers")
        for name in ("live", "slow"):
            value = np.asarray(data[f"calibration/{name}"], np.float64)
            if value.shape != (length, agents) or not np.isfinite(value).all():
                raise ValueError("Value and roster shapes must match and be finite")
            bootstrapped = np.full_like(value, np.nan)
            observed = np.full(length, np.nan, np.float64)
            for step in range(length - 1):
                endpoint = min(step + horizon, length - 1)
                if endpoint - step < horizon and not terminated:
                    continue
                total = sum(
                    gamma ** (offset - 1) * reward[step + offset]
                    for offset in range(1, endpoint - step + 1)
                )
                terminal_endpoint = endpoint == length - 1 and terminated
                if endpoint == length - 1 and not terminated:
                    # Truncated endpoint has an observed state and can bootstrap
                    # only when all H rewards have actually been observed.
                    assert endpoint - step == horizon
                tail = 0.0 if terminal_endpoint else gamma**horizon * value[endpoint]
                bootstrapped[step] = total + tail
                observed[step] = total
            result[f"h{horizon}/{name}_bootstrap"] = bootstrapped
            result[f"h{horizon}/observed_reward_sum"] = observed
    return result


def calibration_statistics(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    valid = np.isfinite(prediction) & np.isfinite(target)
    prediction, target = prediction[valid], target[valid]
    if not len(target):
        return {"count": 0}
    error = prediction - target
    variance = float(target.var())
    return {
        "count": len(target),
        "prediction_mean": float(prediction.mean()),
        "target_mean": float(target.mean()),
        "bias": float(error.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mae": float(np.abs(error).mean()),
        "explained_variance": 1 - float(error.var()) / variance
        if variance > 0
        else None,
        "correlation": float(np.corrcoef(prediction, target)[0, 1])
        if variance > 0 and float(prediction.var()) > 0
        else None,
    }


def summarize(records, *, gamma, sample_stride=1, horizons=(5, 15)):
    if not 0 < gamma <= 1 or sample_stride < 1:
        raise ValueError("Require 0 < gamma <= 1 and a positive sample stride")
    all_data, all_targets, sampled, episode_summaries = [], [], [], []
    for index, record in enumerate(records):
        data = record["data"]
        target = episode_targets(data, gamma, horizons)
        length = len(data["reward"])
        take = np.arange(length) % sample_stride == 0
        take[-1] = False
        # Retain the first post-death state even with sparse regular sampling.
        alive = np.asarray(data["controllable_alive"], bool)
        take[1:] |= (alive[:-1] & ~alive[1:]).any(-1)
        take[-1] = False
        sampled.append(take)
        all_data.append(data)
        all_targets.append(target)
        episode_summaries.append(
            {key: value for key, value in record.items() if key != "data"}
            | {
                "episode": index,
                "steps": length - 1,
                "return": float(data["reward"][1:, 0].sum()),
                "discounted_return": float(target["monte_carlo"][0, 0])
                if bool(data["is_terminal"][-1])
                else None,
                "terminated": bool(data["is_terminal"][-1]),
                "win": float(data.get("log/battle_won", np.zeros(length))[-1]),
                "timeout": float(data.get("log/timeout", np.zeros(length))[-1]),
            }
        )
    packed = {
        key: np.concatenate([data[key] for data in all_data]) for key in all_data[0]
    }
    targets = {
        key: np.concatenate([item[key] for item in all_targets])
        for key in all_targets[0]
    }
    sampled = np.concatenate(sampled)
    present = packed["agent_present"].astype(bool)
    alive = packed["controllable_alive"].astype(bool)
    offsets = np.cumsum([0] + [len(data["reward"]) for data in all_data])
    initial = np.zeros(len(present), bool)
    initial[offsets[:-1]] = True
    masks = {
        "all": sampled[:, None] & present,
        "alive": sampled[:, None] & present & alive,
        "dead": sampled[:, None] & present & ~alive,
        "episode_initial": initial[:, None] & present,
        "terminal": packed["is_terminal"][:, None] & present,
    }
    calibration = {}
    for name in ("live", "slow"):
        prediction = packed[f"calibration/{name}"]
        comparisons = {"monte_carlo": targets["monte_carlo"]}
        comparisons.update(
            {
                f"h{horizon}_bootstrap": targets[f"h{horizon}/{name}_bootstrap"]
                for horizon in horizons
            }
        )
        calibration[name] = {
            comparison: {
                slice_name: calibration_statistics(prediction[mask], target[mask])
                for slice_name, mask in masks.items()
            }
            for comparison, target in comparisons.items()
        }
    summary = {
        "episodes": len(records),
        "complete_terminal_episodes": sum(
            episode["terminated"] for episode in episode_summaries
        ),
        "state_count": len(present),
        "sampled_state_count": int(sampled.sum()),
        "return_mean": float(
            np.mean([episode["return"] for episode in episode_summaries])
        ),
        "win_rate": float(np.mean([episode["win"] for episode in episode_summaries])),
        "calibration": calibration,
        "episode_records": episode_summaries,
    }
    packed = {key.replace("/", "__"): value for key, value in packed.items()}
    packed.update(
        {
            "episode_offsets": offsets,
            "sampled_state": sampled,
            "team_reward": packed["reward"][:, 0],
            "monte_carlo_return": targets["monte_carlo"][:, 0].astype(np.float32),
        }
    )
    return summary, packed


def environment_provenance(env):
    """Record the running engine and SMAC implementation, not just a package name."""
    while "env" in vars(env):
        env = vars(env)["env"]
    sc2 = vars(env).get("_env")
    if sc2 is None:
        return {"adapter": f"{type(env).__module__}.{type(env).__name__}"}
    result = {
        "adapter": f"{type(env).__module__}.{type(env).__name__}",
        "environment_info": dict(sc2.get_env_info()),
        "settings": {
            key: vars(sc2)[key]
            for key in (
                "map_name",
                "difficulty",
                "continuing_episode",
                "_seed",
                "seed",
                "game_version",
                "step_mul",
                "episode_limit",
                "reward_scale",
                "reward_scale_rate",
                "reward_sparse",
                "reward_death_value",
                "reward_win",
                "reward_only_positive",
            )
            if key in vars(sc2) and isinstance(vars(sc2)[key], (str, int, float, bool))
        },
    }
    controller = vars(sc2).get("_controller")
    if controller is not None:
        try:
            from google.protobuf.json_format import MessageToDict

            result["engine_ping"] = MessageToDict(controller.ping())
        except Exception as error:
            result["engine_ping_error"] = f"{type(error).__name__}: {error}"
    module = sys.modules.get(type(sc2).__module__)
    path = Path(module.__file__).resolve() if module and module.__file__ else None
    if path:
        result["implementation"] = {"path": str(path), "sha256": digest_file(path)}
    root = (
        Path(os.environ["SC2PATH"]).expanduser() if os.environ.get("SC2PATH") else None
    )
    if root and root.is_dir():
        result["sc2_path"] = str(root)
        result["installed_versions"] = sorted(
            path.name for path in (root / "Versions").glob("*")
        )
        map_name = vars(sc2).get("map_name")
        if map_name:
            result["map_files"] = {
                str(path): digest_file(path)
                for path in (root / "Maps").rglob(f"{map_name}.SC2Map")
            }
    return result


def run(args):
    import elements
    import embodied
    import ruamel.yaml as yaml
    from majepa.main import make_agent, make_env
    from majepa.runtime import repository_root

    if args.episodes < 32:
        raise ValueError("Real calibration requires at least 32 complete episodes")
    if args.envs < 1 or args.envs > args.episodes or args.sample_stride < 1:
        raise ValueError("Invalid environment count or sample stride")
    if (
        min(args.eval_seed, args.worker_offset, args.diagnostic_seed) < 0
        or args.max_driver_steps < 1
    ):
        raise ValueError(
            "Seeds/worker offset must be nonnegative and step budget positive"
        )
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "done").is_file():
        raise ValueError(f"Checkpoint must be a completed directory: {checkpoint}")
    checkpoint_files = sorted(checkpoint.glob("agent*.pkl"))
    if not checkpoint_files:
        raise ValueError(f"Checkpoint has no agent payload: {checkpoint}")
    config_path = args.config.expanduser().resolve()
    config = elements.Config(yaml.YAML(typ="safe").load(config_path.read_text()))
    if not str(config.task).startswith("smac_") or not config.env.smac.use_seed:
        raise ValueError("Calibration requires seeded SMAC environments")
    output = args.output.expanduser().resolve()
    # No checkpoint selection or inferred current defaults: load exact saved config.
    overrides = {
        "logdir": str(output / "runtime"),
        "script": "eval_only",
        "seed": int(args.eval_seed),
        "jax.precompile": False,
        "jax.policy_devices": [0],
        "jax.train_devices": [0],
        "jax.expect_devices": 0,
        "jax.profiler": False,
        "run.debug": True,
    }
    if args.platform:
        overrides["jax.platform"] = args.platform
    evaluated = config.update(overrides)
    gamma = 1.0 - 1.0 / float(config.agent.horizon) if config.agent.contdisc else 1.0
    output.mkdir(parents=True, exist_ok=False)
    evaluated.save(elements.Path(output / "evaluation_config.yaml"))
    (output / "training_config.yaml").write_bytes(config_path.read_bytes())
    training_manifest = None
    if args.training_manifest:
        manifest_path = args.training_manifest.expanduser().resolve()
        training_manifest = {
            "path": str(manifest_path),
            "sha256": digest_file(manifest_path),
            "contents": json.loads(manifest_path.read_text()),
        }
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.executable,
        "checkpoint": str(checkpoint),
        "checkpoint_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": digest_file(path)}
            for path in checkpoint_files
        },
        "training_config": {
            "path": str(config_path),
            "sha256": digest_file(config_path),
        },
        "training_manifest": training_manifest,
        "training_source_note": "Only a supplied training manifest establishes training provenance; evaluation source is recorded independently.",
        "evaluation_overrides": overrides,
        "source": source_provenance(repository_root()),
        "packages": {},
        "runtime": runtime_provenance(embodied),
        "protocol": {
            "episodes": args.episodes,
            "envs": args.envs,
            "parallel_envs": False,
            "eval_seed": args.eval_seed,
            "worker_offset": args.worker_offset,
            "environment_seeds": [
                args.eval_seed + args.worker_offset + i for i in range(args.envs)
            ],
            "policy_mode": args.policy_mode,
            "diagnostic_seed": args.diagnostic_seed,
            "max_driver_steps": args.max_driver_steps,
            "sample_stride": args.sample_stride,
            "force_sample_first_post_death": True,
            "discount": gamma,
            "horizons": [5, 15],
            "policy_distribution": "learned stochastic policy without collection_unimix"
            if args.policy_mode == "eval_sample"
            else "modal actions; differs from stochastic training policy",
            "actor_rng": "canonical runtime; evaluation seed plus restored checkpoint n_actions",
            "critic_input": "current local posterior from returned actor carry; synchronized real roster/liveness",
            "value_target": "V(s_t) versus sum of gamma^(k-1)*reward[t+k] through true terminal",
            "finite_horizon": "H-step Bellman target includes gamma^H times same frozen head at actual successor; zero at terminal. This is not an H-step return predictor.",
            "timeout_semantics": "unchanged SMAC adapter: all is_last (including timeouts) are is_terminal",
            "reward": "one shared SMAC reward, never the sum of duplicated per-agent rewards",
            "state_weighting": "uniform sampled state-agent pairs; episode_initial and alive/dead slices reported separately; agents/states are correlated",
            "raw_alignment": "observation[t], action[t] chosen there, reward[t] received on arrival; terminal action is Driver-masked and not executed; episode_offsets includes reset and terminal states",
        },
    }
    for package in (
        "jax",
        "jaxlib",
        "numpy",
        "smac",
        "pysc2",
        "s2clientprotocol",
        "elements",
        "ninjax",
    ):
        try:
            provenance["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            provenance["packages"][package] = None
    write_json(output / "provenance.json", provenance)
    agent = make_agent(evaluated)
    loader = elements.Checkpoint()
    loader.agent = agent
    loader.load(str(checkpoint), keys=["agent"])
    before = {"parameters_sha256": parameter_digest(agent), "counters": counters(agent)}
    diagnostic = FrozenCentralCritic(agent, args.diagnostic_seed)
    environments = []
    records = collect_episodes(
        agent,
        partial(make_env, evaluated),
        diagnostic,
        episodes=args.episodes,
        envs=args.envs,
        worker_offset=args.worker_offset,
        policy_mode=args.policy_mode,
        max_driver_steps=args.max_driver_steps,
        environment_records=environments,
    )
    after = {"parameters_sha256": parameter_digest(agent), "counters": counters(agent)}
    if before != after:
        raise RuntimeError(
            f"Frozen evaluation changed agent state: before={before}, after={after}"
        )
    summary, packed = summarize(records, gamma=gamma, sample_stride=args.sample_stride)
    raw_path = output / "episodes.npz"
    with raw_path.open("xb") as stream:
        np.savez_compressed(stream, **packed)
    summary.update(
        {
            "frozen_state_before": before,
            "frozen_state_after": after,
            "environment_provenance": environments,
            "diagnostic_calls": diagnostic.calls,
            "raw_archive": {
                "file": raw_path.name,
                "sha256": digest_file(raw_path),
                "bytes": raw_path.stat().st_size,
            },
            "protocol": provenance["protocol"],
        }
    )
    write_json(output / "summary.json", summary)
    write_json(
        output / "done.json", {"summary_sha256": digest_file(output / "summary.json")}
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": summary["episodes"],
                "return_mean": summary["return_mean"],
                "win_rate": summary["win_rate"],
            }
        )
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Exact saved training config.yaml"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Completed checkpoint directory"
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        help="Optional exact launch/provenance JSON for this checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True, help="Must not exist")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=50_000)
    parser.add_argument("--worker-offset", type=int, default=100_000)
    parser.add_argument("--diagnostic-seed", type=int, default=9_001)
    parser.add_argument(
        "--policy-mode", choices=("eval", "eval_sample"), default="eval_sample"
    )
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-driver-steps", type=int, default=1_000_000)
    parser.add_argument("--platform", choices=("cpu", "cuda", "tpu"))
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
