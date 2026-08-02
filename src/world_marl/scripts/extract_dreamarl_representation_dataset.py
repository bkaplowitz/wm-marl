"""Extract aligned multi-agent tensors from a frozen DreaMARL run."""

from __future__ import annotations

import argparse
import gc
import pickle
from pathlib import Path

import elements
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import ruamel.yaml as yaml

from embodied.jax import internal

from world_marl.dreamarl import rssm, transformer_rssm
from world_marl.dreamarl.agent import _remove_agent_axis
from world_marl.dreamarl.axes import (
    GLOBAL_OBSERVATION_KEYS,
    broadcast_global_sequence,
    fold_agent_sequence,
    unfold_agent_sequence,
)
from world_marl.dreamarl.main import make_env
from world_marl.dreamarl.diagnostic_dataset import (
    save_dataset,
    sha256_file,
    valid_episode_starts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=32)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-deter", action="store_true")
    parser.add_argument("--transition-contract", action="store_true")
    parser.add_argument("--policy-checkpoint-label", default="unknown")
    parser.add_argument("--include-observations", action="store_true")
    return parser


def _load_config(run_dir: Path) -> elements.Config:
    data = yaml.YAML(typ="safe").load((run_dir / "config.yaml").read_text())
    config = elements.Config(data)
    config = config.update(
        jax={**dict(config.jax), "prealloc": False, "profiler": False},
    )
    return config


def _checkpoint_path(run_dir: Path) -> Path:
    root = run_dir / "ckpt"
    latest = (root / "latest").read_text().strip()
    path = root / latest / "agent.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_model_params(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    params = {
        key: value
        for key, value in checkpoint["params"].items()
        if key.startswith(("enc/", "dyn/"))
    }
    del checkpoint
    gc.collect()
    if not any(key.startswith("enc/") for key in params):
        raise ValueError("checkpoint contains no encoder parameters")
    if not any(key.startswith("dyn/") for key in params):
        raise ValueError("checkpoint contains no dynamics parameters")
    return params


def _spaces_and_modules(config):
    env = make_env(config, 0)
    if env.num_agents != config.agent.num_agents:
        raise ValueError((env.num_agents, config.agent.num_agents))
    joint_obs_space = {
        key: value for key, value in env.obs_space.items() if not key.startswith("log/")
    }
    joint_act_space = {key: value for key, value in env.act_space.items() if key != "reset"}
    env.close()
    local_obs_space = {
        key: (
            value
            if key in GLOBAL_OBSERVATION_KEYS
            else _remove_agent_axis(key, value, config.agent.num_agents)
        )
        for key, value in joint_obs_space.items()
    }
    local_act_space = {
        key: _remove_agent_axis(key, value, config.agent.num_agents)
        for key, value in joint_act_space.items()
    }
    exclude = {"is_first", "is_last", "is_terminal", "reward"}
    enc_space = {
        key: value for key, value in local_obs_space.items() if key not in exclude
    }
    enc_config = config.agent.enc[config.agent.enc.typ]
    encoder = rssm.Encoder(enc_space, **enc_config, name="enc")
    output_dim = encoder.calculate_encoder_output_dim(local_obs_space, enc_config)
    dyn_config = config.agent.dyn[config.agent.dyn.typ]
    dynamics = transformer_rssm.TransformerRSSM(
        local_act_space, output_dim, **dyn_config, name="dyn"
    )
    return joint_obs_space, joint_act_space, encoder, dynamics


def _sample_windows(
    replay_dir: Path,
    keys: set[str],
    trajectories: int,
    window: int,
    seed: int,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, object]]]:
    files = sorted(replay_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no replay chunks under {replay_dir}")
    generator = np.random.default_rng(seed)
    generator.shuffle(files)
    windows = []
    provenance = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            missing = keys - set(archive.files)
            if missing:
                raise ValueError(f"{path} is missing {sorted(missing)}")
            length = archive[sorted(keys)[0]].shape[0]
            first = np.asarray(archive["is_first"], bool)
            last = np.asarray(archive["is_last"], bool)
            starts = valid_episode_starts(first, last, length, window)
            if not starts:
                continue
            starts = np.asarray(starts)
            generator.shuffle(starts)
            for start in starts:
                windows.append(
                    {key: archive[key][start : start + window] for key in keys}
                )
                provenance.append(
                    {
                        "source_chunk": path.name,
                        "source_chunk_sha256": sha256_file(path),
                        "source_start": int(start),
                    }
                )
                # One window per chunk avoids overlapping replay windows in
                # different trajectory partitions.
                break
            if len(windows) == trajectories:
                return windows, provenance
    raise ValueError(
        f"requested {trajectories} windows but found only {len(windows)}"
    )


def _stack(windows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([window[key] for window in windows])
        for key in windows[0]
    }


def _extract_function(
    encoder,
    dynamics,
    joint_obs_space,
    joint_act_space,
    num_agents: int,
    context: int,
    length: int,
    include_observations: bool,
):
    local_obs_keys = set(joint_obs_space) - GLOBAL_OBSERVATION_KEYS

    def extract(data):
        context_entries = {
            "pair": fold_agent_sequence(data["dyn/pair"][:, :context], num_agents),
            "reset": fold_agent_sequence(
                data["dyn/reset"][:, :context], num_agents
            ),
            "stoch": fold_agent_sequence(
                data["dyn/stoch"][:, :context], num_agents
            ),
        }
        if "dyn/memory" in data:
            context_entries["memory"] = fold_agent_sequence(
                data["dyn/memory"][:, :context], num_agents
            )
        dyn_carry = dynamics.truncate(context_entries)
        target = slice(context, None)
        obs = {
            key: (
                fold_agent_sequence(data[key][:, target], num_agents)
                if key in local_obs_keys
                else broadcast_global_sequence(
                    data[key][:, target], num_agents
                )
            )
            for key in joint_obs_space
        }
        reset = obs["is_first"]
        _, _, tokens = encoder({}, obs, reset, training=False)
        previous_actions = {
            key: fold_agent_sequence(
                data[key][:, context - 1 : -1], num_agents
            )
            for key in joint_act_space
        }
        _, tensors = dynamics.representation_diagnostics(
            dyn_carry,
            tokens,
            previous_actions,
            reset,
            training=False,
        )
        grouped_all = {
            key: unfold_agent_sequence(value, num_agents)
            for key, value in tensors.items()
        }
        sequence_length = reset.shape[1]
        _, imagined, _ = dynamics.imagine(
            dyn_carry,
            previous_actions,
            length=sequence_length,
            training=False,
        )
        previous_stoch = jnp.concatenate(
            [dyn_carry["stoch"][:, None], imagined["stoch"][:, :-1]], 1
        )
        action_embedding = nn.DictConcat(dynamics.act_space, 1)(previous_actions)
        action_embedding /= jax.lax.stop_gradient(
            jnp.maximum(1, jnp.abs(action_embedding))
        )
        openloop_pair = jnp.concatenate(
            [previous_stoch.reshape((*previous_stoch.shape[:2], -1)), action_embedding],
            -1,
        )
        openloop = {
            "openloop_pair": jnp.float32(openloop_pair),
            "openloop_stoch": jnp.float32(previous_stoch),
            "openloop_prior_logit": jnp.float32(imagined["logit"]),
            "openloop_pred_token": jnp.float32(
                dynamics.predictor(imagined["deter"])
            ),
            **(
                {"openloop_memory": jnp.float32(imagined["memory"])}
                if "memory" in imagined
                else {}
            ),
        }
        grouped_all.update(
            {
                key: unfold_agent_sequence(value, num_agents)
                for key, value in openloop.items()
            }
        )
        grouped_all["valid"] = jnp.cumprod(
            ~grouped_all["reset"], axis=1
        ).astype(bool)

        # Preserve the legacy frozen-representation output while adding the
        # explicit causal transition contract used by the information ladder.
        grouped = {key: value[:, :length] for key, value in grouped_all.items()}
        source_deter = grouped_all["deter"][:, :length]
        source_stoch = grouped_all["stoch"][:, :length]
        grouped["belief"] = jnp.concatenate(
            [source_deter, source_stoch.reshape((*source_stoch.shape[:3], -1))],
            -1,
        )
        grouped["source_observation_token"] = grouped_all["target_token"][:, :length]
        grouped["next_target"] = grouped_all["target_token"][:, 1 : length + 1]
        grouped_previous_action = {
            key: unfold_agent_sequence(value, num_agents)
            for key, value in previous_actions.items()
        }
        if set(grouped_previous_action) != {"action"}:
            raise ValueError(
                "transition ladder currently requires one explicit action field"
            )
        grouped["action"] = grouped_previous_action["action"][:, 1 : length + 1]
        source = slice(context, context + length)
        successor = slice(context + 1, context + length + 1)
        source_last = data["is_last"][:, source]
        target_first = data["is_first"][:, successor]
        transition_valid = ~(source_last | target_first)
        grouped["reward"] = data["reward"][:, successor]
        grouped["is_last"] = data["is_last"][:, successor]
        grouped["is_terminal"] = data["is_terminal"][:, successor]
        grouped["valid"] = transition_valid
        grouped["reward_event"] = jnp.abs(grouped["reward"]) > 1e-8
        if include_observations:
            if "image" not in data:
                raise ValueError("run has no image observation for X3")
            grouped["observation"] = data["image"][:, source]
        return grouped

    return extract


def run(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir.expanduser().resolve()
    config = _load_config(run_dir)
    internal.setup(
        platform=str(config.jax.platform),
        compute_dtype=str(config.jax.compute_dtype),
        prealloc=False,
        transfer_guard=False,
    )
    joint_obs, joint_act, encoder, dynamics = _spaces_and_modules(config)
    required = set(joint_obs) | set(joint_act)
    required.update(
        {
            "dyn/pair",
            "dyn/reset",
            "dyn/stoch",
            "is_first",
            "is_last",
            "is_terminal",
            "reward",
        }
    )
    dyn_config = config.agent.dyn[config.agent.dyn.typ]
    if int(getattr(dyn_config, "memory_tokens", 0)):
        required.add("dyn/memory")
    extra = 1 if args.transition_contract else 0
    windows, provenance = _sample_windows(
        run_dir / "replay",
        required,
        args.trajectories,
        args.context + args.length + extra,
        args.seed,
    )
    checkpoint_path = _checkpoint_path(run_dir)
    params = _load_model_params(checkpoint_path)
    if bool(config.agent.slowenc.enable):
        raise NotImplementedError(
            "EMA target extraction must instantiate the resolved SlowModel; "
            "this run enables slowenc but the extractor does not"
        )
    if getattr(dyn_config, "interaction", "none") != "none":
        raise ValueError(
            "the neutral information ladder requires an interaction-free checkpoint"
        )
    extract = nj.pure(
        _extract_function(
            encoder,
            dynamics,
            joint_obs,
            joint_act,
            int(config.agent.num_agents),
            args.context,
            args.length,
            args.include_observations,
        )
    )
    outputs: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(windows), args.batch_size):
        batch = jax.tree.map(
            jnp.asarray, _stack(windows[start : start + args.batch_size])
        )
        _, tensors = extract(params, batch, seed=args.seed + start)
        tensors = jax.device_get(tensors)
        for key, value in tensors.items():
            if key == "deter" and not args.include_deter and not args.transition_contract:
                continue
            outputs.setdefault(key, []).append(np.asarray(value))
    dataset = {key: np.concatenate(values, 0) for key, values in outputs.items()}
    if args.transition_contract:
        dataset = {
            key: value[:, : args.length]
            for key, value in dataset.items()
        }
        trajectories, time, agents = dataset["belief"].shape[:3]
        dataset.update(
            trajectory_id=np.arange(trajectories, dtype=np.int64),
            episode_id=np.arange(trajectories, dtype=np.int64),
            timestep=np.broadcast_to(
                np.arange(time, dtype=np.int32)[None], (trajectories, time)
            ).copy(),
            policy_checkpoint=np.full(
                (trajectories,), args.policy_checkpoint_label, dtype="U128"
            ),
            agent_valid=np.broadcast_to(
                dataset["valid"][..., None], (trajectories, time, agents)
            ).copy(),
            action_available=np.broadcast_to(
                dataset["valid"][..., None], (trajectories, time, agents)
            ).copy(),
            track_id=np.broadcast_to(
                np.arange(agents, dtype=np.int32)[None, None],
                (trajectories, time, agents),
            ).copy(),
        )
        manifest = {
            "schema": "dreamarl_transition_ladder",
            "schema_version": 1,
            "temporal_contract": (
                "(belief_t,joint_action_t)->stopped_target_t_plus_1"
            ),
            "run_dir": str(run_dir),
            "task": str(config.task),
            "environment_seed": int(config.seed),
            "environment_config": {"task": str(config.task)},
            "num_agents": int(config.agent.num_agents),
            "action_dim": int(np.asarray(joint_act["action"].high).max()),
            "policy_checkpoint_label": args.policy_checkpoint_label,
            "representation_checkpoint": str(checkpoint_path),
            "representation_checkpoint_sha256": sha256_file(checkpoint_path),
            "target_encoder": "online_encoder_stop_gradient",
            "agent_ordering": "environment_possible_agents",
            "pre_or_post_action": {
                "belief": "post_observation_pre_action",
                "action": "action_applied_to_source_belief",
                "next_target": "post_action_successor_observation",
                "reward": "post_action_successor_reward",
            },
            "observation_available": bool(args.include_observations),
            "oracle_state_available": False,
            "interaction_labels": ["reward_event"],
            "trajectory_provenance": provenance,
        }
        save_dataset(args.output, dataset, manifest)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **dataset)
    print(f"Wrote {args.output}")
    for key, value in sorted(dataset.items()):
        print(f"  {key:16s} {value.shape} {value.dtype}")
    return args.output


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.context < 1 or args.length < 1 or args.batch_size < 1:
        raise ValueError("context, length, and batch-size must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
