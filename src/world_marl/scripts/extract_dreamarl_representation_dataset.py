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
) -> list[dict[str, np.ndarray]]:
    files = sorted(replay_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no replay chunks under {replay_dir}")
    generator = np.random.default_rng(seed)
    generator.shuffle(files)
    windows = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            missing = keys - set(archive.files)
            if missing:
                raise ValueError(f"{path} is missing {sorted(missing)}")
            length = archive[next(iter(keys))].shape[0]
            if length < window:
                continue
            starts = np.arange(length - window + 1)
            generator.shuffle(starts)
            # One window per chunk prevents nearly identical overlapping
            # transitions from leaking across the held-out split.
            start = starts[0]
            windows.append(
                {key: archive[key][start : start + window] for key in keys}
            )
            if len(windows) == trajectories:
                return windows
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
        grouped = {
            key: unfold_agent_sequence(value, num_agents)
            for key, value in tensors.items()
        }
        length = reset.shape[1]
        _, imagined, _ = dynamics.imagine(
            dyn_carry,
            previous_actions,
            length=length,
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
        }
        grouped.update(
            {
                key: unfold_agent_sequence(value, num_agents)
                for key, value in openloop.items()
            }
        )
        grouped["valid"] = jnp.cumprod(~grouped["reset"], axis=1).astype(bool)
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
    required.update({"dyn/pair", "dyn/reset", "dyn/stoch"})
    windows = _sample_windows(
        run_dir / "replay",
        required,
        args.trajectories,
        args.context + args.length,
        args.seed,
    )
    params = _load_model_params(_checkpoint_path(run_dir))
    extract = nj.pure(
        _extract_function(
            encoder,
            dynamics,
            joint_obs,
            joint_act,
            int(config.agent.num_agents),
            args.context,
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
            if key == "deter" and not args.include_deter:
                continue
            outputs.setdefault(key, []).append(np.asarray(value))
    dataset = {key: np.concatenate(values, 0) for key, values in outputs.items()}
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
