"""Verify dependency imports, environment stepping, rollout, and PPO update."""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp

from world_marl.algs.ippo import (
    IPPOConfig,
    create_train_state as create_ippo_train_state,
    ppo_update,
)
from world_marl.algs.mappo import (
    MAPPOConfig,
    create_train_state as create_mappo_train_state,
    mappo_update,
)
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.logging import dependency_versions, to_jsonable
from world_marl.training import (
    central_observation_shape,
    collect_mappo_rollout,
    collect_rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ippo", "mappo"), default="ippo")
    parser.add_argument("--substrate", default="coins")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--observation-size", type=int, default=22)
    parser.add_argument(
        "--append-agent-id",
        action="store_true",
        help="Append one-hot agent identity channels to each RGB observation.",
    )
    parser.add_argument(
        "--include-observation-scalars",
        action="store_true",
        help="Append scalar Melting Pot observation keys as constant image channels.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit nonzero unless JAX exposes at least one GPU device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if args.require_gpu and not gpu_devices:
        raise SystemExit(f"JAX did not expose a GPU. Devices: {devices}")

    adapter = MeltingPotVectorAdapter(
        substrate=args.substrate,
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        observation_size=args.observation_size,
        include_observation_scalars=args.include_observation_scalars,
        append_agent_id=args.append_agent_id,
    )
    try:
        observations = adapter.reset()
        config_cls = MAPPOConfig if args.algorithm == "mappo" else IPPOConfig
        config = config_cls(update_epochs=1, num_minibatches=1, learning_rate=1e-4)
        key = jax.random.PRNGKey(0)
        key, init_key, rollout_key, update_key = jax.random.split(key, 4)
        if args.algorithm == "mappo":
            train_state = create_mappo_train_state(
                init_key,
                adapter.observation_shape,
                central_observation_shape(
                    adapter.observation_shape, adapter.num_agents
                ),
                adapter.action_dim,
                config,
            )
            rollout = collect_mappo_rollout(
                adapter,
                train_state,
                observations,
                rollout_key,
                rollout_steps=4,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
            update_fn = jax.jit(
                lambda state, batch, last_values, rng: mappo_update(
                    state,
                    batch,
                    last_values,
                    rng,
                    config,
                )
            )
        else:
            train_state = create_ippo_train_state(
                init_key,
                adapter.observation_shape,
                adapter.action_dim,
                config,
            )
            rollout = collect_rollout(
                adapter,
                train_state,
                observations,
                rollout_key,
                rollout_steps=4,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
            update_fn = jax.jit(
                lambda state, batch, last_values, rng: ppo_update(
                    state,
                    batch,
                    last_values,
                    rng,
                    config,
                )
            )
        train_state, metrics = update_fn(
            train_state,
            rollout.batch,
            rollout.last_values,
            update_key,
        )
        jax.block_until_ready(jnp.asarray(metrics["total_loss"]))
        payload = {
            "status": "ok",
            "versions": dependency_versions(),
            "jax_default_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in devices],
            "jax_gpu_devices": [str(device) for device in gpu_devices],
            "algorithm": args.algorithm,
            "substrate": adapter.substrate,
            "num_envs": adapter.num_envs,
            "num_agents": adapter.num_agents,
            "observation_shape": adapter.observation_shape,
            "central_observation_shape": (
                central_observation_shape(adapter.observation_shape, adapter.num_agents)
                if args.algorithm == "mappo"
                else None
            ),
            "raw_observation_shape": adapter.raw_observation_shape,
            "include_observation_scalars": adapter.include_observation_scalars,
            "scalar_observation_keys": adapter.scalar_observation_keys,
            "append_agent_id": adapter.append_agent_id,
            "action_dim": adapter.action_dim,
            "rollout": rollout.metrics,
            "update_metrics": metrics,
        }
        print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
