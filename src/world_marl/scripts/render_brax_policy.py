"""Render learned policies from run checkpoints as MP4 videos in the real Brax env.

Point it at any level of a run tree — a ``wm_comparison_*`` root, an arm dir, a
single ``run_*`` dir, or a checkpoint dir — and it discovers every renderable
checkpoint (``checkpoint/`` from ``train_dmc_jepa``, ``policy_checkpoint/`` from
``train_single_genwm``), replays the deterministic evaluation policy in the real
Brax environment, and writes ``policy_video.mp4`` plus a ``policy_video.json``
sidecar (rendered returns/lengths) next to each checkpoint.

Requires the ``brax`` extra (brax + imageio + imageio-ffmpeg).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

CHECKPOINT_DIRNAMES = ("checkpoint", "policy_checkpoint")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="run/arm/comparison dirs (or checkpoint dirs) to search",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="per-episode step cap (also the env's truncation length)",
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--camera", default=None, help="mujoco camera name (default: free camera)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="playback fps (default: 1/env.dt, capped at 60)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-controls",
        action="store_true",
        help="also render negative-control checkpoints",
    )
    parser.add_argument("--out-name", default="policy_video.mp4")
    return parser.parse_args(argv)


@dataclass(frozen=True)
class CheckpointRef:
    checkpoint_dir: Path
    metadata: dict[str, Any]


def _read_ref(metadata_path: Path) -> CheckpointRef:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return CheckpointRef(checkpoint_dir=metadata_path.parent, metadata=metadata)


def _skip_reason(ref: CheckpointRef, *, include_controls: bool) -> str | None:
    env = str(ref.metadata.get("env", ""))
    if not env.startswith("brax:"):
        return f"env {env!r} is not a brax substrate"
    control = str(ref.metadata.get("control", "none"))
    if control != "none" and not include_controls:
        return f"negative control {control!r} (use --include-controls)"
    if ref.metadata.get("policy_trained") is False:
        return "checkpoint saved without a trained policy"
    return None


def discover_checkpoints(
    paths: list[Path], *, include_controls: bool = False
) -> tuple[list[CheckpointRef], list[str]]:
    """Find renderable checkpoints under ``paths``; returns (refs, skip messages)."""
    metadata_paths: set[Path] = set()
    for root in paths:
        if root.name in CHECKPOINT_DIRNAMES and (root / "metadata.json").is_file():
            metadata_paths.add(root / "metadata.json")
        for name in CHECKPOINT_DIRNAMES:
            metadata_paths.update(root.rglob(f"{name}/metadata.json"))
    refs: list[CheckpointRef] = []
    skipped: list[str] = []
    for metadata_path in sorted(metadata_paths):
        ref = _read_ref(metadata_path)
        reason = _skip_reason(ref, include_controls=include_controls)
        if reason is None:
            refs.append(ref)
        else:
            skipped.append(f"{ref.checkpoint_dir}: {reason}")
    return refs, skipped


ActFn = Callable[[jax.Array], jax.Array]


def load_jepa_actor(ref: CheckpointRef) -> ActFn:
    from world_marl.checkpointing import load_params
    from world_marl.jepa.models import JepaConfig
    from world_marl.jepa.training import (
        create_jepa_train_state,
        select_continuous_actions,
    )

    config = JepaConfig(**ref.metadata["jepa_config"])
    if config.action_mode != "continuous":
        raise ValueError(f"brax rendering needs continuous actions, got {config!r}")
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    state = state.replace(
        params=load_params(ref.checkpoint_dir / "checkpoint.msgpack", state.params)
    )
    action_low = jnp.full((config.action_dim,), -1.0, dtype=jnp.float32)
    action_high = jnp.full((config.action_dim,), 1.0, dtype=jnp.float32)

    def act(observations: jax.Array) -> jax.Array:
        return select_continuous_actions(
            state, observations, config, action_low, action_high
        )

    return act


def load_genwm_actor(ref: CheckpointRef) -> ActFn:
    from world_marl.checkpointing import load_params
    from world_marl.genwm import (
        GenWMConfig,
        GenieTokenizer,
        PPOConfig,
        create_genie_state,
        create_policy_state,
        make_genie_encode,
    )

    config = GenWMConfig(**ref.metadata["genwm_config"])
    if config.action_mode != "continuous":
        raise ValueError(f"brax rendering needs continuous actions, got {config!r}")
    ppo_config = PPOConfig(**ref.metadata["ppo_config"])
    policy_state = create_policy_state(jax.random.PRNGKey(0), config, ppo_config)
    params = load_params(ref.checkpoint_dir / "checkpoint.msgpack", policy_state.params)

    genie_kwargs = ref.metadata.get("genie")
    if genie_kwargs is not None:
        module = GenieTokenizer(**genie_kwargs)
        template = create_genie_state(jax.random.PRNGKey(0), module, learning_rate=1e-3)
        genie_params = load_params(
            ref.checkpoint_dir / "genie" / "checkpoint.msgpack", template.params
        )
        genie_encode = make_genie_encode(module)

        def encode(observations: jax.Array) -> jax.Array:
            return genie_encode(genie_params, observations)

    elif ref.metadata.get("latent_encoder") is not None:
        from world_marl.jepa.training import load_frozen_encoder

        encode, _ = load_frozen_encoder(ref.checkpoint_dir / "latent_encoder")
    else:

        def encode(observations: jax.Array) -> jax.Array:
            return observations

    apply_fn = policy_state.apply_fn

    @jax.jit
    def act(observations: jax.Array) -> jax.Array:
        policy, _ = apply_fn({"params": params}, encode(observations))
        return jnp.clip(policy.mode(), -1.0, 1.0)

    return act


def load_actor(ref: CheckpointRef) -> ActFn:
    algorithm = str(ref.metadata.get("algorithm", ""))
    if algorithm == "single_agent_sigreg_jepa_world_model":
        return load_jepa_actor(ref)
    if algorithm == "single_genwm_policy":
        return load_genwm_actor(ref)
    raise ValueError(f"unknown checkpoint algorithm {algorithm!r}")


def rollout_episodes(
    env: Any,
    act_fn: ActFn,
    *,
    episodes: int,
    max_steps: int,
    seed: int,
) -> tuple[list[Any], list[float], list[int]]:
    """Deterministic single-env rollouts; returns (pipeline states, returns, lengths)."""
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(seed)
    pipeline_states: list[Any] = []
    returns: list[float] = []
    lengths: list[int] = []
    for _ in range(episodes):
        key, reset_key = jax.random.split(key)
        state = reset(reset_key)
        pipeline_states.append(state.pipeline_state)
        episode_return = 0.0
        steps = 0
        while steps < max_steps:
            observations = state.obs.reshape((1, -1)).astype(jnp.float32)
            actions = act_fn(observations).reshape((-1,)).astype(jnp.float32)
            state = step(state, actions)
            pipeline_states.append(state.pipeline_state)
            episode_return += float(state.reward)
            steps += 1
            if float(state.done) > 0.5:
                break
        returns.append(episode_return)
        lengths.append(steps)
    return pipeline_states, returns, lengths


def video_fps(env: Any, fps_override: float | None) -> float:
    if fps_override is not None:
        return float(fps_override)
    return min(60.0, 1.0 / float(env.dt))


def render_checkpoint(ref: CheckpointRef, args: argparse.Namespace) -> Path:
    from brax.io import image

    from world_marl.envs.brax_adapter import brax_env_name, make_brax_env

    act_fn = load_actor(ref)
    env = make_brax_env(
        brax_env_name(str(ref.metadata["env"])),
        episode_length=args.max_steps,
        auto_reset=False,
    )
    pipeline_states, returns, lengths = rollout_episodes(
        env,
        act_fn,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    frames = image.render_array(
        env.unwrapped.sys,
        [jax.device_get(state) for state in pipeline_states],
        args.height,
        args.width,
        args.camera,
    )

    import imageio.v2 as imageio

    run_dir = ref.checkpoint_dir.parent
    out_path = run_dir / args.out_name
    imageio.mimwrite(out_path, frames, fps=video_fps(env, args.fps))
    sidecar = {
        "checkpoint_dir": str(ref.checkpoint_dir),
        "env": ref.metadata["env"],
        "algorithm": ref.metadata.get("algorithm"),
        "arm": ref.metadata.get("arm"),
        "episodes": len(returns),
        "returns": returns,
        "lengths": lengths,
        "mean_return": float(np.mean(returns)),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "fps": video_fps(env, args.fps),
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    refs, skipped = discover_checkpoints(
        args.paths, include_controls=args.include_controls
    )
    for message in skipped:
        print(f"[skip] {message}")
    if not refs:
        print("no renderable brax checkpoints found")
        return 1
    failures = 0
    for ref in refs:
        try:
            out_path = render_checkpoint(ref, args)
        except Exception as error:  # noqa: BLE001 - report per checkpoint, keep going
            failures += 1
            print(f"[fail] {ref.checkpoint_dir}: {error}")
            continue
        print(f"[done] {out_path}")
    if failures == len(refs):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
