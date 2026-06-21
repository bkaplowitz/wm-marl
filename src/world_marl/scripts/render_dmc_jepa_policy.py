"""Render real DMC rollouts from a trained JEPA actor checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tqdm import tqdm

from world_marl.checkpointing import load_metadata, load_params
from world_marl.envs.dmc_adapter import (
    _flatten_observation,
    _observation_keys,
    dmc_env_name,
    is_dmc_substrate,
    make_dmc_env,
)
from world_marl.jepa.models import JepaConfig
from world_marl.jepa.training import create_jepa_train_state, select_continuous_actions
from world_marl.logging import to_jsonable


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a trained DMC JEPA actor policy to an animated GIF.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "DMC JEPA experiment directory with summary.json, single run directory "
            "with checkpoint/, or direct checkpoint directory."
        ),
    )
    parser.add_argument(
        "--control",
        default="none",
        help="Control to select when --run-dir is an experiment directory.",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help="Optional run index to select from an experiment summary.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output GIF path. Defaults to <selected-run>/policy_rollout.gif.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="DMC camera id passed to physics.render.",
    )
    parser.add_argument(
        "--mujoco-gl",
        default="egl",
        choices=("egl", "osmesa", "glfw"),
        help=(
            "MuJoCo rendering backend. RunPod/headless GPU pods usually want egl; "
            "CPU-only headless machines may need osmesa."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output, metadata = render_policy_rollout(
        args.run_dir,
        control=args.control,
        run_index=args.run_index,
        output_path=args.out,
        seed=args.seed,
        max_steps=args.max_steps,
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_id=args.camera_id,
        mujoco_gl=args.mujoco_gl,
        quiet=args.quiet,
    )
    print(output)
    print(json.dumps(to_jsonable(metadata), indent=2, sort_keys=True))
    return 0


def render_policy_rollout(
    run_dir: str | Path,
    *,
    control: str = "none",
    run_index: int | None = None,
    output_path: str | Path | None = None,
    seed: int = 0,
    max_steps: int = 500,
    width: int = 480,
    height: int = 360,
    fps: int = 30,
    camera_id: int = 0,
    mujoco_gl: str = "egl",
    quiet: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Load a JEPA actor checkpoint, render one real DMC rollout, and save a GIF."""

    configure_mujoco_rendering(mujoco_gl)
    selected = resolve_run_source(run_dir, control=control, run_index=run_index)
    if output_path is None:
        output = selected.run_dir / "policy_rollout.gif"
    else:
        output = Path(output_path)
    if output.suffix.lower() != ".gif":
        output = output.with_suffix(".gif")
    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_metadata = load_metadata(selected.checkpoint_dir)
    config = JepaConfig(**checkpoint_metadata["jepa_config"])
    if config.action_mode != "continuous":
        raise ValueError("DMC policy rendering requires a continuous-action JEPA model")

    env_name = _env_name(checkpoint_metadata, selected.run_dir)
    env = make_dmc_env(env_name, seed=seed)
    try:
        action_spec = env.action_spec()
        action_shape = tuple(int(dim) for dim in action_spec.shape) or (1,)
        action_dim = int(np.prod(action_shape))
        action_low = np.broadcast_to(
            np.asarray(action_spec.minimum, dtype=np.float32),
            action_shape,
        ).reshape((action_dim,))
        action_high = np.broadcast_to(
            np.asarray(action_spec.maximum, dtype=np.float32),
            action_shape,
        ).reshape((action_dim,))

        state = create_jepa_train_state(jax.random.PRNGKey(seed), config)
        state = state.replace(
            params=load_params(
                selected.checkpoint_dir / "checkpoint.msgpack",
                state.params,
            ),
        )
        obs_keys = _observation_keys(env.observation_spec())
        timestep = env.reset()
        frames = [_render_frame(env, width=width, height=height, camera_id=camera_id)]
        episode_return = 0.0
        steps = 0
        actions: list[list[float]] = []

        iterator = tqdm(
            range(max_steps),
            desc="render policy",
            unit="step",
            disable=quiet,
        )
        for _ in iterator:
            observation = _flatten_observation(timestep.observation, obs_keys)
            action = np.asarray(
                select_continuous_actions(
                    state,
                    jnp.asarray(observation[None, :], dtype=jnp.float32),
                    config,
                    jnp.asarray(action_low, dtype=jnp.float32),
                    jnp.asarray(action_high, dtype=jnp.float32),
                )[0],
                dtype=np.float32,
            )
            action = np.clip(action, action_low, action_high)
            timestep = env.step(action.reshape(action_shape))
            reward = 0.0 if timestep.reward is None else float(timestep.reward)
            episode_return += reward
            steps += 1
            actions.append(action.reshape((-1,)).tolist())
            frames.append(
                _render_frame(env, width=width, height=height, camera_id=camera_id)
            )
            if timestep.last():
                break

        save_gif(frames, output, fps=fps)
        metadata = {
            "output": str(output),
            "run_dir": str(selected.run_dir),
            "checkpoint_dir": str(selected.checkpoint_dir),
            "control": selected.control,
            "run_index": selected.run_index,
            "env": f"dmc:{env_name}",
            "seed": seed,
            "episode_return": episode_return,
            "episode_length": steps,
            "fps": fps,
            "width": width,
            "height": height,
            "camera_id": camera_id,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "action_mean": (
                np.asarray(actions, dtype=np.float32).mean(axis=0).tolist()
                if actions
                else []
            ),
            "action_std": (
                np.asarray(actions, dtype=np.float32).std(axis=0).tolist()
                if actions
                else []
            ),
        }
        output.with_suffix(".json").write_text(
            json.dumps(to_jsonable(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output, metadata
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


@dataclasses.dataclass(frozen=True)
class RunSource:
    run_dir: Path
    checkpoint_dir: Path
    control: str
    run_index: int | None


def resolve_run_source(
    path: str | Path,
    *,
    control: str = "none",
    run_index: int | None = None,
) -> RunSource:
    """Resolve an experiment, run, or checkpoint path to one checkpoint."""

    source = Path(path)
    if (source / "checkpoint.msgpack").exists():
        return RunSource(
            run_dir=source.parent,
            checkpoint_dir=source,
            control=control,
            run_index=run_index,
        )
    if (source / "checkpoint" / "checkpoint.msgpack").exists():
        return RunSource(
            run_dir=source,
            checkpoint_dir=source / "checkpoint",
            control=control,
            run_index=run_index,
        )
    summary_path = source / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        candidates = [
            run
            for run in summary.get("runs", [])
            if str(run.get("control", "none")) == control
            and (run_index is None or int(run.get("run_index", -1)) == run_index)
        ]
        if not candidates:
            raise ValueError(
                f"no run matched control={control!r}, run_index={run_index!r}",
            )
        selected = max(
            candidates,
            key=lambda run: _selection_score(run),
        )
        selected_run_dir = Path(selected["run_dir"])
        if not selected_run_dir.is_absolute():
            selected_run_dir = source / selected_run_dir
            if not selected_run_dir.exists():
                selected_run_dir = Path(selected["run_dir"])
        checkpoint_dir = selected_run_dir / "checkpoint"
        return RunSource(
            run_dir=selected_run_dir,
            checkpoint_dir=checkpoint_dir,
            control=control,
            run_index=selected.get("run_index"),
        )
    raise FileNotFoundError(
        "expected a checkpoint directory, run directory with checkpoint/, "
        f"or experiment directory with summary.json: {source}",
    )


def save_gif(frames: list[np.ndarray], output: str | Path, *, fps: int) -> None:
    if not frames:
        raise ValueError("cannot save GIF without frames")
    if fps < 1:
        raise ValueError("fps must be >= 1")
    images = [
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("P")
        for frame in frames
    ]
    duration_ms = max(1, int(round(1000 / fps)))
    images[0].save(
        Path(output),
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def configure_mujoco_rendering(mujoco_gl: str) -> None:
    """Set MuJoCo's rendering backend before dm_control imports MuJoCo."""

    if mujoco_gl not in ("egl", "osmesa", "glfw"):
        raise ValueError("mujoco_gl must be one of: egl, osmesa, glfw")
    os.environ["MUJOCO_GL"] = mujoco_gl
    if os.environ["MUJOCO_GL"] == "egl":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    elif os.environ["MUJOCO_GL"] == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def _selection_score(run: dict[str, Any]) -> float:
    for key in ("policy_trained_mean", "policy_improvement", "final_open_loop_loss"):
        value = run.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            score = float(value)
            return -score if key == "final_open_loop_loss" else score
    return 0.0


def _env_name(checkpoint_metadata: dict[str, Any], run_dir: Path) -> str:
    env = checkpoint_metadata.get("env")
    if not env:
        config_path = run_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            env = config.get("args", {}).get("env")
    if not env:
        raise ValueError("checkpoint metadata did not include a DMC env")
    return dmc_env_name(env) if is_dmc_substrate(env) else str(env)


def _render_frame(env, *, width: int, height: int, camera_id: int) -> np.ndarray:
    return np.asarray(
        env.physics.render(height=height, width=width, camera_id=camera_id),
        dtype=np.uint8,
    )


if __name__ == "__main__":
    raise SystemExit(main())
