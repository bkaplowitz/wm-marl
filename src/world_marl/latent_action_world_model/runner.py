"""Shared staged runner for the independent Jafar and Jasmine CLI arms."""

import argparse
from collections.abc import Mapping
import dataclasses
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Literal, NamedTuple

from flax.core import FrozenDict, freeze, unfreeze
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image

from world_marl.algs.ippo import IPPOConfig, create_train_state, select_actions
from world_marl.checkpointing import save_checkpoint
from world_marl.jafar.config import (
    DynamicsConfig as JafarDynamicsConfig,
    JafarConfig,
    LAMConfig as JafarLAMConfig,
    TokenizerConfig as JafarTokenizerConfig,
)
from world_marl.jafar.lam import LatentActionModel as JafarLAM
from world_marl.jafar.model import JafarWorldModel
from world_marl.jafar.tokenizer import TokenizerVQVAE
from world_marl.jafar import training as jafar_training
from world_marl.jasmine.config import (
    DynamicsConfig as JasmineDynamicsConfig,
    JasmineConfig,
    LAMConfig as JasmineLAMConfig,
    TokenizerConfig as JasmineTokenizerConfig,
)
from world_marl.jasmine.lam import LatentActionModel as JasmineLAM
from world_marl.jasmine.model import JasmineWorldModel
from world_marl.jasmine.tokenizer import TokenizerMAE
from world_marl.jasmine import training as jasmine_training
from world_marl.latent_action_world_model.bridge import load_expert_bridge
from world_marl.latent_action_world_model.heads import (
    RewardContinueHeads,
    create_head_train_state,
    scan_head_updates,
)
from world_marl.latent_action_world_model.policy import scan_simulator_ppo_updates
from world_marl.latent_action_world_model.replay import (
    pair_valid_transitions,
    to_backend_sequence,
)
from world_marl.latent_action_world_model.simulator import (
    create_jafar_replay_pool,
    create_jasmine_replay_pool,
    infer_jafar_codes,
    infer_jasmine_codes,
    initialize_jafar_state,
    initialize_jasmine_state,
    jafar_simulator_step,
    jafar_transition_features,
    jasmine_simulator_step,
    jasmine_transition_features,
)
from world_marl.world_model_foundation.collect import (
    collect_world_model_sequence,
    make_single_agent_adapter,
    write_json_artifact,
    write_jsonl_metrics,
)
from world_marl.world_model_foundation.metrics import scanned_episode_metrics
from world_marl.world_model_foundation.replay import sequence_batch_to_jax

Arm = Literal["jafar", "jasmine"]

SOURCE_COMMITS = {
    "jafar": {
        "repository": "https://github.com/FLAIROx/jafar",
        "commit": "5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665",
        "paths": [
            "jafar/models/tokenizer.py",
            "jafar/models/lam.py",
            "jafar/models/dynamics.py",
            "jafar/models/genie.py",
        ],
    },
    "jasmine": {
        "repository": "https://github.com/p-doom/jasmine",
        "commit": "420859bc99eecf6b07a7e9edf65d5d145935f1e1",
        "paths": [
            "jasmine/models/tokenizer.py",
            "jasmine/models/lam.py",
            "jasmine/models/dynamics.py",
            "jasmine/models/genie.py",
        ],
    },
}


class TrainedArm(NamedTuple):
    config: Any
    model: Any
    tokenizer_state: TrainState
    lam_state: TrainState
    world_model_state: TrainState
    tokenizer_metrics: dict[str, jax.Array]
    lam_metrics: dict[str, jax.Array]
    dynamics_metrics: dict[str, jax.Array]


def parse_args(arm: Arm, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Train the source-derived {arm.capitalize()} world-model arm."
    )
    parser.add_argument("--env", default="synthetic:image-grid")
    parser.add_argument("--out-dir", type=Path, default=Path(f"runs/{arm}"))
    parser.add_argument("--expert-calibration", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-size", choices=("source", "debug"), default="source")
    parser.add_argument("--time-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--tokenizer-steps", type=int, default=None)
    parser.add_argument("--lam-steps", type=int, default=None)
    parser.add_argument("--dynamics-steps", type=int, default=None)
    parser.add_argument("--reward-continue-steps", type=int, default=10_000)
    parser.add_argument("--policy-train-steps", type=int, default=10)
    parser.add_argument("--imagination-horizon", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--brax-backend", default=None)
    parser.add_argument("--dmc-workers", type=int, default=1)
    parser.add_argument("--dmc-camera-id", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "time_steps",
        "batch_size",
        "sequence_length",
        "image_size",
        "action_dim",
        "reward_continue_steps",
        "policy_train_steps",
        "imagination_horizon",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.time_steps < 2 or args.sequence_length < 2:
        parser.error("time and sequence lengths must be at least two")
    return args


def _debug_jafar_config(args: argparse.Namespace) -> JafarConfig:
    return dataclasses.replace(
        JafarConfig(),
        sequence_length=args.sequence_length,
        image_height=args.image_size,
        image_width=args.image_size,
        tokenizer=JafarTokenizerConfig(
            model_dim=8,
            latent_dim=4,
            num_latents=8,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
            codebook_dropout=0.0,
        ),
        lam=JafarLAMConfig(
            model_dim=8,
            latent_dim=4,
            num_latents=6,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
        ),
        dynamics=JafarDynamicsConfig(
            model_dim=8,
            num_latents=8,
            num_blocks=1,
            num_heads=2,
            maskgit_steps=2,
        ),
    )


def _debug_jasmine_config(args: argparse.Namespace) -> JasmineConfig:
    return dataclasses.replace(
        JasmineConfig(),
        sequence_length=args.sequence_length,
        image_height=args.image_size,
        image_width=args.image_size,
        tokenizer=JasmineTokenizerConfig(
            model_dim=8,
            ffn_dim=16,
            latent_dim=4,
            num_latents=8,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
            dtype=jnp.bfloat16,
            use_flash_attention=False,
        ),
        lam=JasmineLAMConfig(
            model_dim=8,
            ffn_dim=16,
            latent_dim=4,
            num_latents=6,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
            dtype=jnp.bfloat16,
            use_flash_attention=False,
        ),
        dynamics=JasmineDynamicsConfig(
            model_dim=8,
            ffn_dim=16,
            latent_patch_dim=4,
            latent_action_dim=4,
            num_blocks=1,
            num_heads=2,
            denoise_steps=4,
            dtype=jnp.bfloat16,
            use_flash_attention=False,
        ),
    )


def _jafar_tokenizer(config: JafarConfig) -> TokenizerVQVAE:
    value = config.tokenizer
    return TokenizerVQVAE(
        in_dim=value.in_dim,
        model_dim=value.model_dim,
        latent_dim=value.latent_dim,
        num_latents=value.num_latents,
        patch_size=value.patch_size,
        num_blocks=value.num_blocks,
        num_heads=value.num_heads,
        dropout=value.dropout,
        codebook_dropout=value.codebook_dropout,
    )


def _jafar_lam(config: JafarConfig) -> JafarLAM:
    value = config.lam
    return JafarLAM(
        in_dim=value.in_dim,
        model_dim=value.model_dim,
        latent_dim=value.latent_dim,
        num_latents=value.num_latents,
        patch_size=value.patch_size,
        num_blocks=value.num_blocks,
        num_heads=value.num_heads,
        dropout=value.dropout,
        codebook_dropout=value.codebook_dropout,
    )


def _jafar_model(config: JafarConfig) -> JafarWorldModel:
    return JafarWorldModel(config.tokenizer, config.lam, config.dynamics)


def _jasmine_tokenizer(config: JasmineConfig) -> TokenizerMAE:
    value = config.tokenizer
    return TokenizerMAE(
        in_dim=value.in_dim,
        model_dim=value.model_dim,
        ffn_dim=value.ffn_dim,
        latent_dim=value.latent_dim,
        num_latents=value.num_latents,
        patch_size=value.patch_size,
        num_blocks=value.num_blocks,
        num_heads=value.num_heads,
        dropout=value.dropout,
        max_mask_ratio=value.max_mask_ratio,
        param_dtype=value.param_dtype,
        dtype=value.dtype,
        use_flash_attention=value.use_flash_attention,
    )


def _jasmine_lam(config: JasmineConfig) -> JasmineLAM:
    value = config.lam
    return JasmineLAM(
        in_dim=value.in_dim,
        model_dim=value.model_dim,
        ffn_dim=value.ffn_dim,
        latent_dim=value.latent_dim,
        num_latents=value.num_latents,
        patch_size=value.patch_size,
        num_blocks=value.num_blocks,
        num_heads=value.num_heads,
        dropout=value.dropout,
        codebook_dropout=value.codebook_dropout,
        param_dtype=value.param_dtype,
        dtype=value.dtype,
        use_flash_attention=value.use_flash_attention,
    )


def _jasmine_model(config: JasmineConfig) -> JasmineWorldModel:
    return JasmineWorldModel(
        config.tokenizer,
        config.lam,
        config.dynamics,
        lam_co_train=True,
    )


def _jafar_schedule(stage, updates: int) -> optax.Schedule:
    warmup = min(stage.warmup_steps, updates)
    return optax.join_schedules(
        (
            optax.linear_schedule(
                stage.initial_learning_rate,
                stage.peak_learning_rate,
                max(warmup, 1),
            ),
            optax.constant_schedule(stage.peak_learning_rate),
        ),
        (warmup,),
    )


def _jasmine_schedule(stage, updates: int) -> optax.Schedule:
    warmup = min(stage.warmup_steps, max(updates - 1, 0))
    decay = min(stage.wsd_decay_steps, max(updates - warmup, 0))
    return jasmine_training.wsd_schedule(
        stage.initial_learning_rate,
        stage.peak_learning_rate,
        stage.decay_end,
        updates,
        warmup,
        decay,
    )


def _overlay_params(target: Any, source: Any) -> Any:
    if isinstance(target, Mapping):
        return {
            key: _overlay_params(value, source[key]) for key, value in target.items()
        }
    return source


def _install_pretrained(
    state: TrainState,
    tokenizer_params: Any,
    lam_params: Any,
) -> TrainState:
    mutable = unfreeze(state.params)
    mutable["tokenizer"] = _overlay_params(mutable["tokenizer"], tokenizer_params)
    mutable["lam"] = _overlay_params(mutable["lam"], lam_params)
    params = freeze(mutable) if isinstance(state.params, FrozenDict) else mutable
    return state.replace(params=params)


def _stage_videos(config: Any, videos: jax.Array) -> tuple[jax.Array, ...]:
    batch_sizes = (
        config.tokenizer_training.batch_size,
        config.lam_training.batch_size,
        config.dynamics_training.batch_size,
    )
    if videos.shape[0] < 1:
        raise ValueError("at least one collected sequence is required")
    return tuple(
        jnp.take(videos, jnp.arange(batch_size) % videos.shape[0], axis=0)
        for batch_size in batch_sizes
    )


def _train_jafar(
    config: JafarConfig,
    videos: jax.Array,
    args: argparse.Namespace,
) -> TrainedArm:
    tokenizer_videos, lam_videos, dynamics_videos = (
        _stage_videos(config, videos)
        if args.model_size == "source"
        else (videos, videos, videos)
    )
    tokenizer_steps = args.tokenizer_steps or config.tokenizer_training.updates
    lam_steps = args.lam_steps or config.lam_training.updates
    dynamics_steps = args.dynamics_steps or config.dynamics_training.updates
    tokenizer = _jafar_tokenizer(config)
    tokenizer_state = jafar_training.create_tokenizer_train_state(
        jax.random.PRNGKey(args.seed + 1),
        tokenizer,
        tokenizer_videos,
        args.learning_rate
        or _jafar_schedule(config.tokenizer_training, tokenizer_steps),
    )
    tokenizer_state, tokenizer_metrics = jafar_training.scan_tokenizer_updates(
        tokenizer_state,
        jnp.repeat(tokenizer_videos[None], tokenizer_steps, axis=0),
        jax.random.split(jax.random.PRNGKey(args.seed + 2), tokenizer_steps),
        config.tokenizer.vq_beta,
    )
    lam = _jafar_lam(config)
    lam_state = jafar_training.create_lam_train_state(
        jax.random.PRNGKey(args.seed + 3),
        lam,
        lam_videos,
        args.learning_rate or _jafar_schedule(config.lam_training, lam_steps),
    )
    lam_state, _, lam_metrics = jafar_training.scan_lam_updates(
        lam_state,
        jnp.zeros((6,), dtype=jnp.int32),
        jnp.repeat(lam_videos[None], lam_steps, axis=0),
        jax.random.split(jax.random.PRNGKey(args.seed + 4), lam_steps),
        config.lam.vq_beta,
        config.lam.reset_inactive_after,
    )
    model = _jafar_model(config)
    example = {
        "videos": dynamics_videos,
        "mask_rng": jax.random.PRNGKey(args.seed + 5),
    }
    world_state = jafar_training.create_dynamics_train_state(
        jax.random.PRNGKey(args.seed + 6),
        model,
        example,
        args.learning_rate or _jafar_schedule(config.dynamics_training, dynamics_steps),
    )
    world_state = _install_pretrained(
        world_state, tokenizer_state.params, lam_state.params
    )
    world_state, dynamics_metrics = jafar_training.scan_dynamics_updates(
        world_state,
        {
            "videos": jnp.repeat(dynamics_videos[None], dynamics_steps, axis=0),
            "mask_rng": jax.random.split(
                jax.random.PRNGKey(args.seed + 7), dynamics_steps
            ),
        },
    )
    return TrainedArm(
        config,
        model,
        tokenizer_state,
        lam_state,
        world_state,
        tokenizer_metrics,
        lam_metrics,
        dynamics_metrics,
    )


def _train_jasmine(
    config: JasmineConfig,
    videos: jax.Array,
    args: argparse.Namespace,
) -> TrainedArm:
    videos = videos.astype(config.tokenizer.dtype)
    tokenizer_videos, lam_videos, dynamics_videos = (
        _stage_videos(config, videos)
        if args.model_size == "source"
        else (videos, videos, videos)
    )
    tokenizer_steps = args.tokenizer_steps or config.tokenizer_training.updates
    lam_steps = args.lam_steps or config.lam_training.updates
    dynamics_steps = args.dynamics_steps or config.dynamics_training.updates
    tokenizer = _jasmine_tokenizer(config)
    tokenizer_state = jasmine_training.create_tokenizer_train_state(
        jax.random.PRNGKey(args.seed + 1),
        tokenizer,
        tokenizer_videos,
        args.learning_rate
        or _jasmine_schedule(config.tokenizer_training, tokenizer_steps),
    )
    tokenizer_state, tokenizer_metrics = jasmine_training.scan_tokenizer_updates(
        tokenizer_state,
        jnp.repeat(tokenizer_videos[None], tokenizer_steps, axis=0),
        jax.random.split(jax.random.PRNGKey(args.seed + 2), tokenizer_steps),
    )
    lam = _jasmine_lam(config)
    lam_state = jasmine_training.create_lam_train_state(
        jax.random.PRNGKey(args.seed + 3),
        lam,
        lam_videos,
        args.learning_rate or _jasmine_schedule(config.lam_training, lam_steps),
    )
    lam_state, _, lam_metrics = jasmine_training.scan_lam_updates(
        lam_state,
        jnp.zeros((6,), dtype=jnp.int32),
        jnp.repeat(lam_videos[None], lam_steps, axis=0),
        jax.random.split(jax.random.PRNGKey(args.seed + 4), lam_steps),
        config.lam.vq_beta,
        config.lam.reset_inactive_after,
    )
    model = _jasmine_model(config)
    example = {
        "videos": dynamics_videos,
        "rng": jax.random.PRNGKey(args.seed + 5),
    }
    world_state = jasmine_training.create_dynamics_train_state(
        jax.random.PRNGKey(args.seed + 6),
        model,
        example,
        args.learning_rate
        or _jasmine_schedule(config.dynamics_training, dynamics_steps),
    )
    world_state = _install_pretrained(
        world_state, tokenizer_state.params, lam_state.params
    )
    world_state, dynamics_metrics = jasmine_training.scan_dynamics_updates(
        world_state,
        {
            "videos": jnp.repeat(dynamics_videos[None], dynamics_steps, axis=0),
            "rng": jax.random.split(jax.random.PRNGKey(args.seed + 7), dynamics_steps),
        },
    )
    return TrainedArm(
        config,
        model,
        tokenizer_state,
        lam_state,
        world_state,
        tokenizer_metrics,
        lam_metrics,
        dynamics_metrics,
    )


def _metric_rows(metrics: Mapping[str, jax.Array]) -> list[dict[str, float | int]]:
    values = jax.device_get(metrics)
    steps = len(next(iter(values.values())))
    return [
        {"step": step, **{name: float(value[step]) for name, value in values.items()}}
        for step in range(steps)
    ]


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type) or "dtype" in type(value).__name__.lower():
        return str(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_rollout(path: Path, observations: jax.Array) -> None:
    frames = np.asarray(jax.device_get(observations[-1, :, 0]), dtype=np.float32)
    panel = np.concatenate(frames, axis=1)
    Image.fromarray(np.asarray(np.clip(panel, 0, 1) * 255, dtype=np.uint8)).save(path)


def _evaluate_real(
    args: argparse.Namespace,
    batch: Any,
    policy_state: TrainState,
    bridge: Any,
) -> list[dict[str, float | str]]:
    if args.env.startswith("synthetic:"):
        return [
            {
                "episode": 0,
                "return": float(np.mean(np.sum(batch.rewards, axis=0))),
                "length": float(batch.time_steps),
                "policy_source": "latent_policy_with_expert_bridge",
                "evaluation_execution": "synthetic",
            }
        ]
    target_episodes = max(args.eval_episodes, 1)
    evaluation_num_envs = math.gcd(args.num_envs, target_episodes)
    adapter = make_single_agent_adapter(
        args.env,
        num_envs=evaluation_num_envs,
        max_cycles=args.max_cycles,
        seed=args.seed + 20_000,
        brax_backend=args.brax_backend,
        dmc_workers=args.dmc_workers,
        image_size=args.image_size,
        dmc_camera_id=args.dmc_camera_id,
    )
    try:
        scan_rollout = getattr(adapter, "scan_recurrent_rollout", None)
        if scan_rollout is None:
            raise RuntimeError(f"{args.env} must expose scan_recurrent_rollout")
        observations = np.asarray(adapter.reset(), dtype=np.float32).reshape(
            (adapter.num_envs, args.image_size, args.image_size, 3)
        )

        def policy_step(state, key, flat_observations, is_first):
            del is_first
            key, action_key = jax.random.split(key)
            current = flat_observations.reshape(
                (adapter.num_envs, args.image_size, args.image_size, 3)
            )
            latent_codes, _, _ = select_actions(
                state,
                action_key,
                current,
                deterministic=True,
            )
            action_key, bridge_key = jax.random.split(action_key)
            choices = jax.random.uniform(bridge_key, latent_codes.shape)
            indices = jnp.floor(choices * bridge.counts[latent_codes]).astype(jnp.int32)
            real_actions = bridge.actions[latent_codes, indices]
            return key, real_actions

        evaluation_steps = math.ceil(target_episodes / adapter.num_envs) * (
            args.max_cycles + 1
        )
        ys, _, _ = scan_rollout(
            policy_step,
            policy_state,
            jax.random.PRNGKey(args.seed + 30_000),
            evaluation_steps,
            observations=observations,
        )
        _, _, rewards, _, dones = ys
        return scanned_episode_metrics(
            rewards,
            dones,
            target_episodes=target_episodes,
            policy_source="latent_policy_with_expert_bridge",
            arrival_aligned=True,
        )
    finally:
        close = getattr(adapter, "close", None)
        if close is not None:
            close()


def _save_checkpoints(
    out_dir: Path,
    arm: Arm,
    trained: TrainedArm,
    head_state: TrainState,
    policy_state: TrainState,
    steps: Mapping[str, int],
) -> None:
    states = {
        "tokenizer": trained.tokenizer_state,
        "lam": trained.lam_state,
        "world_model": trained.world_model_state,
        "reward_continue": head_state,
        "ppo": policy_state,
    }
    for stage, state in states.items():
        save_checkpoint(
            out_dir / "checkpoints" / stage,
            state,
            metadata={
                "model": arm,
                "stage": stage,
                "updates": int(steps[stage]),
                "source": SOURCE_COMMITS[arm],
            },
        )


def main(arm: Arm, argv: list[str] | None = None) -> int:
    args = parse_args(arm, argv)
    started = time.monotonic()
    batch = collect_world_model_sequence(
        env_name=args.env,
        time_steps=args.time_steps,
        batch_size=args.batch_size,
        observation_shape=(args.image_size, args.image_size, 3),
        action_dim=args.action_dim,
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=args.seed,
        brax_backend=args.brax_backend,
        dmc_workers=args.dmc_workers,
        image_size=args.image_size,
        dmc_camera_id=args.dmc_camera_id,
    )
    replay = sequence_batch_to_jax(batch)
    backend = to_backend_sequence(replay)
    sequence_length = min(args.sequence_length, backend.observations.shape[1])
    videos = backend.observations[:, :sequence_length]

    if arm == "jafar":
        config = (
            _debug_jafar_config(args) if args.model_size == "debug" else JafarConfig()
        )
        trained = _train_jafar(config, videos, args)
        feature_fn = jafar_transition_features
        infer_fn = infer_jafar_codes
        initialize_state = initialize_jafar_state
        create_pool = create_jafar_replay_pool
    else:
        config = (
            _debug_jasmine_config(args)
            if args.model_size == "debug"
            else JasmineConfig()
        )
        trained = _train_jasmine(config, videos, args)
        feature_fn = jasmine_transition_features
        infer_fn = infer_jasmine_codes
        initialize_state = initialize_jasmine_state
        create_pool = create_jasmine_replay_pool

    pairs = pair_valid_transitions(backend)
    transition_videos = jnp.stack([pairs.observations, pairs.next_observations], axis=1)
    features, inferred_codes = feature_fn(
        trained.model,
        trained.world_model_state.params,
        transition_videos,
    )
    head_state = create_head_train_state(
        jax.random.PRNGKey(args.seed + 8),
        RewardContinueHeads(),
        features,
        learning_rate=args.learning_rate or 1e-3,
    )
    head_state, head_metrics = scan_head_updates(
        head_state,
        jnp.repeat(features[None], args.reward_continue_steps, axis=0),
        jnp.repeat(pairs.rewards[None], args.reward_continue_steps, axis=0),
        jnp.repeat(pairs.continues[None], args.reward_continue_steps, axis=0),
    )

    def infer_calibration(values):
        return infer_fn(trained.model, trained.world_model_state.params, values)

    bridge = load_expert_bridge(
        args.expert_calibration,
        infer_codes=infer_calibration,
    )
    if bridge.environment != args.env:
        raise ValueError(
            f"calibration environment {bridge.environment!r} does not match {args.env!r}"
        )

    initial_pixels = videos[:, :1]
    replay_pixels = backend.observations.reshape(-1, 1, *backend.observations.shape[2:])
    simulator_state = initialize_state(
        trained.model, trained.world_model_state.params, initial_pixels
    )
    replay_pool = create_pool(
        trained.model, trained.world_model_state.params, replay_pixels
    )
    policy_config = IPPOConfig(
        learning_rate=args.learning_rate or 5e-4,
        update_epochs=1 if args.model_size == "debug" else 4,
        num_minibatches=1 if args.model_size == "debug" else 4,
        network_arch="cnn",
    )
    policy_state = create_train_state(
        jax.random.PRNGKey(args.seed + 9),
        (args.image_size, args.image_size, 3),
        action_dim=6,
        config=policy_config,
    )

    if arm == "jafar":

        def simulator_step(state, codes, key):
            return jafar_simulator_step(
                trained.model,
                trained.world_model_state.params,
                head_state,
                replay_pool,
                state,
                codes,
                key,
                sampler_steps=trained.config.dynamics.maskgit_steps,
                sample_argmax=True,
            )
    else:

        def simulator_step(state, codes, key):
            return jasmine_simulator_step(
                trained.model,
                trained.world_model_state.params,
                head_state,
                replay_pool,
                state,
                codes,
                key,
                diffusion_steps=trained.config.dynamics.denoise_steps,
                context_corruption=trained.config.dynamics.context_corruption,
            )

    policy_state, _, rollouts, ppo_metrics = scan_simulator_ppo_updates(
        policy_state,
        simulator_state,
        simulator_step,
        jax.random.PRNGKey(args.seed + 10),
        updates=args.policy_train_steps,
        horizon=args.imagination_horizon,
        config=policy_config,
    )
    real_evaluation = _evaluate_real(args, batch, policy_state, bridge)

    tokenizer_rows = _metric_rows(trained.tokenizer_metrics)
    lam_rows = _metric_rows(trained.lam_metrics)
    dynamics_rows = _metric_rows(trained.dynamics_metrics)
    head_rows = _metric_rows(head_metrics)
    ppo_rows = _metric_rows(ppo_metrics)
    elapsed = time.monotonic() - started
    gate_passed = all(
        rows[-1]["loss"] <= rows[0]["loss"]
        for rows in (tokenizer_rows, lam_rows, dynamics_rows, head_rows)
    )
    status = "ok" if gate_passed else "learning_gate_failed"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = {
        "tokenizer": len(tokenizer_rows),
        "lam": len(lam_rows),
        "world_model": len(dynamics_rows),
        "reward_continue": len(head_rows),
        "ppo": len(ppo_rows),
    }
    _save_checkpoints(out_dir, arm, trained, head_state, policy_state, steps)
    write_json_artifact(
        out_dir / "config.json",
        {
            "model": arm,
            "model_size": args.model_size,
            "source_defaults": _jsonable(trained.config),
            "runtime": _jsonable(vars(args)),
        },
    )
    write_json_artifact(out_dir / "sources.json", SOURCE_COMMITS[arm])
    write_json_artifact(
        out_dir / "replay_metadata.json",
        {
            **_jsonable(batch.metadata),
            "time_major_shape": list(batch.observations.shape),
            "batch_major_shape": list(backend.observations.shape),
            "valid_transitions": int(backend.valid_transitions.sum()),
        },
    )
    write_jsonl_metrics(out_dir / "tokenizer_metrics.jsonl", tokenizer_rows)
    write_jsonl_metrics(out_dir / "lam_metrics.jsonl", lam_rows)
    write_jsonl_metrics(out_dir / "dynamics_metrics.jsonl", dynamics_rows)
    write_jsonl_metrics(out_dir / "reward_continue_metrics.jsonl", head_rows)
    write_jsonl_metrics(out_dir / "ppo_metrics.jsonl", ppo_rows)
    write_jsonl_metrics(out_dir / "real_evaluation.jsonl", real_evaluation)
    code_counts = np.bincount(
        np.asarray(jax.device_get(inferred_codes)), minlength=6
    ).tolist()
    write_json_artifact(
        out_dir / "code_usage.json",
        {
            "training_transition_counts": code_counts,
            "training_coverage": int(np.count_nonzero(code_counts)),
            "bridge_counts": np.asarray(bridge.counts).tolist(),
            "bridge_coverage": 6,
        },
    )
    bridge_payload = {
        "calibration_path": str(args.expert_calibration),
        "calibration_sha256": _sha256(args.expert_calibration),
        "environment": bridge.environment,
        "provenance": _jsonable(bridge.provenance),
        "counts": np.asarray(bridge.counts).tolist(),
        "coverage": 6,
        "sampling": "uniform_over_every_recorded_action_per_code",
        "fallback": None,
    }
    write_json_artifact(out_dir / "bridge.json", bridge_payload)
    _save_rollout(out_dir / "rollout.png", rollouts.observations)

    learned_return = float(jnp.mean(jnp.sum(rollouts.rewards[-1], axis=0)))
    real_return = float(np.mean([row["return"] for row in real_evaluation]))
    random_return = float(np.mean(np.sum(batch.rewards, axis=0)))
    outcome = {
        "status": status,
        "learning_gate_passed": gate_passed,
        "final_tokenizer_loss": tokenizer_rows[-1]["loss"],
        "final_lam_loss": lam_rows[-1]["loss"],
        "final_dynamics_loss": dynamics_rows[-1]["loss"],
        "final_reward_continue_loss": head_rows[-1]["loss"],
        "random_return": random_return,
        "learned_simulator_return": learned_return,
        "bridged_real_return": real_return,
        "elapsed_seconds": elapsed,
        "updates_per_second": sum(steps.values()) / max(elapsed, 1e-9),
        "jax_platform": jax.default_backend(),
    }
    write_json_artifact(out_dir / "outcome.json", outcome)
    write_json_artifact(
        out_dir / "summary.json",
        {
            "schema_version": 1,
            "model": arm,
            "env": args.env,
            "seed": args.seed,
            "status": status,
            "budgets": steps,
            "bridge": bridge_payload,
            "metrics": outcome,
        },
    )
    return 0 if gate_passed or args.allow_fail else 1
