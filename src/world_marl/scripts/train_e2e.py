"""End-to-end learning validation CLI."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

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
from world_marl.checkpointing import load_metadata, load_params, save_checkpoint
from world_marl.config import TrainConfig
from world_marl.envs.gymnax_adapter import (
    GymnaxVectorAdapter,
    gymnax_env_name,
    is_gymnax_substrate,
)
from world_marl.envs.jaxmarl_coin_adapter import (
    JaxMARLCoinGameVectorAdapter,
    coin_game_reward_done,
)
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.evaluation import (
    EvaluationResult,
    evaluate_policy,
    evaluate_policy_scan,
    evaluate_random_policy_scan,
    mappo_train_state_policy,
    random_policy,
    train_state_policy,
)
from world_marl.logging import RunLogger, dependency_versions, timestamp, to_jsonable
from world_marl.training import (
    ObservationMode,
    central_observation_shape,
    collect_mappo_rollout,
    collect_rollout,
    train_real_scan,
    training_window_means,
)
from world_marl.world_model import (
    VectorWorldModelConfig,
    create_world_model_state,
    train_imagined_scan,
)
from world_marl.world_model_training import (
    collect_policy_transition_batch_scan,
    collect_random_transition_batch_scan,
    concatenate_transition_batches,
    fit_world_model_steps,
)

TrainingAdapter = (
    MeltingPotVectorAdapter | JaxMARLCoinGameVectorAdapter | GymnaxVectorAdapter
)


@dataclass(frozen=True)
class RunOutcome:
    name: str
    run_dir: str
    control: str | None
    random_mean: float
    initial_mean: float
    trained_mean: float
    improvement: float
    random_improvement: float
    initial_improvement: float
    first_window_mean: float
    final_window_mean: float
    checkpoint_dir: str
    runtime_seconds: float
    real_env_steps: int
    imagined_env_steps: int
    cumulative_real_episodes: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class RunTimer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def to_dict(self) -> dict[str, Any]:
        return {"runtime_seconds": self.elapsed()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="YAML file of defaults; explicit CLI flags override its values.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Mirror per-update metrics to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", default="world-marl")
    parser.add_argument("--algorithm", choices=("ippo", "mappo"), default="ippo")
    parser.add_argument("--substrate", default="coins")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--total-env-steps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument(
        "--observation-size",
        type=int,
        default=None,
        help="Optional square RGB downsample size, e.g. 22 or 44. Default keeps 88x88.",
    )
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
        "--stochastic-eval",
        action="store_true",
        help="Evaluate learned policies by sampling instead of taking argmax actions.",
    )
    parser.add_argument("--eval-max-steps", type=int, default=None)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--min-improvement", type=float, default=0.2)
    parser.add_argument(
        "--negative-control",
        choices=("none", "freeze-policy", "shuffle-rewards", "zero-advantages"),
        default="freeze-policy",
    )
    parser.add_argument(
        "--prefit-world-model",
        action="store_true",
        help="Fit a vector-state world model before PPO and train on model rollouts.",
    )
    parser.add_argument("--wm-random-rollouts", type=int, default=1)
    parser.add_argument("--wm-initial-rollouts", type=int, default=1)
    parser.add_argument("--wm-fit-steps", type=int, default=10_000)
    parser.add_argument("--wm-learning-rate", type=float, default=3e-4)
    parser.add_argument("--wm-hidden-dim", type=int, default=256)
    parser.add_argument("--wm-integration-steps", type=int, default=10)
    parser.add_argument(
        "--wm-policy-warmup-updates",
        type=int,
        default=0,
        help=(
            "With --prefit-world-model, run this many real-env PPO/MAPPO "
            "updates before collecting policy rollouts for the world model."
        ),
    )
    parser.add_argument(
        "--wm-flow-type",
        choices=("gaussian", "linear", "discrete", "transformer"),
        default="linear",
    )
    parser.add_argument(
        "--wm-num-categories",
        type=int,
        default=9,
        help=(
            "Per-factor category count for --wm-flow-type discrete (coins = 9); "
            "must divide num_agents*state_dim. Ignored by the continuous flows."
        ),
    )

    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--activation", choices=("relu", "tanh"), default="relu")

    parser.add_argument(
        "--eval-checkpoint",
        default=None,
        help=argparse.SUPPRESS,
    )

    known, _ = parser.parse_known_args()
    if known.config:
        raw = OmegaConf.to_container(OmegaConf.load(known.config), resolve=True)
        if not isinstance(raw, dict):
            parser.error(f"{known.config} must contain a top-level mapping")
        overrides = {str(key): value for key, value in raw.items()}
        unknown = set(overrides) - set(vars(known))
        if unknown:
            parser.error(f"unknown keys in {known.config}: {sorted(unknown)}")
        parser.set_defaults(**overrides)

    args = parser.parse_args()
    if args.prefit_world_model:
        if args.wm_random_rollouts < 1:
            parser.error("--wm-random-rollouts must be >= 1")
        if args.wm_initial_rollouts < 1:
            parser.error("--wm-initial-rollouts must be >= 1")
        if args.wm_fit_steps < 1:
            parser.error("--wm-fit-steps must be >= 1")
        if args.wm_hidden_dim < 1:
            parser.error("--wm-hidden-dim must be >= 1")
        if args.wm_integration_steps < 1:
            parser.error("--wm-integration-steps must be >= 1")
        if args.wm_policy_warmup_updates < 0:
            parser.error("--wm-policy-warmup-updates must be >= 0")
    elif args.wm_policy_warmup_updates:
        parser.error("--wm-policy-warmup-updates requires --prefit-world-model")
    return args


def algorithm_config_from_args(
    cfg: TrainConfig,
    control: str | None = None,
) -> IPPOConfig | MAPPOConfig:
    config_cls = MAPPOConfig if cfg.algorithm == "mappo" else IPPOConfig
    uses_vector_policy = (
        cfg.substrate == "coins"
        or is_gymnax_substrate(cfg.substrate)
        or getattr(cfg, "prefit_world_model", False)
    )
    config = config_cls(
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_eps=cfg.clip_eps,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        update_epochs=cfg.update_epochs,
        num_minibatches=cfg.num_minibatches,
        activation=cfg.activation,
        network_arch="mlp" if uses_vector_policy else "cnn",
    )
    if control == "shuffle-rewards":
        return replace(config, shuffle_rewards=True)
    if control == "zero-advantages":
        return replace(config, zero_advantages=True)
    return config


def create_algorithm_train_state(
    algorithm: str,
    rng: jax.Array,
    adapter: TrainingAdapter,
    config: IPPOConfig | MAPPOConfig,
    *,
    observation_mode: ObservationMode = "image",
):
    observation_shape = _policy_observation_shape(adapter, observation_mode)
    if algorithm == "mappo":
        if not isinstance(config, MAPPOConfig):
            raise TypeError("MAPPO training requires a MAPPOConfig")
        return create_mappo_train_state(
            rng,
            observation_shape,
            central_observation_shape(
                observation_shape,
                adapter.num_agents,
                observation_mode=observation_mode,
            ),
            adapter.action_dim,
            config,
        )
    if not isinstance(config, IPPOConfig):
        raise TypeError("IPPO training requires an IPPOConfig")
    return create_ippo_train_state(
        rng,
        observation_shape,
        adapter.action_dim,
        config,
    )


def policy_from_train_state(
    algorithm: str,
    train_state,
    *,
    adapter: TrainingAdapter,
    deterministic: bool,
    seed: int,
    observation_mode: ObservationMode = "image",
):
    policy_fn = mappo_train_state_policy if algorithm == "mappo" else train_state_policy
    return policy_fn(
        train_state,
        num_envs=adapter.num_envs,
        num_agents=adapter.num_agents,
        deterministic=deterministic,
        seed=seed,
        observation_mode=observation_mode,
    )


def _policy_observation_shape(
    adapter: TrainingAdapter,
    observation_mode: ObservationMode,
) -> tuple[int, ...]:
    if observation_mode == "vector":
        return (int(np.prod(adapter.observation_shape)),)
    if observation_mode == "image":
        return adapter.observation_shape
    raise ValueError(f"unsupported observation_mode {observation_mode!r}")


def _make_training_adapter(cfg: TrainConfig, *, seed: int) -> TrainingAdapter:
    if cfg.substrate == "coins":
        return JaxMARLCoinGameVectorAdapter(
            num_envs=cfg.num_envs,
            max_cycles=cfg.max_cycles,
            seed=seed,
        )
    if is_gymnax_substrate(cfg.substrate):
        return GymnaxVectorAdapter(
            env_name=gymnax_env_name(cfg.substrate),
            num_envs=cfg.num_envs,
            max_cycles=cfg.max_cycles,
            seed=seed,
        )
    return MeltingPotVectorAdapter(
        substrate=cfg.substrate,
        num_envs=cfg.num_envs,
        max_cycles=cfg.max_cycles,
        observation_size=cfg.observation_size,
        include_observation_scalars=cfg.include_observation_scalars,
        append_agent_id=cfg.append_agent_id,
    )


def _make_reward_done_fn(cfg: TrainConfig):
    """Return the analytic reward/done callback for the world-model rollout.

    The world model predicts only next-state dynamics, so rewards/dones come from
    the environment's known reward function evaluated on the model's states.
    """
    if cfg.substrate == "coins":
        return coin_game_reward_done
    raise NotImplementedError(
        "--prefit-world-model needs an analytic reward_done_fn for substrate "
        f"{cfg.substrate!r}; only 'coins' is currently supported."
    )


def _evaluate_train_state(
    *,
    adapter: TrainingAdapter,
    train_state,
    algorithm: str,
    observation_mode: ObservationMode,
    deterministic: bool,
    seed: int,
    episodes: int,
    max_steps: int | None,
) -> EvaluationResult:
    """Evaluate a train state on the accelerator for vector policies, else via
    the Python loop (MeltingPot only). The scan reproduces the Python loop's
    deterministic episodes exactly, so the logged value is unchanged -- only the
    rollout leaves the CPU. The scan does not advance the adapter PRNG state;
    callers reusing the adapter for training reset it afterwards.
    """
    if observation_mode == "vector" and hasattr(adapter, "scan_rewards_dones"):
        return evaluate_policy_scan(
            adapter,
            train_state,
            episodes=episodes,
            deterministic=deterministic,
            observation_mode=observation_mode,
            seed=seed,
            algorithm=algorithm,
        )
    return evaluate_policy(
        adapter,
        policy_from_train_state(
            algorithm,
            train_state,
            adapter=adapter,
            deterministic=deterministic,
            seed=seed,
            observation_mode=observation_mode,
        ),
        episodes=episodes,
        max_steps=max_steps,
    )


def evaluate_checkpoint_mode(cfg: TrainConfig) -> None:
    checkpoint_dir = Path(cfg.eval_checkpoint)
    metadata = load_metadata(checkpoint_dir)
    algorithm = metadata.get("algorithm", "ippo")

    cfg.substrate = cfg.substrate or metadata["substrate"]
    if cfg.observation_size is None:
        cfg.observation_size = metadata.get("observation_size")
    cfg.include_observation_scalars = cfg.include_observation_scalars or metadata.get(
        "include_observation_scalars", False
    )
    cfg.append_agent_id = cfg.append_agent_id or metadata.get("append_agent_id", False)
    adapter = _make_training_adapter(cfg, seed=cfg.seed)
    try:
        config_payload = metadata.get("algorithm_config", metadata.get("ippo_config"))
        if config_payload is None:
            raise KeyError("checkpoint metadata missing algorithm_config")
        config = (
            MAPPOConfig(**config_payload)
            if algorithm == "mappo"
            else IPPOConfig(**config_payload)
        )
        train_state = create_algorithm_train_state(
            algorithm,
            jax.random.PRNGKey(0),
            adapter,
            config,
            observation_mode=metadata.get("observation_mode", "image"),
        )
        params = load_params(checkpoint_dir / "checkpoint.msgpack", train_state.params)
        train_state = train_state.replace(params=params)
        result = _evaluate_train_state(
            adapter=adapter,
            train_state=train_state,
            algorithm=algorithm,
            observation_mode=metadata.get("observation_mode", "image"),
            deterministic=not cfg.stochastic_eval,
            seed=cfg.seed,
            episodes=cfg.eval_episodes,
            max_steps=cfg.eval_max_steps,
        )
        print(json.dumps(to_jsonable(result.to_dict()), sort_keys=True))
    finally:
        adapter.close()


def evaluate_random_baseline(cfg: TrainConfig, seed: int) -> dict[str, Any]:
    adapter = _make_training_adapter(cfg, seed=seed)
    try:
        if hasattr(adapter, "scan_rewards_dones"):
            result = evaluate_random_policy_scan(
                adapter,
                episodes=cfg.eval_episodes,
                seed=seed,
            )
        else:
            result = evaluate_policy(
                adapter,
                random_policy(adapter, np.random.default_rng(seed)),
                episodes=cfg.eval_episodes,
                max_steps=cfg.eval_max_steps,
            )
        return result.to_dict()
    finally:
        adapter.close()


def evaluate_checkpoint_subprocess(
    cfg: TrainConfig,
    checkpoint_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "world_marl.scripts.train_e2e",
        "--eval-checkpoint",
        str(checkpoint_dir),
        "--substrate",
        cfg.substrate,
        "--num-envs",
        str(cfg.num_envs),
        "--eval-episodes",
        str(cfg.eval_episodes),
        "--seed",
        str(seed),
        "--max-cycles",
        str(cfg.max_cycles),
    ]
    if cfg.observation_size is not None:
        command.extend(["--observation-size", str(cfg.observation_size)])
    if cfg.include_observation_scalars:
        command.append("--include-observation-scalars")
    if cfg.append_agent_id:
        command.append("--append-agent-id")
    if cfg.stochastic_eval:
        command.append("--stochastic-eval")
    if cfg.eval_max_steps is not None:
        command.extend(["--eval-max-steps", str(cfg.eval_max_steps)])
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip())


def _collect_real_env_rollout(
    cfg: TrainConfig,
    collect_fn,
    adapter: TrainingAdapter,
    train_state,
    observations: np.ndarray,
    rollout_key: jax.Array,
    config: IPPOConfig | MAPPOConfig,
    observation_mode: ObservationMode,
) -> Any:
    kwargs: dict[str, Any] = {}
    if cfg.algorithm == "mappo":
        kwargs["observation_mode"] = observation_mode
    return collect_fn(
        adapter,
        train_state,
        observations,
        rollout_key,
        rollout_steps=cfg.rollout_steps,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        **kwargs,
    )


def _rows_from_stacked(
    stacked: dict[str, Any],
    *,
    steps_per_update: int,
    real: bool,
    real_env_steps: int,
    imagined_env_steps: int,
    cumulative_real_episodes: int,
    extra_fields: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Convert stacked scan metrics into host-schema rows, advancing the counters."""
    host = {key: np.asarray(value) for key, value in jax.device_get(stacked).items()}
    num_updates = int(host["rollout_mean_reward"].shape[0])
    env_steps = 0
    rows: list[dict[str, Any]] = []
    for index in range(num_updates):
        metrics: dict[str, Any] = {}
        for key, values in host.items():
            value = values[index]
            if key == "completed_episodes":
                metrics[key] = int(value)
            elif key in {"episode_return_mean", "episode_length_mean"} and np.isnan(
                value
            ):
                metrics[key] = None
            else:
                metrics[key] = float(value)
        completed = metrics.get("completed_episodes", 0) if real else 0
        env_steps += steps_per_update
        if real:
            real_env_steps += steps_per_update
            cumulative_real_episodes += completed
        else:
            imagined_env_steps += steps_per_update
        rows.append(
            {
                "update": index + 1,
                "env_steps": env_steps,
                "real_env_steps": real_env_steps,
                "imagined_env_steps": imagined_env_steps,
                "completed_real_episodes": completed,
                "cumulative_real_episodes": cumulative_real_episodes,
                **(extra_fields or {}),
                **metrics,
            }
        )
    return rows, real_env_steps, imagined_env_steps, cumulative_real_episodes


def run_training(
    cfg: TrainConfig,
    *,
    run_dir: Path,
    name: str,
    run_index: int,
    control: str | None,
) -> RunOutcome:
    wandb_run = None
    if cfg.wandb:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project,
            group=run_dir.parent.name,
            name=name,
            config=dataclasses.asdict(cfg),
            reinit=True,
        )
    logger = RunLogger(run_dir, wandb_run=wandb_run)
    timer = RunTimer()
    seed = cfg.seed + run_index * 10_000 + (5_000 if control else 0)
    rng = jax.random.PRNGKey(seed)
    config = algorithm_config_from_args(cfg, control)
    freeze_policy = control == "freeze-policy"
    observation_mode: ObservationMode = (
        "vector"
        if cfg.substrate == "coins"
        or is_gymnax_substrate(cfg.substrate)
        or cfg.prefit_world_model
        else "image"
    )

    logger.write_json(
        "config.json",
        {
            "args": dataclasses.asdict(cfg),
            "run_index": run_index,
            "control": control,
            "seed": seed,
            "algorithm": cfg.algorithm,
            "observation_mode": observation_mode,
            "algorithm_config": dataclasses.asdict(config),
        },
    )
    logger.write_json("versions.json", dependency_versions())

    random_result = evaluate_random_baseline(cfg, seed=seed + 1)
    logger.write_json("random_baseline.json", random_result)

    adapter = _make_training_adapter(cfg, seed=seed)
    scan_path = hasattr(adapter, "scan_rollout")
    rows: list[dict[str, Any]] = []
    try:
        observations = adapter.reset()
        rng, init_key = jax.random.split(rng)
        train_state = create_algorithm_train_state(
            cfg.algorithm,
            init_key,
            adapter,
            config,
            observation_mode=observation_mode,
        )
        initial_result = _evaluate_train_state(
            adapter=adapter,
            train_state=train_state,
            algorithm=cfg.algorithm,
            observation_mode=observation_mode,
            deterministic=not cfg.stochastic_eval,
            seed=seed + 2,
            episodes=cfg.eval_episodes,
            max_steps=cfg.eval_max_steps,
        ).to_dict()
        logger.write_json("initial_policy_evaluation.json", initial_result)
        observations = adapter.reset()
        world_model_state = None
        world_model_config = None
        model_start_states = None
        world_model_prefit_loss = None
        reward_done_fn = None
        real_env_steps = 0
        imagined_env_steps = 0
        cumulative_real_episodes = 0

        if cfg.prefit_world_model:
            reward_done_fn = _make_reward_done_fn(cfg)
            if cfg.wm_policy_warmup_updates:
                train_state, observations, rng, warmup_stacked = train_real_scan(
                    adapter,
                    train_state,
                    observations,
                    rng,
                    num_updates=cfg.wm_policy_warmup_updates,
                    config=config,
                    rollout_steps=cfg.rollout_steps,
                    algorithm=cfg.algorithm,
                    freeze_policy=freeze_policy,
                )
                (
                    warmup_rows,
                    real_env_steps,
                    _,
                    cumulative_real_episodes,
                ) = _rows_from_stacked(
                    warmup_stacked,
                    steps_per_update=cfg.num_envs * cfg.rollout_steps,
                    real=True,
                    real_env_steps=real_env_steps,
                    imagined_env_steps=0,
                    cumulative_real_episodes=cumulative_real_episodes,
                )
                logger.write_json(
                    "world_model_policy_warmup.json",
                    {
                        "updates": cfg.wm_policy_warmup_updates,
                        "real_env_steps": (
                            cfg.wm_policy_warmup_updates
                            * cfg.num_envs
                            * cfg.rollout_steps
                        ),
                        "rows": warmup_rows,
                    },
                )

            rng, random_collect_key = jax.random.split(rng)
            random_batch, observations, random_start_states, random_stats = (
                collect_random_transition_batch_scan(
                    adapter,
                    observations,
                    random_collect_key,
                    rollout_steps=cfg.wm_random_rollouts,
                )
            )
            real_env_steps += random_stats.real_env_steps
            cumulative_real_episodes += random_stats.completed_episodes
            rng, policy_collect_key = jax.random.split(rng)
            (
                policy_batch,
                observations,
                rng,
                policy_start_states,
                policy_stats,
            ) = collect_policy_transition_batch_scan(
                adapter,
                train_state,
                observations,
                policy_collect_key,
                rollout_steps=cfg.wm_initial_rollouts,
                algorithm=cfg.algorithm,
            )
            real_env_steps += policy_stats.real_env_steps
            cumulative_real_episodes += policy_stats.completed_episodes
            prefit_batch = concatenate_transition_batches([random_batch, policy_batch])
            model_start_states = jnp.concatenate(
                [random_start_states, policy_start_states],
                axis=0,
            )
            state_dim = int(prefit_batch.states.shape[-1])
            is_discrete_flow = cfg.wm_flow_type in {"discrete", "transformer"}
            num_categories = cfg.wm_num_categories if is_discrete_flow else 0
            if is_discrete_flow:
                transition_dim = adapter.num_agents * state_dim
                if num_categories <= 0 or transition_dim % num_categories != 0:
                    raise ValueError(
                        "--wm-num-categories must be > 0 and divide "
                        f"num_agents*state_dim={transition_dim} for discrete flow "
                        f"(got {num_categories})"
                    )
            world_model_config = VectorWorldModelConfig(
                state_dim=state_dim,
                num_agents=adapter.num_agents,
                action_dim=adapter.action_dim,
                hidden_dims=(cfg.wm_hidden_dim, cfg.wm_hidden_dim),
                learning_rate=cfg.wm_learning_rate,
                integration_steps=cfg.wm_integration_steps,
                flow_type="discrete" if is_discrete_flow else cfg.wm_flow_type,
                num_categories=num_categories,
                discrete_arch=(
                    "transformer" if cfg.wm_flow_type == "transformer" else "mlp"
                ),
            )
            rng, world_model_key = jax.random.split(rng)
            world_model_state = create_world_model_state(
                world_model_key,
                world_model_config,
            )
            (
                world_model_state,
                rng,
                world_model_prefit_loss,
                world_model_loss_history,
            ) = fit_world_model_steps(
                world_model_state,
                rng,
                prefit_batch,
                world_model_config,
                steps=cfg.wm_fit_steps,
            )
            loss_history = [float(value) for value in world_model_loss_history]
            logger.write_json(
                "world_model_prefit.json",
                {
                    "random_rollouts": cfg.wm_random_rollouts,
                    "initial_policy_rollouts": cfg.wm_initial_rollouts,
                    "policy_warmup_updates": cfg.wm_policy_warmup_updates,
                    "policy_warmup_real_env_steps": (
                        cfg.wm_policy_warmup_updates * cfg.num_envs * cfg.rollout_steps
                    ),
                    "random_real_env_steps": random_stats.real_env_steps,
                    "random_completed_episodes": random_stats.completed_episodes,
                    "random_episode_return_mean": random_stats.episode_return_mean,
                    "random_episode_length_mean": random_stats.episode_length_mean,
                    "initial_policy_real_env_steps": policy_stats.real_env_steps,
                    "initial_policy_completed_episodes": (
                        policy_stats.completed_episodes
                    ),
                    "initial_policy_episode_return_mean": (
                        policy_stats.episode_return_mean
                    ),
                    "initial_policy_episode_length_mean": (
                        policy_stats.episode_length_mean
                    ),
                    "prefit_real_env_steps": (
                        random_stats.real_env_steps + policy_stats.real_env_steps
                    ),
                    "prefit_completed_episodes": (
                        random_stats.completed_episodes
                        + policy_stats.completed_episodes
                    ),
                    "fit_steps": cfg.wm_fit_steps,
                    "transition_count": int(prefit_batch.states.shape[0]),
                    "loss": float(world_model_prefit_loss),
                    "loss_history": loss_history,
                    "config": dataclasses.asdict(world_model_config),
                },
            )
            logger.plot_world_model_loss(loss_history)
            observations = adapter.reset()

        updates = max(1, cfg.total_env_steps // (cfg.num_envs * cfg.rollout_steps))
        if cfg.prefit_world_model:
            if (
                world_model_state is None
                or world_model_config is None
                or model_start_states is None
                or reward_done_fn is None
                or world_model_prefit_loss is None
            ):
                raise RuntimeError("world model prefit did not initialize")
            train_state, rng, main_stacked = train_imagined_scan(
                world_model_state,
                train_state,
                model_start_states,
                rng,
                num_updates=updates,
                policy_config=config,
                world_model_config=world_model_config,
                rollout_steps=cfg.rollout_steps,
                reward_done_fn=reward_done_fn,
                num_envs=cfg.num_envs,
                algorithm=cfg.algorithm,
                freeze_policy=freeze_policy,
            )
            (
                main_rows,
                real_env_steps,
                imagined_env_steps,
                cumulative_real_episodes,
            ) = _rows_from_stacked(
                main_stacked,
                steps_per_update=cfg.num_envs * cfg.rollout_steps,
                real=False,
                real_env_steps=real_env_steps,
                imagined_env_steps=imagined_env_steps,
                cumulative_real_episodes=cumulative_real_episodes,
                extra_fields={
                    "control": control,
                    "world_model/prefit_loss": float(world_model_prefit_loss),
                },
            )
        elif scan_path:
            train_state, observations, rng, main_stacked = train_real_scan(
                adapter,
                train_state,
                observations,
                rng,
                num_updates=updates,
                config=config,
                rollout_steps=cfg.rollout_steps,
                algorithm=cfg.algorithm,
                freeze_policy=freeze_policy,
            )
            (
                main_rows,
                real_env_steps,
                imagined_env_steps,
                cumulative_real_episodes,
            ) = _rows_from_stacked(
                main_stacked,
                steps_per_update=cfg.num_envs * cfg.rollout_steps,
                real=True,
                real_env_steps=real_env_steps,
                imagined_env_steps=imagined_env_steps,
                cumulative_real_episodes=cumulative_real_episodes,
                extra_fields={"control": control},
            )
        else:
            if cfg.algorithm == "mappo":
                if not isinstance(config, MAPPOConfig):
                    raise TypeError("MAPPO updates require a MAPPOConfig")
                mappo_config = config
                update_fn = jax.jit(
                    lambda state, batch, last_values, update_rng: mappo_update(
                        state,
                        batch,
                        last_values,
                        update_rng,
                        mappo_config,
                    )
                )
                collect_fn = collect_mappo_rollout
            else:
                if not isinstance(config, IPPOConfig):
                    raise TypeError("IPPO updates require an IPPOConfig")
                ippo_config = config
                update_fn = jax.jit(
                    lambda state, batch, last_values, update_rng: ppo_update(
                        state,
                        batch,
                        last_values,
                        update_rng,
                        ippo_config,
                    )
                )
                collect_fn = collect_rollout
            main_rows = []
            env_steps = 0
            for update in range(1, updates + 1):
                rng, rollout_key, update_key = jax.random.split(rng, 3)
                rollout = _collect_real_env_rollout(
                    cfg,
                    collect_fn,
                    adapter,
                    train_state,
                    observations,
                    rollout_key,
                    config,
                    observation_mode,
                )
                observations = rollout.next_observations
                real_env_steps += cfg.num_envs * cfg.rollout_steps
                completed_real_episodes = int(
                    rollout.metrics.get("completed_episodes") or 0
                )
                cumulative_real_episodes += completed_real_episodes
                update_metrics: dict[str, Any] = {}
                if not freeze_policy:
                    train_state, update_metrics = update_fn(
                        train_state,
                        rollout.batch,
                        rollout.last_values,
                        update_key,
                    )
                env_steps += cfg.num_envs * cfg.rollout_steps
                main_rows.append(
                    {
                        "update": update,
                        "env_steps": env_steps,
                        "real_env_steps": real_env_steps,
                        "imagined_env_steps": imagined_env_steps,
                        "completed_real_episodes": completed_real_episodes,
                        "cumulative_real_episodes": cumulative_real_episodes,
                        "control": control,
                        **rollout.metrics,
                        **{
                            f"ppo/{key}": value for key, value in update_metrics.items()
                        },
                    }
                )
        for row in main_rows:
            rows.append(to_jsonable(row))
            logger.append_metrics(row)

        first_window_mean, final_window_mean = training_window_means(rows)
        logger.plot_returns(rows)

        checkpoint_dir = run_dir / "checkpoint"
        save_checkpoint(
            checkpoint_dir,
            train_state,
            metadata={
                "substrate": cfg.substrate,
                "num_envs": cfg.num_envs,
                "num_agents": adapter.num_agents,
                "observation_shape": adapter.observation_shape,
                "raw_observation_shape": adapter.raw_observation_shape,
                "observation_size": adapter.observation_size,
                "include_observation_scalars": adapter.include_observation_scalars,
                "scalar_observation_keys": adapter.scalar_observation_keys,
                "append_agent_id": adapter.append_agent_id,
                "algorithm": cfg.algorithm,
                "observation_mode": observation_mode,
                "central_observation_shape": (
                    central_observation_shape(
                        _policy_observation_shape(adapter, observation_mode),
                        adapter.num_agents,
                        observation_mode=observation_mode,
                    )
                    if cfg.algorithm == "mappo"
                    else None
                ),
                "action_dim": adapter.action_dim,
                "algorithm_config": dataclasses.asdict(config),
                "ippo_config": (
                    dataclasses.asdict(config) if cfg.algorithm == "ippo" else None
                ),
                "prefit_world_model": cfg.prefit_world_model,
                "world_model_config": (
                    dataclasses.asdict(world_model_config)
                    if world_model_config is not None
                    else None
                ),
                "seed": seed,
                "control": control,
            },
        )
    finally:
        adapter.close()

    reload_result = evaluate_checkpoint_subprocess(
        cfg,
        checkpoint_dir,
        seed=seed + 2,
    )
    logger.write_json("reload_evaluation.json", reload_result)
    timing = timer.to_dict()
    logger.write_json("timings.json", timing)

    random_mean = float(random_result["mean_return_per_agent"])
    initial_mean = float(initial_result["mean_return_per_agent"])
    trained_mean = float(reload_result["mean_return_per_agent"])
    random_improvement = trained_mean - random_mean
    initial_improvement = trained_mean - initial_mean
    outcome = RunOutcome(
        name=name,
        run_dir=str(run_dir),
        control=control,
        random_mean=random_mean,
        initial_mean=initial_mean,
        trained_mean=trained_mean,
        improvement=random_improvement,
        random_improvement=random_improvement,
        initial_improvement=initial_improvement,
        first_window_mean=first_window_mean,
        final_window_mean=final_window_mean,
        checkpoint_dir=str(checkpoint_dir),
        runtime_seconds=float(timing["runtime_seconds"]),
        real_env_steps=real_env_steps,
        imagined_env_steps=imagined_env_steps,
        cumulative_real_episodes=cumulative_real_episodes,
    )
    logger.write_json("outcome.json", outcome.to_dict())
    if wandb_run is not None:
        wandb_run.finish()
    return outcome


def summarize(
    outcomes: list[RunOutcome],
    control_outcome: RunOutcome | None,
    *,
    min_improvement: float,
) -> dict[str, Any]:
    improvements = np.asarray(
        [outcome.improvement for outcome in outcomes], dtype=float
    )
    initial_improvements = np.asarray(
        [outcome.initial_improvement for outcome in outcomes],
        dtype=float,
    )
    trained = np.asarray([outcome.trained_mean for outcome in outcomes], dtype=float)
    random = np.asarray([outcome.random_mean for outcome in outcomes], dtype=float)
    initial = np.asarray([outcome.initial_mean for outcome in outcomes], dtype=float)
    first_windows = np.asarray(
        [outcome.first_window_mean for outcome in outcomes],
        dtype=float,
    )
    final_windows = np.asarray(
        [outcome.final_window_mean for outcome in outcomes],
        dtype=float,
    )

    required_successes = max(1, math.ceil(len(outcomes) * 2 / 3))
    runs_beating_random = int(np.sum(improvements > 0.0))
    runs_beating_initial = int(np.sum(initial_improvements > 0.0))
    aggregate_improvement = float(trained.mean() - random.mean())
    aggregate_initial_improvement = float(trained.mean() - initial.mean())
    curve_improved = bool(final_windows.mean() > first_windows.mean())

    control_would_pass = False
    if control_outcome is not None:
        control_would_pass = bool(
            control_outcome.initial_improvement >= min_improvement
        )

    passed = bool(
        runs_beating_random >= required_successes
        and runs_beating_initial >= required_successes
        and aggregate_improvement >= min_improvement
        and aggregate_initial_improvement >= min_improvement
        and curve_improved
        and not control_would_pass
    )

    return {
        "passed": passed,
        "required_successes": required_successes,
        "runs_beating_random": runs_beating_random,
        "runs_beating_initial": runs_beating_initial,
        "aggregate_random_mean": float(random.mean()),
        "aggregate_initial_mean": float(initial.mean()),
        "aggregate_trained_mean": float(trained.mean()),
        "aggregate_improvement": aggregate_improvement,
        "aggregate_random_improvement": aggregate_improvement,
        "aggregate_initial_improvement": aggregate_initial_improvement,
        "min_improvement": min_improvement,
        "curve_first_window_mean": float(first_windows.mean()),
        "curve_final_window_mean": float(final_windows.mean()),
        "curve_improved": curve_improved,
        "control_would_pass": control_would_pass,
        "runs": [outcome.to_dict() for outcome in outcomes],
        "control": control_outcome.to_dict() if control_outcome else None,
    }


def append_progress(experiment_dir: Path, outcome: RunOutcome) -> None:
    row = {"completed_at": timestamp(), **outcome.to_dict()}
    with (experiment_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


def main() -> None:
    cfg = TrainConfig.from_namespace(parse_args())
    if cfg.eval_checkpoint:
        evaluate_checkpoint_mode(cfg)
        return

    experiment_dir = Path(cfg.out_dir) / f"e2e_{timestamp()}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for run_index in range(cfg.num_runs):
        outcome = run_training(
            cfg,
            run_dir=experiment_dir / f"run_{run_index:03d}",
            name=f"run_{run_index:03d}",
            run_index=run_index,
            control=None,
        )
        append_progress(experiment_dir, outcome)
        outcomes.append(outcome)

    control_outcome = None
    if cfg.negative_control != "none":
        control_outcome = run_training(
            cfg,
            run_dir=experiment_dir / f"control_{cfg.negative_control}",
            name=f"control_{cfg.negative_control}",
            run_index=cfg.num_runs,
            control=cfg.negative_control,
        )
        append_progress(experiment_dir, control_outcome)

    summary = summarize(
        outcomes,
        control_outcome,
        min_improvement=cfg.min_improvement,
    )
    RunLogger(experiment_dir).write_json("summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

# TODO:  Two timing approaches for e2e, one for fit-in-advance world model on data and then train, just focusing on env dynamics and agent effect on env seperately, one for fit dyna style.
