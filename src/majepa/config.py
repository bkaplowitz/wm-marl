"""Canonical launch configuration for MA-JEPA."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

from majepa.runtime import absolute_path, infrastructure_root, runtime_python


PUBLIC_ALGORITHMS = ("ma-jepa",)
DUAL_VIEW_REPLAY_SAMPLINGS = frozenset(
    {
        "recent_world_uniform_behavior",
        "truncated_geometric_world_uniform_behavior",
    }
)


def algorithm_config_profiles(algorithm: str) -> list[str]:
    """Return the canonical profiles for the only supported algorithm."""

    if algorithm == "ma-jepa":
        return ["ma_jepa"]
    raise ValueError(f"unsupported algorithm: {algorithm!r}")


def environment_config_profile(task: str, num_agents: int) -> str:
    """Resolve the observation/environment profile without instantiating the env."""

    if task.startswith("smac_"):
        return "smac_vector"
    raise ValueError("MA-JEPA supports SMAC tasks")


@dataclass(frozen=True, slots=True)
class MAJEPARunSpec:
    """One reproducible MA-JEPA training run."""

    experiment_dir: Path
    task: str
    num_agents: int
    algorithm: str = "ma-jepa"
    seed: int = 0
    train_steps: int = 50_000
    platform: str = "cuda"
    infrastructure_root: Path = field(default_factory=infrastructure_root)
    python: Path = field(default_factory=runtime_python)
    save_every_seconds: int | None = 900
    wandb_project: str | None = None
    wandb_entity: str | None = None
    curve_eval_interval: int = 5_000
    curve_eval_episodes: int | None = None
    curve_eval_envs: int | None = None
    curve_eval_seed_offset: int | None = None
    imag_action_samples: int = 1
    replay_sampling: str = "recent_world_uniform_behavior"
    recency_decay: float = 0.9998
    world_uniform_mix: float = 0.5
    truncated_geometric_alpha: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_dir", Path(self.experiment_dir).expanduser().resolve()
        )
        object.__setattr__(
            self,
            "infrastructure_root",
            Path(self.infrastructure_root).expanduser().resolve(),
        )
        object.__setattr__(self, "python", absolute_path(self.python))
        if self.algorithm not in PUBLIC_ALGORITHMS:
            raise ValueError(f"unsupported algorithm: {self.algorithm!r}")
        if self.num_agents < 1:
            raise ValueError("num_agents must be positive")
        if self.num_agents < 2:
            raise ValueError("MA-JEPA requires at least two agents")
        if type(
            self.imag_action_samples
        ) is not int or self.imag_action_samples not in (1, 2):
            raise ValueError("imag_action_samples must be 1 or 2")
        if self.train_steps < 1:
            raise ValueError("train_steps must be positive")
        if self.curve_eval_interval < 0:
            raise ValueError("curve_eval_interval must be non-negative")
        if self.platform not in {"cpu", "cuda", "tpu"}:
            raise ValueError(f"unsupported platform: {self.platform!r}")
        if self.replay_sampling not in DUAL_VIEW_REPLAY_SAMPLINGS:
            raise ValueError(
                "MA-JEPA requires a supported dual-view replay sampler, got "
                f"{self.replay_sampling!r}"
            )
        if not 0.0 < float(self.recency_decay) <= 1.0:
            raise ValueError("recency_decay must be in (0, 1]")
        if not 0.0 <= float(self.world_uniform_mix) <= 1.0:
            raise ValueError("world_uniform_mix must be in [0, 1]")
        if not float(self.truncated_geometric_alpha) >= 0.0 or not math.isfinite(
            float(self.truncated_geometric_alpha)
        ):
            raise ValueError("truncated_geometric_alpha must be finite and nonnegative")

        smac = self.task.startswith("smac_")
        if self.curve_eval_episodes is None:
            object.__setattr__(self, "curve_eval_episodes", 32 if smac else 20)
        if self.curve_eval_envs is None:
            object.__setattr__(self, "curve_eval_envs", 4)
        if self.curve_eval_seed_offset is None:
            object.__setattr__(
                self, "curve_eval_seed_offset", 50_000 if smac else 10_000
            )
        if int(self.curve_eval_episodes) < 1:
            raise ValueError("curve_eval_episodes must be positive")
        if int(self.curve_eval_envs) < 1:
            raise ValueError("curve_eval_envs must be positive")
        if int(self.curve_eval_seed_offset) < 0:
            raise ValueError("curve_eval_seed_offset must be non-negative")
        object.__setattr__(
            self,
            "curve_eval_envs",
            min(int(self.curve_eval_envs), int(self.curve_eval_episodes)),
        )

        environment_config_profile(self.task, self.num_agents)

    @property
    def logdir(self) -> Path:
        return self.experiment_dir / "run"

    @property
    def environment_profile(self) -> str:
        return environment_config_profile(self.task, self.num_agents)

    @property
    def configs(self) -> list[str]:
        return [self.environment_profile, *algorithm_config_profiles(self.algorithm)]

    @property
    def marl_stage(self) -> str:
        return "ctde"

    @property
    def optimizer_topology(self) -> str:
        return "separated"

    @property
    def ctde_rollout_steps(self) -> int | None:
        return 1

    @property
    def ctde_version(self) -> str | None:
        return "ma-jepa"

    @property
    def effective_train_ratio(self) -> float:
        return 128.0

    @property
    def effective_replay_sampling(self) -> str:
        return self.replay_sampling

    @property
    def command(self) -> list[str]:
        outputs = ["jsonl", "scope"]
        if self.wandb_project:
            outputs.append("wandb")
        command = [
            str(self.python),
            "-m",
            "majepa.main",
            "--logdir",
            str(self.logdir),
            "--configs",
            *self.configs,
            "--task",
            self.task,
            "--seed",
            str(self.seed),
            "--agent.num_agents",
            str(self.num_agents),
            "--agent.imag_action_samples",
            str(self.imag_action_samples),
            "--run.steps",
            str(self.train_steps),
            "--run.curve_eval_interval",
            str(self.curve_eval_interval),
            "--jax.platform",
            self.platform,
            "--logger.outputs",
            *outputs,
            "--logger.filter",
            (
                "score|return|length|fps|ratio|train/loss/|train/rand/|"
                "schedule/|train/ppo/|train/opt/|"
                "train/posterior_jepa/|train/dynamics_jepa/|"
                "train/ctde/|report/ctde/|central_critic/|"
                "report/world_model/|report/openloop/|battle_won|win_rate|"
                "legacy_|corrected_|enemy_|ally_|timeout|action_|"
                "attack_target_|eval/"
            ),
        ]
        if self.save_every_seconds is not None:
            command.extend(["--run.save_every", str(self.save_every_seconds)])
        if self.curve_eval_interval:
            command.extend(
                [
                    "--run.curve_eval_eps",
                    str(self.curve_eval_episodes),
                    "--run.eval_envs",
                    str(self.curve_eval_envs),
                    "--run.curve_eval_seed_offset",
                    str(self.curve_eval_seed_offset),
                    "--run.curve_eval_policy_mode",
                    "eval",
                ]
            )
        command.extend(
            [
                "--replay.sampling",
                self.replay_sampling,
                "--replay.recency_decay",
                str(self.recency_decay),
                "--replay.world_uniform_mix",
                str(self.world_uniform_mix),
                "--replay.truncated_geometric_alpha",
                str(self.truncated_geometric_alpha),
            ]
        )
        return command

    @property
    def ctde_manifest(self) -> dict[str, object] | None:
        return {
            "version": self.ctde_version,
            "rollout_steps": self.ctde_rollout_steps,
            "two_step_anchors": 0,
            "self_fed_training": True,
            "self_fed_bptt_steps": 2,
            "self_fed_horizons": [2, 4, 5],
            "self_fed_anchors": 8,
            "self_fed_scale": 0.1,
            "trajectory_kl_scale": 0.1,
            "fresh_history": False,
            "imagination_mask_sampling": "bernoulli",
            "agent_attention": {"width": 256, "layers": 2, "heads": 4},
            "temporal_transformer": {
                "width": 256,
                "layers": 12,
                "heads": 4,
                "context": 16,
            },
            "optimizer_groups": ["local_world", "joint_world", "actor", "critic"],
            "learning_rate": 4e-5,
            "teammate_belief": True,
            "multi_step_jepa": True,
            "role_aware_peer_plan": True,
            "actor_units": 1024,
            "actor_learning_rate": 3e-5,
            "critic_learning_rate": 3e-5,
            "world_model_start_step": 5000,
            "world_optimizer_warmup": 0,
            "ppo_start_step": 5000,
            "imagination_horizon": 5,
            "imag_action_samples": self.imag_action_samples,
            "replay_sampling": self.effective_replay_sampling,
            "recency_decay": self.recency_decay,
            "world_uniform_mix": self.world_uniform_mix,
            "truncated_geometric_alpha": (
                self.truncated_geometric_alpha
                if self.effective_replay_sampling
                == "truncated_geometric_world_uniform_behavior"
                else None
            ),
            "ppo": {
                "epochs": 5,
                "clip_epsilon": 0.2,
                "entropy_coefficient": 1e-2,
                "lambda": 0.95,
                "target_critic_rate": 1.0,
                "world_update_before_imagination": True,
                "frozen_imagined_batch": True,
                "shared_team_returns_after_death": True,
                "replay_value_scale": 0.3,
                "replay_value_lambda": 0.95,
                "factual_value": False,
                "factual_representation_scale": 0.0,
            },
            "action_counterfactuals": "all_legal_mean",
            "action_counterfactual_scale": 0.25,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the single authoritative launch manifest payload."""

        evaluation = {
            "policy_mode": "deterministic",
            "interval": self.curve_eval_interval,
            "episodes": int(self.curve_eval_episodes),
            "envs": int(self.curve_eval_envs),
            "seed_offset": int(self.curve_eval_seed_offset),
        }
        return {
            "implementation": "MA-JEPA",
            "algorithm": self.algorithm,
            "experiment_dir": str(self.experiment_dir),
            "logdir": str(self.logdir),
            "infrastructure_root": str(self.infrastructure_root),
            "python": str(self.python),
            "task": self.task,
            "seed": self.seed,
            "train_env_steps_budget": self.train_steps,
            "train_agent_steps_budget": self.train_steps * self.num_agents,
            "num_agents": self.num_agents,
            "environment_profile": self.environment_profile,
            "configs": self.configs,
            "marl_stage": self.marl_stage,
            "ctde_version": self.ctde_version,
            "ctde_rollout_steps": self.ctde_rollout_steps,
            "ctde": self.ctde_manifest,
            "world_model": "parallel_transformer",
            "world_model_objective": "embedding",
            "embedding_target": "ema",
            "embedding_loss": "cosine",
            "posterior_jepa": True,
            "dynamics_jepa": True,
            "spatial_jepa": False,
            "spatial_mask_ratio": 0.5,
            "sigreg": True,
            "sigreg_aggregation": "per_agent",
            "replay_context": 192,
            "replay_sampling": self.effective_replay_sampling,
            "world_uniform_mix": self.world_uniform_mix,
            "isolate_report_rng": True,
            "development_reference": "am1-bernoulli-20260907",
            "recency_decay": self.recency_decay,
            "truncated_geometric_alpha": (
                self.truncated_geometric_alpha
                if self.effective_replay_sampling
                == "truncated_geometric_world_uniform_behavior"
                else None
            ),
            "actor_objective": "clipped_imagined_ppo",
            "optimizer_topology": self.optimizer_topology,
            "train_ratio": self.effective_train_ratio,
            "learner_batches_per_environment_step": (
                self.effective_train_ratio / (16 * 64)
            ),
            "actor_updates_per_environment_step": (
                5 * self.effective_train_ratio / (16 * 64)
            ),
            "critic_updates_per_environment_step": (
                5 * self.effective_train_ratio / (16 * 64)
            ),
            "execution": "strict decentralized parameter-shared actors",
            "policy_information": "one observation-local latent history per agent",
            "policy_peer_access": False,
            "policy_modules": [
                "enc",
                "dyn",
                "pol",
                "ctde_teammate_belief",
                "ctde_teammate_actor",
            ],
            "evaluation_protocol": evaluation,
            "platform": self.platform,
            "save_every_seconds": self.save_every_seconds,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "command": self.command,
        }
