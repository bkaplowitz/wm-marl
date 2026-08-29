"""Canonical launch configuration for DreaMARL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dreamarl.baselines.dreamerv3.config import (
    absolute_path,
    default_dreamerv3_python,
    default_upstream_root,
)


PUBLIC_ALGORITHMS = ("final-dreamarl",)


def algorithm_config_profiles(algorithm: str) -> list[str]:
    """Return the canonical profiles for the only supported algorithm."""

    if algorithm == "final-dreamarl":
        return ["dreamarl_final"]
    raise ValueError(f"unsupported algorithm: {algorithm!r}")


def environment_config_profile(task: str, num_agents: int) -> str:
    """Resolve the observation/environment profile without instantiating the env."""

    if task.startswith("smac_"):
        return "smac_vector"
    raise ValueError("final-dreamarl supports SMAC tasks")


@dataclass(frozen=True, slots=True)
class DreaMARLRunSpec:
    """One reproducible final-DreaMARL training run."""

    experiment_dir: Path
    task: str
    num_agents: int
    algorithm: str = "final-dreamarl"
    seed: int = 0
    train_steps: int = 50_000
    platform: str = "cuda"
    infrastructure_root: Path = field(default_factory=default_upstream_root)
    python: Path = field(default_factory=default_dreamerv3_python)
    save_every_seconds: int | None = 1_800
    wandb_project: str | None = None
    wandb_entity: str | None = None
    curve_eval_interval: int = 0
    curve_eval_episodes: int | None = None
    curve_eval_envs: int | None = None
    curve_eval_seed_offset: int | None = None

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
            raise ValueError("final-dreamarl requires at least two agents")
        if self.train_steps < 1:
            raise ValueError("train_steps must be positive")
        if self.curve_eval_interval < 0:
            raise ValueError("curve_eval_interval must be non-negative")
        if self.platform not in {"cpu", "cuda", "tpu"}:
            raise ValueError(f"unsupported platform: {self.platform!r}")

        smac = self.task.startswith("smac_")
        if self.curve_eval_episodes is None:
            object.__setattr__(self, "curve_eval_episodes", 32 if smac else 20)
        if self.curve_eval_envs is None:
            object.__setattr__(self, "curve_eval_envs", 1 if smac else 4)
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
        return "final"

    @property
    def effective_train_ratio(self) -> float:
        return 1024.0

    @property
    def effective_replay_sampling(self) -> str:
        return "recent_world_uniform_behavior"

    @property
    def command(self) -> list[str]:
        outputs = ["jsonl", "scope"]
        if self.wandb_project:
            outputs.append("wandb")
        command = [
            str(self.python),
            "-m",
            "dreamarl.main",
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
            "--run.steps",
            str(self.train_steps),
            "--jax.platform",
            self.platform,
            "--logger.outputs",
            *outputs,
            "--logger.filter",
            (
                "score|return|length|fps|ratio|train/loss/|train/rand/|"
                "train/dyn_ent|train/rep_ent|train/adv|train/ent/|train/opt/|"
                "train/posterior_jepa/|train/dynamics_jepa/|"
                "train/ctde/|report/ctde/|train/critic/|"
                "train/reploss/critic/|central_critic/|"
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
                    "--run.curve_eval_interval",
                    str(self.curve_eval_interval),
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
        return command

    @property
    def ctde_manifest(self) -> dict[str, object] | None:
        return {
            "version": self.ctde_version,
            "rollout_steps": self.ctde_rollout_steps,
            "two_step_anchors": 0,
            "self_fed_training": False,
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
            "actor_units": 512,
            "actor_learning_rate": 1e-5,
            "actor_critic_start_step": 3000,
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
            "implementation": "first-party decoder-free DreaMARL",
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
            "recency_decay": (
                0.9998
                if self.effective_replay_sampling
                in {"recent", "recent_world_uniform_behavior"}
                else None
            ),
            "actor_objective": "score_function_reinforce",
            "optimizer_topology": self.optimizer_topology,
            "train_ratio": self.effective_train_ratio,
            "optimizer_updates_per_environment_step": (
                self.effective_train_ratio / (16 * 64)
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
