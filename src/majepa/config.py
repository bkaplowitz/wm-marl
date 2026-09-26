"""Locked launch specification for the reproduced MA-JEPA baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from majepa.runtime import absolute_path, infrastructure_root, runtime_python

PUBLIC_ALGORITHMS = ("ma-jepa",)


def algorithm_config_profiles(algorithm: str) -> list[str]:
    if algorithm != "ma-jepa":
        raise ValueError(f"unsupported algorithm: {algorithm!r}")
    return ["baseline"]


def environment_config_profile(task: str, num_agents: int) -> str:
    if not task.startswith("smac_"):
        raise ValueError("MA-JEPA supports SMAC tasks")
    if num_agents < 2:
        raise ValueError("MA-JEPA requires at least two agents")
    return "baseline"


@dataclass(frozen=True, slots=True)
class MAJEPARunSpec:
    """One run of the sole architecture and training configuration."""

    experiment_dir: Path
    task: str
    num_agents: int
    seed: int = 0
    train_steps: int = 50_000
    platform: str = "cuda"
    infrastructure_root: Path = field(default_factory=infrastructure_root)
    python: Path = field(default_factory=runtime_python)
    save_every_seconds: int | None = 5_000
    wandb_project: str | None = None
    wandb_entity: str | None = None
    curve_eval_interval: int = 5_000
    curve_eval_episodes: int = 32
    curve_eval_envs: int = 4
    curve_eval_seed_offset: int = 50_000
    algorithm: str = "ma-jepa"

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
        algorithm_config_profiles(self.algorithm)
        environment_config_profile(self.task, self.num_agents)
        if self.train_steps < 1:
            raise ValueError("train_steps must be positive")
        if self.platform not in {"cpu", "cuda", "tpu"}:
            raise ValueError(f"unsupported platform: {self.platform!r}")
        if self.curve_eval_interval < 0:
            raise ValueError("curve_eval_interval must be nonnegative")
        if min(self.curve_eval_episodes, self.curve_eval_envs) < 1:
            raise ValueError("evaluation episodes and environments must be positive")
        object.__setattr__(
            self,
            "curve_eval_envs",
            min(self.curve_eval_envs, self.curve_eval_episodes),
        )

    @property
    def logdir(self) -> Path:
        return self.experiment_dir / "run"

    @property
    def configs(self) -> list[str]:
        return ["baseline"]

    @property
    def environment_profile(self) -> str:
        return "baseline"

    @property
    def optimizer_topology(self) -> str:
        return "separated"

    @property
    def effective_train_ratio(self) -> float:
        return 128.0

    @property
    def effective_replay_sampling(self) -> str:
        return "50% recent + 50% uniform for independent WM and PPO views"

    @property
    def command(self) -> list[str]:
        outputs = ["jsonl", "scope"]
        if self.wandb_project:
            outputs.append("wandb")
        command = [
            str(self.python), "-m", "majepa.main",
            "--logdir", str(self.logdir),
            "--configs", "baseline",
            "--task", self.task,
            "--seed", str(self.seed),
            "--agent.num_agents", str(self.num_agents),
            "--run.steps", str(self.train_steps),
            "--run.curve_eval_interval", str(self.curve_eval_interval),
            "--run.curve_eval_eps", str(self.curve_eval_episodes),
            "--run.eval_envs", str(self.curve_eval_envs),
            "--run.curve_eval_seed_offset", str(self.curve_eval_seed_offset),
            "--jax.platform", self.platform,
            "--logger.outputs", *outputs,
        ]
        if self.save_every_seconds is not None:
            command.extend(["--run.save_every", str(self.save_every_seconds)])
        return command

    @property
    def ctde_manifest(self) -> dict[str, object]:
        return {
            "joint_predictor": "12-layer temporal attention, width 256",
            "local_history": "2-layer causal transformer, deter 4096",
            "categorical_latent": "32x64",
            "decoder": False,
            "imagined_action_mask": "local Bernoulli prediction",
            "local_outcome_heads": False,
            "actor": "3x512",
            "central_critic": "2-layer attention, width 256",
            "world_model_lr": 1e-4,
            "actor_lr": 3e-5,
            "critic_lr": 3e-5,
            "ppo_epochs": 5,
            "ppo_clip": 0.2,
            "entropy": 0.003,
            "imagination_horizon": 5,
            "self_fed_bptt": 2,
            "self_fed_horizons": [2, 4, 5],
            "multi_step_jepa_horizons": [1, 2, 4, 8],
            "action_margin_scale": 0.1,
            "posterior_alignment_scale": 0.05,
            "sigreg_scale": 0.05,
        }

    def to_dict(self) -> dict[str, object]:
        evaluation = {
            "policy_mode": "deterministic",
            "interval": self.curve_eval_interval,
            "episodes": self.curve_eval_episodes,
            "envs": self.curve_eval_envs,
            "seed_offset": self.curve_eval_seed_offset,
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
            "num_agents": self.num_agents,
            "environment_profile": self.environment_profile,
            "configs": self.configs,
            "platform": self.platform,
            "curve_eval_interval": self.curve_eval_interval,
            "curve_eval_episodes": self.curve_eval_episodes,
            "curve_eval_envs": self.curve_eval_envs,
            "curve_eval_seed_offset": self.curve_eval_seed_offset,
            "evaluation_protocol": evaluation,
            "policy_modules": ["encoder", "local history", "actor"],
            "ctde": self.ctde_manifest,
            "command": self.command,
        }
