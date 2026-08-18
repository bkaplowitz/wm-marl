"""Launch configuration for the single maintained DreaMARL algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dreamarl.baselines.dreamerv3.config import (
    absolute_path,
    default_dreamerv3_python,
    default_upstream_root,
)


@dataclass(frozen=True, slots=True)
class DreaMARLRunSpec:
    """Infrastructure and evaluation settings for canonical DreaMARL."""

    experiment_dir: Path
    task: str
    num_agents: int
    seed: int = 0
    train_steps: int = 50_000
    platform: str = "cuda"
    infrastructure_root: Path = field(default_factory=default_upstream_root)
    python: Path = field(default_factory=default_dreamerv3_python)
    save_every_seconds: int | None = 1_800
    wandb_project: str | None = None
    wandb_entity: str | None = None
    curve_eval_interval: int = 0
    curve_eval_episodes: int = 20
    curve_eval_seed_offset: int = 10_000
    curve_eval_policy_mode: str = "deterministic"
    imagination_starts: int = 0
    marl_stage: str = "b0"
    agent_jepa_local_grad_scale: float = 0.0
    agent_jepa_k0_scale: float = 0.1
    agent_jepa_future_scale: float = 1.0
    agent_jepa_future_set_scale: float = 1.0

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
        if self.num_agents < 1:
            raise ValueError("num_agents must be positive")
        if self.train_steps < 1:
            raise ValueError("train_steps must be positive")
        if self.curve_eval_interval < 0:
            raise ValueError("curve_eval_interval must be non-negative")
        if self.imagination_starts < 0:
            raise ValueError("imagination_starts must be non-negative")
        if self.marl_stage not in {"b0", "b1"}:
            raise ValueError(f"unsupported MARL stage: {self.marl_stage!r}")
        for name in (
            "agent_jepa_local_grad_scale",
            "agent_jepa_k0_scale",
            "agent_jepa_future_scale",
            "agent_jepa_future_set_scale",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def logdir(self) -> Path:
        return self.experiment_dir / "run"

    @property
    def configs(self) -> list[str]:
        if self.task.startswith("meltingpot_"):
            return ["meltingpot_vision"]
        if self.task.startswith("dmc_") and self.num_agents == 1:
            return ["dmc_vision"]
        raise ValueError("DreaMARL supports Melting Pot or singleton visual DMC tasks")

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
            "--agent.marl.stage",
            self.marl_stage,
            "--agent.marl.agent_jepa.local_grad_scale",
            str(self.agent_jepa_local_grad_scale),
            "--agent.marl.agent_jepa.k0_scale",
            str(self.agent_jepa_k0_scale),
            "--agent.marl.agent_jepa.future_scale",
            str(self.agent_jepa_future_scale),
            "--agent.marl.agent_jepa.future_set_scale",
            str(self.agent_jepa_future_set_scale),
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
                "train/agent_jepa/|report/agent_jepa/|"
                "report/world_model/|report/openloop/|eval/"
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
                    "--run.curve_eval_seed_offset",
                    str(self.curve_eval_seed_offset),
                    "--run.curve_eval_policy_mode",
                    (
                        "eval"
                        if self.curve_eval_policy_mode == "deterministic"
                        else "eval_sample"
                    ),
                ]
            )
        if self.imagination_starts:
            command.extend(["--agent.imag_last", str(self.imagination_starts)])
        return command

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation": "first-party decoder-free DreaMARL",
            "experiment_dir": str(self.experiment_dir),
            "logdir": str(self.logdir),
            "infrastructure_root": str(self.infrastructure_root),
            "python": str(self.python),
            "task": self.task,
            "seed": self.seed,
            "train_env_steps_budget": self.train_steps,
            "train_agent_steps_budget": self.train_steps * self.num_agents,
            "num_agents": self.num_agents,
            "agent_axis_native": True,
            "marl_stage": self.marl_stage,
            "marl_architecture": self._marl_architecture,
            "team_contract": "explicit [B,T,A] axes with shared local modules",
            "world_model": "parallel_transformer",
            "world_model_objective": "embedding",
            "embedding_target": "ema",
            "embedding_loss": "cosine",
            "posterior_jepa": True,
            "dynamics_jepa": True,
            "spatial_jepa": True,
            "spatial_mask_ratio": 0.5,
            "spatial_mask_topology": "fixed_count",
            "spatial_fill_value": 128,
            "posterior_jepa_scale": 2.0,
            "dynamics_jepa_scale": 2.0,
            "spatial_jepa_scale": 1.0,
            "sigreg": True,
            "sigreg_scale": 0.05,
            "sigreg_knots": 17,
            "sigreg_num_proj": 256,
            "sigreg_aggregation": "per_agent",
            "policy_information": "observation-local latent history",
            "action_masking": "observed online and learned for imagination",
            "replay_sampling": "uniform",
            "replay_context": 128,
            "execution": "strict decentralized parameter-shared actors",
            "imagination": "synchronized independent local rollouts",
            "executable_state_supervision": "locked local JEPA/latent losses",
            "training_state": self._training_state,
            "posterior_context": "history",
            "visual_encoder": "simple",
            "visual_resolution": 64,
            "critic": "parameter-shared observation-local critic",
            "agent_jepa_enabled": self.marl_stage == "b1" and self.num_agents > 1,
            "agent_jepa_horizon": (
                1
                if self.marl_stage == "b1"
                and self.num_agents > 1
                and self.agent_jepa_future_scale > 0.0
                else 0
            ),
            "agent_jepa_local_grad_scale": self.agent_jepa_local_grad_scale,
            "agent_jepa_k0_scale": self.agent_jepa_k0_scale,
            "agent_jepa_future_scale": self.agent_jepa_future_scale,
            "agent_jepa_future_set_scale": self.agent_jepa_future_set_scale,
            "team_slots": 8,
            "team_slot_width": 256,
            "team_teacher_rate": 0.01,
            "agent_mask_ratio": [0.25, 0.5],
            "agent_jepa_matching": (
                "mean-centered agent-relative balanced stop-gradient Sinkhorn"
            ),
            "agent_jepa_matching_temperature": 0.02,
            "agent_jepa_sinkhorn_iterations": 10,
            "agent_jepa_predicted_set_scale": 1.0,
            "agent_jepa_source_set_scale": 1.0,
            "agent_jepa_hidden_coverage_scale": 1.0,
            "team_slot_variance_scale": 0.1,
            "team_slot_decorrelation_scale": 0.1,
            "algorithm_components": self._algorithm_components,
            "platform": self.platform,
            "observation_mode": "local RGB vision",
            "meltingpot_reward_mode": (
                "collective" if self.task.startswith("meltingpot_") else "native"
            ),
            "environment_seed_mode": (
                "construction-time Lab2D seed stream; Shimmy reset(seed) is ignored"
                if self.task.startswith("meltingpot_")
                else "suite construction seed"
            ),
            "environment_reproducibility": (
                "construction_seed_controlled_not_trajectory_deterministic"
                if self.task.startswith("meltingpot_")
                else "suite_seed_semantics"
            ),
            "accelerator_memory_preallocation": False,
            "configs": self.configs,
            "save_every_seconds": self.save_every_seconds,
            "curve_eval_interval": self.curve_eval_interval,
            "curve_eval_episodes": self.curve_eval_episodes,
            "curve_eval_seed_offset": self.curve_eval_seed_offset,
            "curve_eval_policy_mode": self.curve_eval_policy_mode,
            "imagination_starts": self.imagination_starts,
            "actor_entropy": 3e-4,
            "behavior_objective": "reinforce",
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "command": self.command,
        }

    @property
    def _algorithm_components(self) -> list[str]:
        components = [
            "DreamerV3 convolutional encoder",
            "history-conditioned observation posterior",
            "causal Transformer temporal dynamics",
            (
                "decoder-free posterior, action-conditioned dynamics, and "
                "fixed-count masked-spatial EMA-target cosine prediction"
            ),
            "SIGReg embedding anti-collapse regularization",
            "128-step loss-excluded Transformer replay burn-in",
            "uniform replay",
            "explicit environment, time, and agent replay axes",
            "parameter-shared local world model, actor, and critic",
            "synchronized independent local imagination",
        ]
        if self.num_agents > 1:
            components.append("strict decentralized execution with no peer tensors")
            if self.marl_stage == "b1":
                if self.agent_jepa_future_scale > 0.0:
                    components.append(
                        "training-only whole-agent-masked current and joint-action-"
                        "conditioned one-step future team EMA-slot JEPA"
                    )
                else:
                    components.append(
                        "training-only detached whole-agent-masked current team "
                        "EMA-slot JEPA control"
                    )
        else:
            components.append("locked singleton execution")
        return components

    @property
    def _marl_architecture(self) -> str:
        if self.marl_stage == "b1" and self.num_agents > 1:
            if self.agent_jepa_future_scale > 0.0:
                return (
                    "shared local JEPA plus detached current and joint-action-"
                    "conditioned future team EMA-slot JEPA"
                )
            return "shared local JEPA plus detached current team EMA-slot JEPA"
        return "shared independent local JEPA"

    @property
    def _training_state(self) -> str:
        return (
            "history-conditioned categorical posterior with a strict-causal "
            "Transformer over local latent-action histories"
        )
