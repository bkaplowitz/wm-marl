"""Configuration surface for reproducible DreaMARL ablations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dreamarl.baselines.dreamerv3.config import (
    absolute_path,
    default_dreamerv3_python,
    default_upstream_root,
)


@dataclass(frozen=True, slots=True)
class AblationRunSpec:
    """One explicit non-canonical DreaMARL ablation."""

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
    temporal_model: str = "parallel_transformer"
    world_model_objective: str = "embedding"
    embedding_target: str = "ema"
    embedding_loss: str = "cosine"
    posterior_jepa: bool = True
    dynamics_jepa: bool = True
    spatial_jepa: bool = True
    spatial_mask_ratio: float = 0.5
    spatial_mask_topology: str = "fixed_count"
    spatial_fill_value: int = 128
    posterior_jepa_scale: float = 2.0
    dynamics_jepa_scale: float = 2.0
    spatial_jepa_scale: float = 1.0
    sigreg: bool = True
    sigreg_scale: float = 0.05
    sigreg_knots: int = 17
    sigreg_num_proj: int = 256
    sigreg_aggregation: str = "pooled"
    posterior_context: str = "history"
    visual_encoder: str = "simple"
    curve_eval_interval: int = 0
    curve_eval_episodes: int = 20
    curve_eval_seed_offset: int = 10_000
    curve_eval_policy_mode: str = "deterministic"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_dir",
            Path(self.experiment_dir).expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "infrastructure_root",
            Path(self.infrastructure_root).expanduser().resolve(),
        )
        object.__setattr__(self, "python", absolute_path(self.python))
        if self.num_agents < 1:
            raise ValueError("num_agents must be positive")
        if not self.task:
            raise ValueError("task must be non-empty")
        if self.train_steps < 1:
            raise ValueError("train_steps must be positive")
        if self.platform not in {"cpu", "cuda", "tpu"}:
            raise ValueError("platform must be one of: cpu, cuda, tpu")
        if self.save_every_seconds is not None and self.save_every_seconds < 1:
            raise ValueError("save_every_seconds must be positive")
        if self.temporal_model not in {"rssm", "parallel_transformer"}:
            raise ValueError(
                "temporal_model must be either rssm or parallel_transformer"
            )
        if self.world_model_objective not in {"reconstruction", "embedding"}:
            raise ValueError(
                "world_model_objective must be reconstruction or embedding"
            )
        if self.embedding_target not in {"ema", "online"}:
            raise ValueError("embedding_target must be ema or online")
        if self.embedding_loss not in {"cosine", "mse"}:
            raise ValueError("embedding_loss must be cosine or mse")
        if self.world_model_objective == "embedding" and not (
            self.posterior_jepa or self.dynamics_jepa or self.spatial_jepa
        ):
            raise ValueError(
                "embedding training requires at least one explicit JEPA objective"
            )
        if not 0.0 < self.spatial_mask_ratio < 1.0:
            raise ValueError("spatial_mask_ratio must be between zero and one")
        if self.spatial_mask_topology not in {
            "bernoulli",
            "fixed_count",
            "multiblock",
            "vjepa_multiblock",
        }:
            raise ValueError(
                "spatial_mask_topology must be bernoulli, fixed_count, or "
                "multiblock, or vjepa_multiblock"
            )
        if not 0 <= self.spatial_fill_value <= 255:
            raise ValueError("spatial_fill_value must fit in an uint8 image")
        if (
            min(
                self.posterior_jepa_scale,
                self.dynamics_jepa_scale,
                self.spatial_jepa_scale,
                self.sigreg_scale,
            )
            <= 0
        ):
            raise ValueError("representation loss scales must be positive")
        if self.sigreg_knots < 2:
            raise ValueError("sigreg_knots must be at least 2")
        if self.sigreg_num_proj < 1:
            raise ValueError("sigreg_num_proj must be positive")
        if self.sigreg_aggregation not in {"pooled", "per_timestep"}:
            raise ValueError("sigreg_aggregation must be pooled or per_timestep")
        if self.embedding_target == "online" and self.spatial_jepa:
            raise ValueError(
                "online embedding targets are incompatible with spatial masking"
            )
        if self.posterior_context not in {"observation", "history"}:
            raise ValueError("posterior_context must be either observation or history")
        if self.visual_encoder not in {"simple", "vit", "vjepa"}:
            raise ValueError("visual_encoder must be simple, vit, or vjepa")
        if (self.visual_encoder == "vjepa") != (
            self.spatial_mask_topology == "vjepa_multiblock"
        ):
            raise ValueError(
                "the vjepa encoder and vjepa_multiblock topology must be "
                "enabled together"
            )
        if self.visual_encoder == "vjepa" and (
            self.world_model_objective != "embedding"
            or self.embedding_target != "ema"
            or not self.spatial_jepa
        ):
            raise ValueError(
                "the faithful V-JEPA variant requires decoder-free embedding "
                "training, EMA targets, and spatial JEPA"
            )
        if self.curve_eval_interval < 0:
            raise ValueError("curve_eval_interval must be non-negative")
        if self.curve_eval_episodes < 1:
            raise ValueError("curve_eval_episodes must be positive")
        if self.curve_eval_seed_offset < 0:
            raise ValueError("curve_eval_seed_offset must be non-negative")
        if self.curve_eval_policy_mode not in {"deterministic", "stochastic"}:
            raise ValueError(
                "curve_eval_policy_mode must be deterministic or stochastic"
            )

    @property
    def logdir(self) -> Path:
        return self.experiment_dir / "run"

    @property
    def configs(self) -> list[str]:
        if self.task.startswith("meltingpot_"):
            return ["meltingpot_vision"]
        if self.task.startswith("dmc_"):
            if self.num_agents != 1:
                raise ValueError("visual DMC tasks require num_agents=1")
            return ["dmc_vision"]
        else:
            raise ValueError(
                "maintained launches require a Melting Pot or visual DMC task"
            )

    @property
    def command(self) -> list[str]:
        outputs = ["jsonl", "scope"]
        if self.wandb_project:
            outputs.append("wandb")
        command = [
            str(self.python),
            "-m",
            "dreamarl.ablations.main",
            "--logdir",
            str(self.logdir),
            "--configs",
            *self.configs,
            "ablation_components",
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
            "--agent.dyn.typ",
            self.temporal_model,
            "--agent.enc.typ",
            self.visual_encoder,
            "--agent.objective",
            self.world_model_objective,
            "--agent.embedding_target",
            self.embedding_target,
            "--agent.embedding_loss",
            self.embedding_loss,
            "--agent.posterior_jepa",
            str(self.posterior_jepa),
            "--agent.dynamics_jepa",
            str(self.dynamics_jepa),
            "--agent.spatial_jepa.enabled",
            str(self.spatial_jepa),
            "--agent.spatial_jepa.mask_ratio",
            str(self.spatial_mask_ratio),
            "--agent.spatial_jepa.topology",
            self.spatial_mask_topology,
            "--agent.spatial_jepa.fill_value",
            str(self.spatial_fill_value),
            "--agent.loss_scales.posterior_jepa",
            str(self.posterior_jepa_scale),
            "--agent.loss_scales.dynamics_jepa",
            str(self.dynamics_jepa_scale),
            "--agent.loss_scales.spatial_jepa",
            str(self.spatial_jepa_scale),
            "--agent.sigreg.enabled",
            str(self.sigreg),
            "--agent.sigreg.knots",
            str(self.sigreg_knots),
            "--agent.sigreg.num_proj",
            str(self.sigreg_num_proj),
            "--agent.sigreg.aggregation",
            self.sigreg_aggregation,
            "--agent.loss_scales.sigreg",
            str(self.sigreg_scale),
            "--logger.outputs",
            *outputs,
            "--logger.filter",
            (
                "score|length|fps|ratio|train/loss/|train/rand/|"
                "train/dyn_ent|train/rep_ent|"
                "report/world_model/|eval/"
            ),
        ]
        if self.temporal_model == "parallel_transformer":
            command.extend(
                [
                    "--agent.dyn.parallel_transformer.posterior_context",
                    self.posterior_context,
                ]
            )
        if self.visual_encoder == "vjepa":
            environment = "dmc" if self.task.startswith("dmc_") else "meltingpot"
            command.extend([f"--env.{environment}.size", "224", "224"])
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
            "num_agents": self.num_agents,
            "agent_axis_native": True,
            "world_model": self.temporal_model,
            "world_model_objective": self.world_model_objective,
            "embedding_target": self.embedding_target,
            "embedding_loss": self.embedding_loss,
            "posterior_jepa": self.posterior_jepa,
            "dynamics_jepa": self.dynamics_jepa,
            "spatial_jepa": self.spatial_jepa,
            "spatial_mask_ratio": self.spatial_mask_ratio,
            "spatial_mask_topology": self.spatial_mask_topology,
            "spatial_mask_recipe": (
                {
                    "grid": [14, 14],
                    "patch": 16,
                    "frames_per_replay_sequence": 4,
                    "tube_consistent": True,
                    "groups": [
                        {"blocks": 8, "scale": 0.15, "aspect": [0.75, 1.5]},
                        {"blocks": 2, "scale": 0.7, "aspect": [0.75, 1.5]},
                    ],
                    "context": "visible tokens only",
                    "target": "full EMA encoder then output-token selection",
                    "predictor": "12-layer 384-wide positional mask-token transformer",
                    "loss": "per-token normalized L1",
                }
                if self.visual_encoder == "vjepa"
                else None
            ),
            "spatial_fill_value": self.spatial_fill_value,
            "posterior_jepa_scale": self.posterior_jepa_scale,
            "dynamics_jepa_scale": self.dynamics_jepa_scale,
            "spatial_jepa_scale": self.spatial_jepa_scale,
            "sigreg": self.sigreg,
            "sigreg_scale": self.sigreg_scale,
            "sigreg_knots": self.sigreg_knots,
            "sigreg_num_proj": self.sigreg_num_proj,
            "sigreg_aggregation": self.sigreg_aggregation,
            "replay_sampling": "uniform",
            "execution": "shared decentralized actor over local latent state",
            "training_state": self._training_state,
            "posterior_context": (
                self.posterior_context
                if self.temporal_model == "parallel_transformer"
                else "history"
            ),
            "visual_encoder": self.visual_encoder,
            "visual_resolution": 224 if self.visual_encoder == "vjepa" else 64,
            "critic": "DreamerV3 latent value model",
            "algorithm_components": self._algorithm_components,
            "platform": self.platform,
            "observation_mode": "local RGB vision",
            "accelerator_memory_preallocation": False,
            "configs": [*self.configs, "ablation_components"],
            "save_every_seconds": self.save_every_seconds,
            "curve_eval_interval": self.curve_eval_interval,
            "curve_eval_episodes": self.curve_eval_episodes,
            "curve_eval_seed_offset": self.curve_eval_seed_offset,
            "curve_eval_policy_mode": self.curve_eval_policy_mode,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "command": self.command,
        }

    @property
    def _posterior_description(self) -> str:
        if self.temporal_model == "rssm" or self.posterior_context == "history":
            return "history-conditioned observation posterior"
        return "parallel observation posterior"

    @property
    def _temporal_description(self) -> str:
        if self.temporal_model == "rssm":
            return "block-GRU RSSM temporal dynamics"
        return "causal Transformer temporal dynamics"

    @property
    def _objective_description(self) -> str:
        if self.visual_encoder == "vjepa":
            return (
                "decoder-free posterior and action-conditioned EMA-target "
                "cosine prediction with V-JEPA positional masked-token L1 "
                "prediction"
            )
        objectives = []
        if self.posterior_jepa:
            objectives.append("posterior")
        if self.dynamics_jepa:
            objectives.append("action-conditioned dynamics")
        if self.spatial_jepa:
            objectives.append("masked-spatial")
        if not objectives:
            return "DreamerV3 reconstruction attribution control"
        joined = (
            objectives[0]
            if len(objectives) == 1
            else ", ".join(objectives[:-1]) + f", and {objectives[-1]}"
        )
        prefix = (
            "DreamerV3 reconstruction with auxiliary"
            if self.world_model_objective == "reconstruction"
            else "decoder-free"
        )
        target = (
            "EMA-target cosine"
            if self.embedding_target == "ema"
            else "full-gradient online-target MSE"
        )
        return f"{prefix} {joined} {target} joint-embedding prediction"

    @property
    def _algorithm_components(self) -> list[str]:
        components = [
            (
                "224px patch-16 ViT on a 14x14 token grid"
                if self.visual_encoder == "vjepa"
                else (
                    "compact spatial ViT encoder"
                    if self.visual_encoder == "vit"
                    else "DreamerV3 convolutional encoder"
                )
            ),
            self._posterior_description,
            self._temporal_description,
            self._objective_description,
        ]
        if self.sigreg:
            components.append("SIGReg embedding anti-collapse regularization")
        components.extend(
            [
                "one-step recurrent replay context",
                "uniform replay",
            ]
        )
        return components

    @property
    def _training_state(self) -> str:
        if self.temporal_model == "rssm":
            return (
                "history-conditioned categorical posterior with DreamerV3 "
                "block-GRU RSSM dynamics"
            )
        qualifier = (
            "history-conditioned"
            if self.posterior_context == "history"
            else "observation-parallel"
        )
        return (
            f"{qualifier} categorical posterior with a strict-causal "
            "Transformer over local latent-action histories"
        )
