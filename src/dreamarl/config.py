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
    train_ratio: float = 256.0
    replay_sampling: str = "uniform"
    behavior_optimizer: str = "joint"
    behavior_objective: str = "reinforce"
    ppo_epochs: int = 3
    ppo_clip: float = 0.2
    repval_grad: bool = True
    anchor_batch: Path | None = None
    from_checkpoint: Path | None = None
    from_checkpoint_regex: str | None = None
    load_replay: bool = False
    replay_source: Path | None = None
    report_actor_gradnorm: bool = False
    marl_stage: str = "b0"
    agent_jepa_local_grad_scale: float = 0.0
    agent_jepa_k0_scale: float = 0.1
    agent_jepa_future_scale: float = 1.0
    agent_jepa_future_set_scale: float = 1.0
    ctde_rollout_steps: int = 1
    ctde_multistep_anchors: int = 128

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
        if self.anchor_batch is not None:
            object.__setattr__(
                self, "anchor_batch", Path(self.anchor_batch).expanduser().resolve()
            )
        if self.from_checkpoint is not None:
            object.__setattr__(
                self,
                "from_checkpoint",
                Path(self.from_checkpoint).expanduser().resolve(),
            )
        if self.replay_source is not None:
            object.__setattr__(
                self,
                "replay_source",
                Path(self.replay_source).expanduser().resolve(),
            )
        if self.from_checkpoint_regex is None:
            object.__setattr__(
                self,
                "from_checkpoint_regex",
                (
                    r"^(?!(?:opt|jecc_opt|actor_opt)/).*"
                    if self.marl_stage == "jecc"
                    else ".*"
                ),
            )
        if self.num_agents < 1:
            raise ValueError("num_agents must be positive")
        if self.train_steps < 1:
            raise ValueError("train_steps must be positive")
        if self.curve_eval_interval < 0:
            raise ValueError("curve_eval_interval must be non-negative")
        if self.imagination_starts < 0:
            raise ValueError("imagination_starts must be non-negative")
        if self.train_ratio <= 0:
            raise ValueError("train_ratio must be positive")
        if self.marl_stage not in {"b0", "b1", "b2", "jecc", "ctde"}:
            raise ValueError(f"unsupported MARL stage: {self.marl_stage!r}")
        if self.marl_stage == "ctde" and self.num_agents < 2:
            raise ValueError("CTDE requires at least two agents")
        if self.ctde_rollout_steps not in {1, 2}:
            raise ValueError("ctde_rollout_steps must be 1 or 2")
        if self.ctde_multistep_anchors < 1:
            raise ValueError("ctde_multistep_anchors must be positive")
        if self.replay_sampling not in {"uniform", "recent"}:
            raise ValueError(f"unsupported replay sampling: {self.replay_sampling!r}")
        if self.behavior_optimizer not in {"joint", "separated", "grouped"}:
            raise ValueError(
                f"unsupported behavior optimizer: {self.behavior_optimizer!r}"
            )
        if self.behavior_objective not in {"reinforce", "ppo"}:
            raise ValueError(
                f"unsupported behavior objective: {self.behavior_objective!r}"
            )
        if self.behavior_objective == "ppo" and self.behavior_optimizer == "joint":
            raise ValueError("imagined PPO requires separated optimizer ownership")
        if self.marl_stage == "jecc" and (
            self.behavior_optimizer != "separated"
            or self.behavior_objective != "reinforce"
        ):
            raise ValueError("JECC requires separated REINFORCE behavior learning")
        if self.marl_stage == "ctde" and (
            self.behavior_optimizer != "separated"
            or self.behavior_objective != "reinforce"
        ):
            raise ValueError("CTDE requires separated REINFORCE behavior learning")
        if self.ppo_epochs < 1:
            raise ValueError("ppo_epochs must be positive")
        if not 0.0 < self.ppo_clip < 1.0:
            raise ValueError("ppo_clip must lie in (0, 1)")
        if self.load_replay and self.from_checkpoint is None:
            raise ValueError("load_replay requires a continuation checkpoint")
        if self.load_replay and self.replay_source is None:
            raise ValueError("load_replay requires replay_source")
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
        if self.task.startswith("smac_"):
            return ["smac_vector"]
        raise ValueError(
            "DreaMARL supports Melting Pot, SMAC, or singleton visual DMC tasks"
        )

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
            "--agent.marl.ctde.rollout_steps",
            str(self.ctde_rollout_steps),
            "--agent.marl.ctde.multistep.anchors",
            str(self.ctde_multistep_anchors),
            "--agent.marl.agent_jepa.local_grad_scale",
            str(self.agent_jepa_local_grad_scale),
            "--agent.marl.agent_jepa.k0_scale",
            str(self.agent_jepa_k0_scale),
            "--agent.marl.agent_jepa.future_scale",
            str(self.agent_jepa_future_scale),
            "--agent.marl.agent_jepa.future_set_scale",
            str(self.agent_jepa_future_set_scale),
            "--replay.sampling",
            self.replay_sampling,
            "--agent.behavior_optimizer",
            self.behavior_optimizer,
            "--agent.behavior_objective",
            self.behavior_objective,
            "--agent.ppo.epochs",
            str(self.ppo_epochs),
            "--agent.ppo.clip",
            str(self.ppo_clip),
            "--agent.repval_grad",
            str(self.repval_grad),
            "--agent.report_actor_gradnorm",
            str(self.report_actor_gradnorm),
            "--run.train_ratio",
            str(self.train_ratio),
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
                "train/ctde/|report/ctde/|"
                "train/critic/|train/reploss/critic/|"
                "train/imag/central_critic/|train/replay/central_critic/|"
                "train/agent_jepa/|report/agent_jepa/|"
                "train/jecc/|report/jecc/|"
                "report/central_critic/|"
                "report/world_model/|report/openloop/|battle_won|win_rate|"
                "legacy_|corrected_|enemy_|ally_|timeout|action_|"
                "attack_target_|train/ppo/|train/actor_opt/|anchor/|eval/"
            ),
        ]
        if self.anchor_batch is not None:
            command.extend(["--agent.anchor_batch", str(self.anchor_batch)])
        if self.from_checkpoint is not None:
            command.extend(
                [
                    "--run.from_checkpoint",
                    str(self.from_checkpoint),
                    "--run.from_checkpoint_regex",
                    str(self.from_checkpoint_regex),
                    "--run.load_replay",
                    str(self.load_replay),
                ]
            )
        if self.replay_sampling == "recent":
            command.extend(
                [
                    "--replay.size",
                    "50000",
                    "--replay.online",
                    "False",
                ]
            )
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
            "marl_stage_status": self._marl_stage_status,
            "marl_architecture": self._marl_architecture,
            "team_contract": "explicit [B,T,A] axes with shared local modules",
            "world_model": "parallel_transformer",
            "world_model_objective": "embedding",
            "embedding_target": "ema",
            "embedding_loss": "cosine",
            "posterior_jepa": True,
            "dynamics_jepa": True,
            "spatial_jepa": not self.task.startswith("smac_"),
            "spatial_mask_ratio": 0.5,
            "spatial_mask_topology": "fixed_count",
            "spatial_fill_value": 128,
            "posterior_jepa_scale": 2.0,
            "dynamics_jepa_scale": 2.0,
            "spatial_jepa_scale": (0.0 if self.task.startswith("smac_") else 1.0),
            "sigreg": True,
            "sigreg_scale": 0.05,
            "sigreg_knots": 17,
            "sigreg_num_proj": 256,
            "sigreg_aggregation": "per_agent",
            "policy_information": "observation-local latent history",
            "action_masking": "observed online and learned for imagination",
            "replay_sampling": self.replay_sampling,
            "world_model_recency_decay": (
                0.9998 if self.replay_sampling == "recent" else None
            ),
            "behavior_replay_sampling": self.replay_sampling,
            "replay_context": 128,
            "execution": "strict decentralized parameter-shared actors",
            "imagination": (
                "training-only joint-JEPA rollouts completed through the local "
                "B0 posterior"
                if self.marl_stage == "ctde" and self.num_agents > 1
                else "synchronized independent local rollouts"
            ),
            "executable_state_supervision": "locked local JEPA/latent losses",
            "training_state": self._training_state,
            "posterior_context": "history",
            "visual_encoder": "simple",
            "visual_resolution": 64,
            "critic": (
                "shared attention critic over synchronized local posterior states"
                if self.marl_stage == "ctde" and self.num_agents > 1
                else (
                    "shared centralized local-state plus JEPA team-belief critic"
                    if self.marl_stage == "b2" and self.num_agents > 1
                    else "parameter-shared observation-local critic"
                )
            ),
            "agent_jepa_enabled": (
                self.marl_stage in {"b1", "b2"} and self.num_agents > 1
            ),
            "agent_jepa_horizon": (
                1
                if self.marl_stage in {"b1", "b2"}
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
            "jecc_enabled": self.marl_stage == "jecc" and self.num_agents > 1,
            "jecc": self._jecc_manifest if self.marl_stage == "jecc" else None,
            "ctde_enabled": self.marl_stage == "ctde" and self.num_agents > 1,
            "ctde": self._ctde_manifest if self.marl_stage == "ctde" else None,
            "algorithm_components": self._algorithm_components,
            "platform": self.platform,
            "observation_mode": (
                "local vector features"
                if self.task.startswith("smac_")
                else "local RGB vision"
            ),
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
            "smac_protocol": (
                {
                    "suite": "SMAC-v1",
                    "starcraft_version": "4.10",
                    "difficulty": "7",
                    "continuing_episode": True,
                    "training_reward": "scalar team reward broadcast to agents",
                    "training_reward_changed_by_diagnostics": False,
                    "primary_metric": "fixed deterministic battle win rate",
                    "fixed_evaluation_episodes": self.curve_eval_episodes,
                    "outcome_diagnostics": (
                        "legacy return, corrected damage return, win, timeout, "
                        "damage, shield regeneration, deaths, survivors, actions"
                    ),
                }
                if self.task.startswith("smac_")
                else None
            ),
            "accelerator_memory_preallocation": False,
            "configs": self.configs,
            "save_every_seconds": self.save_every_seconds,
            "curve_eval_interval": self.curve_eval_interval,
            "curve_eval_episodes": self.curve_eval_episodes,
            "curve_eval_seed_offset": self.curve_eval_seed_offset,
            "curve_eval_policy_mode": self.curve_eval_policy_mode,
            "imagination_starts": self.imagination_starts,
            "train_ratio": self.train_ratio,
            "training_batch_steps": 16 * 64,
            "optimizer_updates_per_environment_step": self.train_ratio / (16 * 64),
            "actor_entropy": 3e-4,
            "behavior_objective": self.behavior_objective,
            "optimizer_topology": self.behavior_optimizer,
            "ppo_epochs": self.ppo_epochs if self.behavior_objective == "ppo" else 0,
            "ppo_clip": self.ppo_clip if self.behavior_objective == "ppo" else None,
            "repval_grad": self.repval_grad,
            "report_actor_gradnorm": self.report_actor_gradnorm,
            "anchor_batch": str(self.anchor_batch) if self.anchor_batch else None,
            "continuation_checkpoint": (
                str(self.from_checkpoint) if self.from_checkpoint else None
            ),
            "continuation_checkpoint_regex": self.from_checkpoint_regex,
            "continuation_replay_loaded": self.load_replay,
            "continuation_replay_source": (
                str(self.replay_source) if self.replay_source else None
            ),
            "world_model_learning_rate": 4e-5,
            "actor_learning_rate": (
                3e-4 if self.behavior_optimizer == "grouped" else 4e-5
            ),
            "critic_learning_rate": (
                3e-4 if self.behavior_optimizer == "grouped" else 4e-5
            ),
            "jecc_learning_rate": 4e-5 if self.marl_stage == "jecc" else None,
            "ctde_joint_learning_rate": (
                4e-5 if self.marl_stage == "ctde" else None
            ),
            "ctde_rollout_steps": (
                self.ctde_rollout_steps if self.marl_stage == "ctde" else None
            ),
            "ctde_multistep_anchors": (
                self.ctde_multistep_anchors if self.marl_stage == "ctde" else None
            ),
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "command": self.command,
        }

    @property
    def _algorithm_components(self) -> list[str]:
        components = [
            (
                "DreamerV3 vector MLP encoder"
                if self.task.startswith("smac_")
                else "DreamerV3 convolutional encoder"
            ),
            "history-conditioned observation posterior",
            "causal Transformer temporal dynamics",
            (
                "decoder-free posterior and action-conditioned dynamics "
                "EMA-target cosine prediction"
                if self.task.startswith("smac_")
                else "decoder-free posterior, action-conditioned dynamics, and "
                "fixed-count masked-spatial EMA-target cosine prediction"
            ),
            "SIGReg embedding anti-collapse regularization",
            "128-step loss-excluded Transformer replay burn-in",
            (
                "single-stream exponentially recent replay for all training losses"
                if self.replay_sampling == "recent"
                else "uniform replay"
            ),
            "explicit environment, time, and agent replay axes",
            "parameter-shared local world model and actor",
            (
                "joint-JEPA synchronized imagination through the local posterior"
                if self.marl_stage == "ctde" and self.num_agents > 1
                else "synchronized independent local imagination"
            ),
        ]
        if self.num_agents > 1:
            components.append(
                "strict decentralized actor execution with no peer tensors"
            )
            if self.marl_stage in {"b1", "b2"}:
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
                if self.marl_stage == "b2":
                    components.append(
                        "JEPA-derived all-active team belief recomputed at every "
                        "replay and imagined state for centralized fast/slow critics"
                    )
            elif self.marl_stage == "jecc":
                components.append(
                    "training-only all-legal-action interventions through the "
                    "authoritative B0 local dynamics plus a multi-horizon outcome "
                    "JEPA and policy-centered counterfactual credit"
                )
            elif self.marl_stage == "ctde":
                components.extend(
                    [
                        "training-only joint-action-conditioned JEPA simulator of "
                        "each agent's next local observation embedding",
                        "joint reward, continuation, availability, and controllable "
                        "liveness "
                        "prediction for synchronized imagination",
                        "existing local temporal proposal and posterior used to "
                        "complete every imagined local actor state",
                        "central attention critic over synchronized local posterior "
                        "states with no action-conditioned central input",
                        "four disjoint local-world, joint-world, actor, and critic "
                        "optimizer groups",
                    ]
                )
            if self.marl_stage not in {"b2", "ctde"}:
                components.append("parameter-shared local critic")
        else:
            components.append("locked singleton execution")
        return components

    @property
    def _marl_architecture(self) -> str:
        if self.marl_stage == "ctde" and self.num_agents > 1:
            return (
                "unchanged local B0 execution model plus a training-only joint "
                "action-conditioned JEPA simulator whose predicted local observation "
                "embeddings are completed through the existing local posterior, and "
                "a centralized attention critic over synchronized local states"
            )
        if self.marl_stage == "jecc" and self.num_agents > 1:
            return (
                "unchanged B0 local execution model used as a stopped-gradient "
                "one-step intervention operator, followed by a training-only "
                "multi-horizon outcome JEPA and all-action policy-centered credit"
            )
        if self.marl_stage == "b2" and self.num_agents > 1:
            return (
                "B1 team EMA-slot JEPA plus a causal JEPA-derived team belief "
                "and centralized fast/slow critics with a decentralized actor"
            )
        if self.marl_stage == "b1" and self.num_agents > 1:
            if self.agent_jepa_future_scale > 0.0:
                return (
                    "shared local JEPA plus detached current and joint-action-"
                    "conditioned future team EMA-slot JEPA"
                )
            return "shared local JEPA plus detached current team EMA-slot JEPA"
        return "shared independent local JEPA"

    @property
    def _marl_stage_status(self) -> str:
        if self.marl_stage == "b0":
            return "maintained_performance_baseline"
        if self.marl_stage == "ctde":
            return "joint_jepa_ctde"
        if self.marl_stage == "jecc":
            return "rejected_acceptance_stage_retained_for_reproduction"
        return "negative_experimental_stage_retained_for_reproduction"

    @property
    def _ctde_manifest(self) -> dict[str, object]:
        return {
            "name": "joint-JEPA centralized training with decentralized execution",
            "version": "2" if self.ctde_rollout_steps == 2 else "1.1",
            "training_from_scratch": True,
            "warm_start": False,
            "execution": {
                "actor": "unchanged shared B0 actor",
                "state": "own local encoder and causal posterior history only",
                "joint_module_access": False,
                "central_critic_access": False,
            },
            "joint_simulator": {
                "inputs": (
                    "synchronized stopped local B0 posterior states, aligned joint "
                    "actions, fixed-roster and controllable-liveness masks, and "
                    "loss-free replay-prefix causal history"
                ),
                "agent_attention": {
                    "width": 256,
                    "layers": 2,
                    "heads": 4,
                    "fixed_agent_ids": False,
                },
                "temporal_transformer": {
                    "width": 256,
                    "layers": 4,
                    "heads": 4,
                    "context": 16,
                },
                "next_state_target": "stopped EMA local observation embedding",
                "posterior_interface_target": "stopped online local embedding",
                "predicted_signals": [
                    "local observation embedding",
                    "reward",
                    "continuation",
                    "action availability",
                    "controllable liveness",
                ],
                "rollout_steps": self.ctde_rollout_steps,
                "multistep_anchors": self.ctde_multistep_anchors,
                "self_fed_training": self.ctde_rollout_steps == 2,
                "self_fed_gradient": (
                    "last_step_only" if self.ctde_rollout_steps == 2 else None
                ),
                "self_fed_local_posterior": (
                    "deterministic_frozen" if self.ctde_rollout_steps == 2 else None
                ),
            },
            "imagination": (
                "local actor samples from its executable state; local dynamics makes "
                "the temporal proposal; the joint JEPA predicts each next local "
                "embedding; the existing local posterior completes the next state"
            ),
            "critic": (
                "permutation-equivariant attention over synchronized stopped local "
                "posterior states before the sampled joint action"
            ),
            "critic_action_leakage": False,
            "optimizer_groups": [
                "local_world",
                "joint_world",
                "actor",
                "critic",
            ],
            "learning_rate": 4e-5,
        }

    @property
    def _jecc_manifest(self) -> dict[str, object]:
        return {
            "name": "Joint-Embedding Counterfactual Credit",
            "version": 2,
            "acceptance_status": "failed_seed123_3s_vs_4z_gate",
            "acceptance_result": {
                "b0_wins": 13,
                "observational_critic_wins": 0,
                "jecc_wins": 0,
                "battles_per_controller": 96,
            },
            "mechanism": (
                "action intervention -> frozen B0 dynamics -> future-outcome JEPA "
                "-> policy-centered credit"
            ),
            "training_only": True,
            "execution_access": False,
            "horizons": [5, 15, 32],
            "activation": "silu",
            "normalization": "rms",
            "weight_initialization": "trunc_normal_in",
            "intervention_operator": {
                "model": "authoritative B0 local dynamics",
                "transition": (
                    "advance -> prior -> complete from the full causal dynamics carry"
                ),
                "latent_completion": (
                    "prior mode (sample=False) for factual and counterfactual paths"
                ),
                "rollout_steps": 1,
                "focal_actions": "every legal action under the exact action mask",
                "peer_actions": "factual joint-action components",
                "peer_next_states": (
                    "one factual B0 team transition computed once and held fixed "
                    "across focal alternatives"
                ),
                "trainable_by_jecc": False,
                "jecc_gradient_into_b0": False,
            },
            "outcome_encoder": {
                "width": 128,
                "embedding_dim": 128,
                "layers": 2,
                "heads": 4,
                "ff_multiplier": 4,
                "teacher_rate": 0.01,
                "teacher_every": 1,
            },
            "outcome_predictor": {
                "width": 256,
                "layers": 2,
                "heads": 4,
                "ff_multiplier": 4,
                "inputs": (
                    "grouped current B0 states, factual joint action, and grouped "
                    "counterfactual B0 next states"
                ),
                "focal_query": "content-aligned current-agent query without fixed IDs",
            },
            "utility": {
                "layers": 2,
                "units": 256,
                "output": "symexp_twohot",
                "bins": 255,
            },
            "loss_scales": {
                "outcome": 1.0,
                "utility": 1.0,
                "predicted_utility": 1.0,
            },
            "credit_baseline": "current-policy expectation over legal actions",
            "actor_advantage": "B0 and JECC return-range-normalized blend",
            "actor_gradient_boundary": "score-function signal only",
            "alpha_schedule_environment_steps": {"start": 5000, "end": 20000},
            "optimizer": {
                "topology": "disjoint Adam-style group",
                "owned_modules": [
                    "outcome_encoder",
                    "outcome_predictor",
                    "outcome_utility",
                ],
                "ema_outcome_teacher_owned": False,
                "b0_dynamics_owned": False,
            },
            "learning_rate": 4e-5,
            "acceptance_gate": {
                "frozen_replay_updates": 5000,
                "fixed_deterministic_battles": 96,
                "b0_frozen": True,
                "pass_condition": "B0 wins are nonzero and JECC wins are nonzero",
            },
        }

    @property
    def _training_state(self) -> str:
        return (
            "history-conditioned categorical posterior with a strict-causal "
            "Transformer over local latent-action histories"
        )
