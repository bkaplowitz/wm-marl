"""Gates, accounting, and pure diagnostics for the single-agent JEPA harness.

These functions carry the pass/fail semantics of ``scripts.train_dmc_jepa``:
candidate-refit gating, champion-policy baselines, online-loop bookkeeping,
real-environment step accounting, action-contrast diagnostics, and the
multi-run ``summarize`` gate. They are pure (no I/O, no argparse) so the test
suite exercises them directly.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.training import (
    ControlMode,
    _all_finite_fraction,
    masked_mean,
)

MIN_TERMINAL_FRACTION_FOR_CONTINUE_BASELINE = 0.01


def candidate_refit_gate_report(
    baseline_anchor: dict[str, Any],
    candidate_anchor: dict[str, Any],
    baseline_recent: dict[str, Any],
    candidate_recent: dict[str, Any],
    *,
    metric: str,
    min_recent_improvement: float,
    max_anchor_degradation: float,
    anchor_penalty: float = 1.0,
) -> dict[str, Any]:
    baseline_anchor_value = float(baseline_anchor[metric])
    candidate_anchor_value = float(candidate_anchor[metric])
    baseline_recent_value = float(baseline_recent[metric])
    candidate_recent_value = float(candidate_recent[metric])
    recent_improvement = baseline_recent_value - candidate_recent_value
    anchor_degradation = candidate_anchor_value - baseline_anchor_value
    gate_score = recent_improvement - anchor_penalty * max(anchor_degradation, 0.0)
    recent_improved = recent_improvement >= min_recent_improvement
    anchor_preserved = anchor_degradation <= max_anchor_degradation
    candidate_metrics_finite = metrics_finite(candidate_anchor) and metrics_finite(
        candidate_recent
    )
    return {
        "model_update_accepted": bool(
            candidate_metrics_finite and recent_improved and anchor_preserved
        ),
        "candidate_gate_metric": metric,
        "candidate_min_recent_improvement": min_recent_improvement,
        "candidate_max_anchor_degradation": max_anchor_degradation,
        "candidate_anchor_penalty": anchor_penalty,
        "candidate_gate_score": gate_score,
        "candidate_metrics_finite": candidate_metrics_finite,
        "recent_validation_baseline": baseline_recent_value,
        "recent_validation_candidate": candidate_recent_value,
        "recent_validation_improvement": recent_improvement,
        "recent_validation_improved": bool(recent_improved),
        "anchor_validation_baseline": baseline_anchor_value,
        "anchor_validation_candidate": candidate_anchor_value,
        "anchor_validation_degradation": anchor_degradation,
        "anchor_validation_preserved": bool(anchor_preserved),
    }


def best_passing_candidate_report(
    reports: list[dict[str, Any]],
) -> dict[str, Any] | None:
    passing = [report for report in reports if report.get("model_update_accepted")]
    if not passing:
        return None
    return max(
        passing,
        key=lambda report: float(
            report.get("gate", {}).get("candidate_gate_score", 0.0)
        ),
    )


def candidate_checkpoint_gate_summary(report: dict[str, Any]) -> dict[str, Any]:
    gate = report["gate"]
    return {
        "candidate_update": report.get("candidate_update"),
        "model_update_accepted": report.get("model_update_accepted"),
        "candidate_gate_score": gate.get("candidate_gate_score"),
        "recent_validation_improvement": gate.get("recent_validation_improvement"),
        "anchor_validation_degradation": gate.get("anchor_validation_degradation"),
        "recent_validation_improved": gate.get("recent_validation_improved"),
        "anchor_validation_preserved": gate.get("anchor_validation_preserved"),
        "candidate_metrics_finite": gate.get("candidate_metrics_finite"),
    }


def _concat_replay_batches(batches: list[ReplayBatch]) -> ReplayBatch:
    if len(batches) == 1:
        return batches[0]
    return ReplayBatch(
        observations=jnp.concatenate([batch.observations for batch in batches], axis=0),
        actions=jnp.concatenate([batch.actions for batch in batches], axis=0),
        rewards=jnp.concatenate([batch.rewards for batch in batches], axis=0),
        dones=jnp.concatenate([batch.dones for batch in batches], axis=0),
    )


def sample_online_candidate_batch(
    np_rng: np.random.Generator,
    *,
    replay: SequenceReplayBuffer,
    anchor_replay: SequenceReplayBuffer,
    recent_replay: SequenceReplayBuffer,
    batch_size: int,
    chunk_length: int,
    max_horizon: int,
    anchor_batch_fraction: float,
) -> ReplayBatch:
    anchor_size = int(round(batch_size * anchor_batch_fraction))
    anchor_size = max(0, min(batch_size, anchor_size))
    recent_size = batch_size - anchor_size
    batches: list[ReplayBatch] = []
    if anchor_size > 0:
        batches.append(
            anchor_replay.sample(
                np_rng,
                batch_size=anchor_size,
                chunk_length=chunk_length,
                max_horizon=max_horizon,
            )
        )
    if recent_size > 0:
        batches.append(
            recent_replay.sample(
                np_rng,
                batch_size=recent_size,
                chunk_length=chunk_length,
                max_horizon=max_horizon,
            )
        )
    if batches:
        return _concat_replay_batches(batches)
    return replay.sample(
        np_rng,
        batch_size=batch_size,
        chunk_length=chunk_length,
        max_horizon=max_horizon,
    )


def maybe_int(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def eval_env_steps(evaluation: dict[str, Any] | None) -> int:
    if evaluation is None:
        return 0
    return maybe_int(evaluation.get("env_steps"))


def eval_completed_episode_steps(evaluation: dict[str, Any] | None) -> int:
    if evaluation is None:
        return 0
    return maybe_int(evaluation.get("completed_episode_steps"))


def merge_online_policy_baseline(
    final_outcome: dict[str, Any],
    initial_outcome: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(final_outcome)
    phase_initial_mean = merged.get("policy_initial_mean")
    phase_random_mean = merged.get("policy_random_mean")
    phase_improvement = merged.get("policy_improvement")
    phase_confirmation_improvement = merged.get("policy_confirmation_improvement")
    merged["policy_online_phase_initial_mean"] = phase_initial_mean
    merged["policy_online_phase_random_mean"] = phase_random_mean
    merged["policy_online_phase_improvement"] = phase_improvement
    merged["policy_online_phase_confirmation_improvement"] = (
        phase_confirmation_improvement
    )
    pre_online_trained_mean = initial_outcome.get("policy_trained_mean")
    merged["policy_pre_online_trained_mean"] = pre_online_trained_mean
    merged["policy_online_total_improvement_vs_pre_online"] = (
        merged["policy_trained_mean"] - pre_online_trained_mean
        if pre_online_trained_mean is not None
        and merged.get("policy_trained_mean") is not None
        else None
    )
    merged["policy_initial_mean"] = initial_outcome["policy_initial_mean"]
    merged["policy_random_mean"] = initial_outcome["policy_random_mean"]
    merged["policy_improvement"] = (
        merged["policy_trained_mean"] - merged["policy_initial_mean"]
    )
    primary_improvement = (
        phase_improvement
        if phase_improvement is not None
        else merged["policy_improvement"]
    )
    merged["policy_primary_improvement"] = primary_improvement
    merged["policy_primary_improvement_key"] = "policy_online_phase_improvement"
    merged["policy_trained_minus_random"] = (
        merged["policy_trained_mean"] - merged["policy_random_mean"]
    )
    confirmation_enabled = bool(merged.get("policy_confirmation_enabled", False))
    if (
        confirmation_enabled
        and initial_outcome.get("policy_confirmation_initial_mean") is not None
    ):
        merged["policy_confirmation_initial_mean"] = initial_outcome[
            "policy_confirmation_initial_mean"
        ]
        merged["policy_confirmation_random_mean"] = initial_outcome[
            "policy_confirmation_random_mean"
        ]
        merged["policy_confirmation_improvement"] = (
            merged["policy_confirmation_trained_mean"]
            - merged["policy_confirmation_initial_mean"]
        )
        merged["policy_confirmation_trained_minus_random"] = (
            merged["policy_confirmation_trained_mean"]
            - merged["policy_confirmation_random_mean"]
        )
    primary_confirmation_improvement = (
        phase_confirmation_improvement
        if phase_confirmation_improvement is not None
        else merged.get("policy_confirmation_improvement")
    )
    merged["policy_primary_confirmation_improvement"] = primary_confirmation_improvement
    confirmation_passed = not confirmation_enabled or (
        primary_confirmation_improvement is not None
        and primary_confirmation_improvement > 0.0
        and merged.get("policy_confirmation_trained_minus_random") is not None
        and merged["policy_confirmation_trained_minus_random"] > 0.0
    )
    merged["policy_confirmation_passed"] = confirmation_passed
    policy_metrics = merged.get("policy_final_metrics", {})
    critic_metrics = merged.get("critic_final_metrics", {})
    nonregressed_from_pre_online = (
        merged["policy_online_total_improvement_vs_pre_online"] is None
        or merged["policy_online_total_improvement_vs_pre_online"] >= 0.0
    )
    merged["policy_passed"] = bool(
        metrics_finite(policy_metrics)
        and metrics_finite(critic_metrics)
        and primary_improvement > 0.0
        and nonregressed_from_pre_online
        and merged["policy_trained_mean"] > merged["policy_random_mean"]
        and confirmation_passed
        and policy_metrics.get("policy/action_saturation_fraction", 1.0) < 0.75
    )
    return merged


def online_history_metrics(
    online_history: list[dict[str, Any]],
    initial_policy_outcome: dict[str, Any],
) -> dict[str, Any]:
    returns = [
        item["actor_replay"].get("mean_return")
        for item in online_history
        if item.get("actor_replay", {}).get("mean_return") is not None
    ]
    policy_improvements = [
        item["policy"].get("policy_improvement")
        for item in online_history
        if item.get("policy", {}).get("policy_improvement") is not None
    ]
    policy_passed = [
        bool(item["policy"].get("policy_passed", False))
        for item in online_history
        if item.get("policy", {}).get("policy_training_enabled", False)
    ]
    policy_candidate_returns = [
        item["candidate_policy"].get("policy_trained_mean")
        for item in online_history
        if item.get("candidate_policy", {}).get("policy_trained_mean") is not None
    ]
    policy_champion_returns = [
        item["policy"].get("policy_champion_return")
        for item in online_history
        if item.get("policy", {}).get("policy_champion_return") is not None
    ]
    policy_update_acceptances = [
        bool(item["policy"].get("policy_update_accepted", False))
        for item in online_history
        if item.get("policy", {}).get("policy_update_accepted") is not None
    ]
    model_jepa_losses = [
        item["model_metrics"].get("model/jepa_loss")
        for item in online_history
        if item.get("model_metrics", {}).get("model/jepa_loss") is not None
    ]
    model_open_loop_losses = [
        item["model_metrics"].get("model/open_loop_loss")
        for item in online_history
        if item.get("model_metrics", {}).get("model/open_loop_loss") is not None
    ]
    candidate_refits = [
        item["candidate_refit"]
        for item in online_history
        if item.get("candidate_refit") is not None
    ]
    candidate_acceptances = [
        bool(item.get("model_update_accepted", False)) for item in candidate_refits
    ]
    candidate_recent_improvements = [
        item["gate"].get("recent_validation_improvement")
        for item in candidate_refits
        if item.get("gate", {}).get("recent_validation_improvement") is not None
    ]
    candidate_anchor_degradations = [
        item["gate"].get("anchor_validation_degradation")
        for item in candidate_refits
        if item.get("gate", {}).get("anchor_validation_degradation") is not None
    ]
    candidate_selected_updates = [
        item.get("checkpoint_selection", {}).get("candidate_selected_update")
        for item in candidate_refits
        if item.get("checkpoint_selection", {}).get("candidate_selected_update")
        is not None
    ]
    candidate_final_acceptances = [
        item.get("checkpoint_selection", {}).get("candidate_final_update_accepted")
        for item in candidate_refits
        if item.get("checkpoint_selection", {}).get("candidate_final_update_accepted")
        is not None
    ]
    baseline = initial_policy_outcome.get("policy_trained_mean")
    shared = {
        "online_policy_phase_improvements": policy_improvements,
        "online_policy_phase_final_improvement": (
            policy_improvements[-1] if policy_improvements else None
        ),
        "online_policy_phase_passes": policy_passed,
        "online_policy_phase_passed": bool(policy_passed and all(policy_passed)),
        "online_policy_candidate_returns": policy_candidate_returns,
        "online_policy_champion_returns": policy_champion_returns,
        "online_policy_update_acceptances": policy_update_acceptances,
        "online_policy_update_acceptance_rate": (
            float(np.mean(policy_update_acceptances))
            if policy_update_acceptances
            else None
        ),
        "online_policy_final_champion_return": (
            policy_champion_returns[-1] if policy_champion_returns else None
        ),
        "online_model_jepa_losses": model_jepa_losses,
        "online_model_open_loop_losses": model_open_loop_losses,
        "online_candidate_refit_iterations": len(candidate_refits),
        "online_model_update_acceptances": candidate_acceptances,
        "online_model_update_acceptance_rate": (
            float(np.mean(candidate_acceptances)) if candidate_acceptances else None
        ),
        "online_candidate_recent_validation_improvements": (
            candidate_recent_improvements
        ),
        "online_candidate_anchor_validation_degradations": (
            candidate_anchor_degradations
        ),
        "online_candidate_recent_validation_improvement_final": (
            candidate_recent_improvements[-1] if candidate_recent_improvements else None
        ),
        "online_candidate_anchor_validation_degradation_final": (
            candidate_anchor_degradations[-1] if candidate_anchor_degradations else None
        ),
        "online_candidate_selected_updates": candidate_selected_updates,
        "online_candidate_final_update_acceptances": candidate_final_acceptances,
    }
    if not returns:
        return {
            "online_actor_replay_iterations": 0,
            "online_actor_replay_returns": [],
            "online_actor_replay_first_mean": None,
            "online_actor_replay_final_mean": None,
            "online_actor_replay_delta": None,
            "online_actor_replay_vs_initial_policy": None,
            "online_actor_replay_trend_passed": False,
            **shared,
            "online_pipeline_completed": False,
        }
    delta = returns[-1] - returns[0] if len(returns) >= 2 else None
    vs_initial = returns[-1] - baseline if baseline is not None else None
    actor_replay_nonregression = vs_initial is None or vs_initial >= 0.0
    actor_replay_trend = len(returns) < 2 or returns[-1] > returns[0]
    return {
        "online_actor_replay_iterations": len(returns),
        "online_actor_replay_returns": returns,
        "online_actor_replay_first_mean": returns[0],
        "online_actor_replay_final_mean": returns[-1],
        "online_actor_replay_delta": delta,
        "online_actor_replay_vs_initial_policy": vs_initial,
        "online_actor_replay_trend_passed": (
            actor_replay_trend and actor_replay_nonregression
        ),
        **shared,
        "online_pipeline_completed": True,
    }


def real_step_accounting(
    *,
    initial_train_replay_env_steps: int,
    initial_validation_env_steps: int,
    initial_policy_outcome: dict[str, Any],
    online_history: list[dict[str, Any]],
    final_policy_eval: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Count real environment interactions used by one run.

    The main sample-efficiency number is the train replay count. Validation and
    policy-selection/evaluation interactions are kept separate because they are
    real environment steps but are not added to the training replay.
    """

    online_actor_replay_env_steps = sum(
        maybe_int(item.get("actor_replay", {}).get("env_steps"))
        for item in online_history
    )
    online_validation_env_steps = sum(
        maybe_int((item.get("recent_policy_validation") or {}).get("env_steps"))
        for item in online_history
    )
    initial_policy_eval_env_steps = maybe_int(
        initial_policy_outcome.get("policy_total_eval_env_steps")
    )
    online_policy_eval_env_steps = sum(
        maybe_int(item.get("candidate_policy", {}).get("policy_total_eval_env_steps"))
        for item in online_history
    )
    initial_policy_completed_steps = maybe_int(
        initial_policy_outcome.get("policy_total_completed_episode_steps")
    )
    online_policy_completed_steps = sum(
        maybe_int(
            item.get("candidate_policy", {}).get("policy_total_completed_episode_steps")
        )
        for item in online_history
    )
    final_policy_eval_env_steps = eval_env_steps(final_policy_eval)
    final_policy_completed_steps = eval_completed_episode_steps(final_policy_eval)

    train_replay_env_steps = (
        initial_train_replay_env_steps + online_actor_replay_env_steps
    )
    validation_replay_env_steps = (
        initial_validation_env_steps + online_validation_env_steps
    )
    policy_eval_env_steps = (
        initial_policy_eval_env_steps
        + online_policy_eval_env_steps
        + final_policy_eval_env_steps
    )
    train_plus_validation_env_steps = (
        train_replay_env_steps + validation_replay_env_steps
    )
    total_real_env_steps = train_plus_validation_env_steps + policy_eval_env_steps

    return {
        "real_initial_train_replay_env_steps": int(initial_train_replay_env_steps),
        "real_online_actor_replay_env_steps": int(online_actor_replay_env_steps),
        "real_train_replay_env_steps": int(train_replay_env_steps),
        "real_initial_validation_env_steps": int(initial_validation_env_steps),
        "real_online_validation_env_steps": int(online_validation_env_steps),
        "real_validation_replay_env_steps": int(validation_replay_env_steps),
        "real_train_plus_validation_env_steps": int(train_plus_validation_env_steps),
        "real_initial_policy_eval_env_steps": int(initial_policy_eval_env_steps),
        "real_online_policy_eval_env_steps": int(online_policy_eval_env_steps),
        "real_final_policy_eval_env_steps": int(final_policy_eval_env_steps),
        "real_policy_eval_env_steps": int(policy_eval_env_steps),
        "real_policy_eval_completed_episode_steps": int(
            initial_policy_completed_steps
            + online_policy_completed_steps
            + final_policy_completed_steps
        ),
        "real_total_env_steps": int(total_real_env_steps),
    }


@partial(jax.jit, static_argnames=("config", "chunk_length", "control"))
def action_contrast_metrics(
    state,
    key: jax.Array,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    chunk_length: int,
    control: ControlMode,
) -> dict[str, jax.Array]:
    """Compare heldout next-latent prediction under true versus wrong actions."""
    observations = batch.observations[:, : chunk_length + 1]
    actions = batch.actions[:, :chunk_length]
    validity = 1.0 - batch.dones[:, :chunk_length]

    current_obs = observations[:, :chunk_length].reshape((-1, config.observation_dim))
    next_obs = observations[:, 1 : chunk_length + 1].reshape(
        (-1, config.observation_dim)
    )
    if config.action_mode == "continuous":
        true_actions = actions.reshape((-1, config.action_dim))
    else:
        true_actions = actions.reshape((-1,)).astype(jnp.int32)
    if control == "no-action-world-model":
        wrong_actions = jnp.zeros_like(true_actions)
        true_actions = jnp.zeros_like(true_actions)
    elif config.action_mode == "continuous":
        wrong_actions = jax.random.permutation(key, true_actions, axis=0)
    else:
        # A permuted wrong action collides with the true one ~1/n of the time
        # (half the batch for CartPole's n=2), so shift by a nonzero random
        # offset to guarantee a genuinely different action.
        offsets = jax.random.randint(key, true_actions.shape, 1, config.action_dim)
        wrong_actions = (true_actions + offsets) % config.action_dim

    current_z = state.apply_fn(
        {"params": state.params},
        current_obs,
        method=JepaWorldModel.encode,
    )
    target_z = jax.lax.stop_gradient(
        state.apply_fn(
            {"params": state.params},
            next_obs,
            method=JepaWorldModel.encode,
        )
    )
    context = current_z[:, None, :]
    true_pred, _, _ = state.apply_fn(
        {"params": state.params},
        context,
        true_actions[:, None],
        method=JepaWorldModel.predict_next_from_history,
    )
    wrong_pred, _, _ = state.apply_fn(
        {"params": state.params},
        context,
        wrong_actions[:, None],
        method=JepaWorldModel.predict_next_from_history,
    )

    target_z = _normalize_latents(target_z)
    true_pred = _normalize_latents(true_pred)
    wrong_pred = _normalize_latents(wrong_pred)
    true_cosine = jnp.sum(true_pred * target_z, axis=-1).reshape(validity.shape)
    wrong_cosine = jnp.sum(wrong_pred * target_z, axis=-1).reshape(validity.shape)
    margin = true_cosine - wrong_cosine
    return {
        "model/action_contrast_true_cosine": masked_mean(true_cosine, validity),
        "model/action_contrast_wrong_cosine": masked_mean(wrong_cosine, validity),
        "model/action_contrast_margin": masked_mean(margin, validity),
        "model/action_contrast_accuracy": masked_mean(
            (margin > 0.0).astype(jnp.float32),
            validity,
        ),
        "model/action_contrast_valid_fraction": jnp.mean(validity),
        "model/action_contrast_finite_fraction": _all_finite_fraction(
            true_cosine,
            wrong_cosine,
            margin,
        ),
    }


def _normalize_latents(x: jax.Array) -> jax.Array:
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def summarize(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    main = [outcome for outcome in outcomes if outcome["control"] == "none"]
    controls = [outcome for outcome in outcomes if outcome["control"] != "none"]
    policy_enabled = any(
        outcome.get("policy_training_enabled", False) for outcome in outcomes
    )
    confirmation_enabled = any(
        outcome.get("policy_confirmation_enabled", False) for outcome in outcomes
    )
    online_enabled = any(
        outcome.get("online_actor_replay_iterations", 0) > 0 for outcome in outcomes
    )
    main_passed = all(
        outcome.get("world_model_passed", outcome["passed"]) for outcome in main
    )
    controls_finite = all(
        metrics_finite(outcome["final_model_metrics"]) for outcome in controls
    )
    main_open_loop = _mean(main, "final_open_loop_loss")
    main_jepa = _mean(main, "final_jepa_loss")
    control_open_loop = _mean(controls, "final_open_loop_loss")
    control_jepa = _mean(controls, "final_jepa_loss")
    policy_objectives = sorted(
        {
            outcome["policy_objective"]
            for outcome in outcomes
            if outcome.get("policy_training_enabled", False)
            and outcome.get("policy_objective") is not None
        }
    )
    primary_policy_objective = (
        policy_objectives[0] if len(policy_objectives) == 1 else None
    )
    direct_policy_mainline = not policy_enabled or primary_policy_objective == "direct"
    policy_comparison_key = _policy_comparison_key(outcomes)
    paired = _paired_control_differences(
        outcomes,
        policy_key=policy_comparison_key,
    )
    main_beats_controls_open_loop = not controls or (
        main_open_loop is not None
        and control_open_loop is not None
        and main_open_loop < control_open_loop
    )
    main_beats_controls_jepa = not controls or (
        main_jepa is not None and control_jepa is not None and main_jepa < control_jepa
    )
    paired_open_loop_ok = not paired or all(
        item["mean_open_loop_advantage"] > 0.0
        and item["runs_main_better_open_loop"] >= item["required_majority_pairs"]
        for item in paired.values()
    )
    paired_jepa_ok = not paired or all(
        item["mean_jepa_advantage"] > 0.0
        and item["runs_main_better_jepa"] >= item["required_majority_pairs"]
        for item in paired.values()
    )
    policy_main_passed = True
    policy_main_successes = 0
    policy_required_successes = 0
    policy_aggregate_improved = True
    policy_aggregate_beats_random = True
    policy_main_beats_controls = True
    paired_policy_ok = True
    policy_confirmation_successes = 0
    online_trend_successes = 0
    online_trend_passed = True
    if policy_enabled:
        policy_main_successes = int(
            sum(outcome.get("policy_passed", False) for outcome in main)
        )
        policy_required_successes = max(1, math.ceil((2 * len(main)) / 3))
        policy_confirmation_successes = int(
            sum(outcome.get("policy_confirmation_passed", False) for outcome in main)
        )
        online_trend_successes = int(
            sum(
                outcome.get("online_actor_replay_trend_passed", False)
                for outcome in main
                if outcome.get("online_actor_replay_iterations", 0) > 0
            )
        )
        online_trend_passed = bool(
            not online_enabled or online_trend_successes >= policy_required_successes
        )
        main_policy_improvement = _mean(main, policy_comparison_key)
        main_policy_minus_random = _mean(main, "policy_trained_minus_random")
        policy_aggregate_improved = bool(
            main_policy_improvement is not None and main_policy_improvement > 0.0
        )
        policy_aggregate_beats_random = bool(
            main_policy_minus_random is not None and main_policy_minus_random > 0.0
        )
        policy_main_passed = bool(
            main
            and policy_main_successes >= policy_required_successes
            and policy_aggregate_improved
            and policy_aggregate_beats_random
            and online_trend_passed
        )
        control_policy_improvement = _mean(controls, policy_comparison_key)
        policy_main_beats_controls = not controls or (
            main_policy_improvement is not None
            and control_policy_improvement is not None
            and main_policy_improvement > control_policy_improvement
        )
        paired_policy_ok = not paired or all(
            item.get("mean_policy_primary_improvement_advantage") is not None
            and item["mean_policy_primary_improvement_advantage"] > 0.0
            and item["runs_main_better_policy_primary"]
            >= item["required_majority_pairs"]
            for item in paired.values()
        )
    return {
        "world_model_passed": bool(
            main
            and main_passed
            and controls_finite
            and main_beats_controls_open_loop
            and main_beats_controls_jepa
            and paired_open_loop_ok
            and paired_jepa_ok
        ),
        "passed": bool(
            main
            and main_passed
            and controls_finite
            and main_beats_controls_open_loop
            and main_beats_controls_jepa
            and paired_open_loop_ok
            and paired_jepa_ok
            and policy_main_passed
            and policy_main_beats_controls
            and paired_policy_ok
            and online_trend_passed
            and direct_policy_mainline
        ),
        "main_runs_passed": int(
            sum(
                outcome.get("world_model_passed", outcome["passed"]) for outcome in main
            )
        ),
        "main_runs": len(main),
        "controls_finite": controls_finite,
        "main_beats_controls_open_loop": main_beats_controls_open_loop,
        "main_beats_controls_jepa": main_beats_controls_jepa,
        "paired_open_loop_ok": paired_open_loop_ok,
        "paired_jepa_ok": paired_jepa_ok,
        "policy_training_enabled": policy_enabled,
        "milestone": (
            "single_agent_direct_latent_imagination_rl"
            if policy_enabled
            else "single_agent_jepa_world_model_validation"
        ),
        "policy_objectives": policy_objectives,
        "primary_policy_objective": primary_policy_objective,
        "direct_policy_mainline": direct_policy_mainline,
        "policy_main_passed": policy_main_passed,
        "policy_main_successes": policy_main_successes,
        "policy_required_successes": policy_required_successes,
        "policy_confirmation_enabled": confirmation_enabled,
        "policy_confirmation_successes": policy_confirmation_successes,
        "policy_aggregate_improved": policy_aggregate_improved,
        "policy_aggregate_beats_random": policy_aggregate_beats_random,
        "policy_main_beats_controls": policy_main_beats_controls,
        "paired_policy_ok": paired_policy_ok,
        "online_training_enabled": online_enabled,
        "online_trend_successes": online_trend_successes,
        "online_trend_passed": online_trend_passed,
        "policy_comparison_key": policy_comparison_key,
        "paired_control_differences": paired,
        "aggregate_initial_jepa_loss": _mean(main, "initial_jepa_loss"),
        "aggregate_final_jepa_loss": main_jepa,
        "aggregate_control_final_jepa_loss": control_jepa,
        "aggregate_initial_open_loop_loss": _mean(main, "initial_open_loop_loss"),
        "aggregate_final_open_loop_loss": main_open_loop,
        "aggregate_control_final_open_loop_loss": control_open_loop,
        "aggregate_policy_random_mean": _mean(main, "policy_random_mean"),
        "aggregate_policy_initial_mean": _mean(main, "policy_initial_mean"),
        "aggregate_policy_trained_mean": _mean(main, "policy_trained_mean"),
        "aggregate_policy_improvement": _mean(main, "policy_improvement"),
        "aggregate_policy_online_phase_improvement": _mean(
            main,
            "policy_online_phase_improvement",
        ),
        "aggregate_policy_online_actor_replay_delta": _mean(
            main,
            "online_actor_replay_delta",
        ),
        "aggregate_policy_online_actor_replay_vs_initial": _mean(
            main,
            "online_actor_replay_vs_initial_policy",
        ),
        "aggregate_model_update_acceptance_rate": _mean(
            main,
            "online_model_update_acceptance_rate",
        ),
        "aggregate_candidate_recent_validation_improvement": _mean(
            main,
            "online_candidate_recent_validation_improvement_final",
        ),
        "aggregate_candidate_anchor_validation_degradation": _mean(
            main,
            "online_candidate_anchor_validation_degradation_final",
        ),
        "aggregate_candidate_selected_update": _flat_mean(
            main,
            "online_candidate_selected_updates",
        ),
        "aggregate_candidate_final_update_acceptance_rate": _flat_mean(
            main,
            "online_candidate_final_update_acceptances",
        ),
        "aggregate_policy_update_acceptance_rate": _flat_mean(
            main,
            "online_policy_update_acceptances",
        ),
        "aggregate_policy_final_champion_return": _mean(
            main,
            "online_policy_final_champion_return",
        ),
        "aggregate_final_policy_eval_mean": _mean(
            main,
            "final_policy_eval_mean",
        ),
        "aggregate_final_policy_eval_std": _mean(
            main,
            "final_policy_eval_std",
        ),
        "aggregate_final_policy_eval_episodes": _mean(
            main,
            "final_policy_eval_episodes",
        ),
        "aggregate_final_policy_eval_env_steps": _mean(
            main,
            "final_policy_eval_env_steps",
        ),
        "aggregate_real_train_replay_env_steps": _mean(
            main,
            "real_train_replay_env_steps",
        ),
        "aggregate_real_validation_replay_env_steps": _mean(
            main,
            "real_validation_replay_env_steps",
        ),
        "aggregate_real_train_plus_validation_env_steps": _mean(
            main,
            "real_train_plus_validation_env_steps",
        ),
        "aggregate_real_policy_eval_env_steps": _mean(
            main,
            "real_policy_eval_env_steps",
        ),
        "aggregate_real_total_env_steps": _mean(
            main,
            "real_total_env_steps",
        ),
        "aggregate_real_policy_eval_completed_episode_steps": _mean(
            main,
            "real_policy_eval_completed_episode_steps",
        ),
        "aggregate_policy_primary_improvement": _mean(
            main,
            policy_comparison_key,
        ),
        "aggregate_policy_primary_confirmation_improvement": _mean(
            main,
            "policy_primary_confirmation_improvement",
        ),
        "aggregate_policy_trained_minus_random": _mean(
            main,
            "policy_trained_minus_random",
        ),
        "aggregate_control_policy_improvement": _mean(
            controls,
            "policy_improvement",
        ),
        "aggregate_control_policy_online_phase_improvement": _mean(
            controls,
            "policy_online_phase_improvement",
        ),
        "aggregate_control_policy_primary_improvement": _mean(
            controls,
            policy_comparison_key,
        ),
        "runs": outcomes,
    }


def run_passed(
    initial_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
    reload_diff: float,
) -> bool:
    return bool(
        metrics_finite(final_metrics)
        and reload_diff <= 1e-6
        and final_metrics["model/open_loop_finite_fraction"] >= 1.0
        and final_metrics["model/jepa_loss"] <= initial_metrics["model/jepa_loss"]
        and final_metrics["model/open_loop_loss"]
        <= initial_metrics["model/open_loop_loss"]
        and final_metrics["model/reward_loss"]
        < final_metrics.get(
            "model/reward_constant_loss",
            final_metrics["model/reward_constant_mse"],
        )
        and _continue_criterion_passed(final_metrics)
    )


def _continue_criterion_passed(final_metrics: dict[str, Any]) -> bool:
    terminal_fraction = final_metrics.get("model/terminal_positive_fraction", 0.0)
    if terminal_fraction >= MIN_TERMINAL_FRACTION_FOR_CONTINUE_BASELINE:
        return (
            final_metrics["model/continue_loss"]
            < final_metrics["model/continue_constant_bce"]
        )
    return (
        math.isfinite(final_metrics["model/continue_loss"])
        and final_metrics.get("model/nonterminal_recall", 0.0) >= 0.95
    )


def metrics_finite(metrics: dict[str, Any]) -> bool:
    for value in metrics.values():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            return False
    return True


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := _metric_value(row, key)) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _flat_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = _metric_value(row, key)
        if value is None:
            continue
        if isinstance(value, list):
            values.extend(item for item in value if item is not None)
        else:
            values.append(value)
    if not values:
        return None
    return float(np.mean(values))


def _metric_value(row: dict[str, Any], key: str) -> Any | None:
    if key == "policy_primary_improvement":
        return row.get(
            "policy_primary_improvement",
            row.get(
                "policy_online_phase_improvement",
                row.get("policy_improvement"),
            ),
        )
    return row.get(key)


def _policy_comparison_key(outcomes: list[dict[str, Any]]) -> str:
    if any("policy_primary_improvement" in outcome for outcome in outcomes):
        return "policy_primary_improvement"
    if any("policy_online_phase_improvement" in outcome for outcome in outcomes):
        return "policy_online_phase_improvement"
    return "policy_improvement"


def _paired_control_differences(
    outcomes: list[dict[str, Any]],
    *,
    policy_key: str,
) -> dict[str, dict[str, Any]]:
    main_by_run = {
        outcome["run_index"]: outcome
        for outcome in outcomes
        if outcome["control"] == "none"
    }
    result: dict[str, dict[str, Any]] = {}
    for control in sorted({outcome["control"] for outcome in outcomes} - {"none"}):
        jepa_advantages = []
        open_loop_advantages = []
        policy_improvement_advantages = []
        policy_online_phase_advantages = []
        policy_primary_advantages = []
        for outcome in outcomes:
            if outcome["control"] != control:
                continue
            main = main_by_run.get(outcome["run_index"])
            if main is None:
                continue
            jepa_advantages.append(outcome["final_jepa_loss"] - main["final_jepa_loss"])
            open_loop_advantages.append(
                outcome["final_open_loop_loss"] - main["final_open_loop_loss"]
            )
            if "policy_improvement" in outcome and "policy_improvement" in main:
                policy_improvement_advantages.append(
                    main["policy_improvement"] - outcome["policy_improvement"]
                )
            main_online = _metric_value(main, "policy_online_phase_improvement")
            control_online = _metric_value(outcome, "policy_online_phase_improvement")
            if main_online is not None and control_online is not None:
                policy_online_phase_advantages.append(main_online - control_online)
            main_primary = _metric_value(main, policy_key)
            control_primary = _metric_value(outcome, policy_key)
            if main_primary is not None and control_primary is not None:
                policy_primary_advantages.append(main_primary - control_primary)
        result[control] = {
            "pairs": len(jepa_advantages),
            "required_majority_pairs": _required_majority(len(jepa_advantages)),
            "mean_jepa_advantage": (
                float(np.mean(jepa_advantages)) if jepa_advantages else None
            ),
            "mean_open_loop_advantage": (
                float(np.mean(open_loop_advantages)) if open_loop_advantages else None
            ),
            "runs_main_better_jepa": int(np.sum(np.asarray(jepa_advantages) > 0.0)),
            "runs_main_better_open_loop": int(
                np.sum(np.asarray(open_loop_advantages) > 0.0)
            ),
            "mean_policy_improvement_advantage": (
                float(np.mean(policy_improvement_advantages))
                if policy_improvement_advantages
                else None
            ),
            "mean_policy_online_phase_improvement_advantage": (
                float(np.mean(policy_online_phase_advantages))
                if policy_online_phase_advantages
                else None
            ),
            "mean_policy_primary_improvement_advantage": (
                float(np.mean(policy_primary_advantages))
                if policy_primary_advantages
                else None
            ),
            "runs_main_better_policy": int(
                np.sum(np.asarray(policy_improvement_advantages) > 0.0)
            ),
            "runs_main_better_policy_online_phase": int(
                np.sum(np.asarray(policy_online_phase_advantages) > 0.0)
            ),
            "runs_main_better_policy_primary": int(
                np.sum(np.asarray(policy_primary_advantages) > 0.0)
            ),
        }
    return result


def _required_majority(count: int) -> int:
    return max(1, math.ceil((2 * count) / 3)) if count > 0 else 0
