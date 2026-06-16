"""Validate a simple world model on Melting Pot state representations."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import numpy as np
from tqdm import tqdm

from world_marl.checkpointing import load_params, save_checkpoint
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.logging import RunLogger, dependency_versions, timestamp, to_jsonable
from world_marl.state_model import (
  StateRepresentationConfig,
  WorldModelConfig,
  collect_transition_dataset,
  create_world_model_train_state,
  evaluate_state_fit,
  fit_feature_normalizer,
  plot_prediction_dashboard,
  plot_state_recoveries,
  predict_world_model,
  prepare_transition_data,
  sigmoid_np,
  split_prepared_data,
  train_world_model,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--substrate", default="coins")
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--max-cycles", type=int, default=500)
  parser.add_argument("--observation-size", type=int, default=22)
  parser.add_argument("--include-observation-scalars", action="store_true")
  parser.add_argument("--append-agent-id", action="store_true")
  parser.add_argument(
    "--target-source",
    choices=("random", "checkpoint"),
    default="random",
    help="Behavior policy used to collect transitions.",
  )
  parser.add_argument("--policy-checkpoint", default=None)
  parser.add_argument("--source-stochastic", action="store_true")
  parser.add_argument("--collect-steps", type=int, default=512)
  parser.add_argument("--validation-fraction", type=float, default=0.25)
  parser.add_argument("--pool-size", type=int, default=4)
  parser.add_argument("--no-channel-stats", action="store_true")
  parser.add_argument("--train-steps", type=int, default=1000)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--hidden-dims", default="256,256")
  parser.add_argument("--next-loss-weight", type=float, default=1.0)
  parser.add_argument("--delta-loss-weight", type=float, default=0.5)
  parser.add_argument("--changed-loss-weight", type=float, default=1.0)
  parser.add_argument("--reward-loss-weight", type=float, default=1.0)
  parser.add_argument("--reward-event-loss-weight", type=float, default=0.25)
  parser.add_argument("--done-loss-weight", type=float, default=0.1)
  parser.add_argument("--policy-loss-weight", type=float, default=0.1)
  parser.add_argument("--max-grad-norm", type=float, default=1.0)
  parser.add_argument("--reward-oversample-factor", type=float, default=8.0)
  parser.add_argument("--delta-oversample-factor", type=float, default=2.0)
  parser.add_argument("--changed-feature-fraction", type=float, default=0.25)
  parser.add_argument("--reward-event-epsilon", type=float, default=1e-6)
  parser.add_argument("--recovery-examples", type=int, default=6)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out-dir", default="runs")
  parser.add_argument("--quiet", action="store_true")
  return parser.parse_args()


def parse_hidden_dims(value: str) -> tuple[int, ...]:
  dims = tuple(int(item.strip()) for item in value.split(",") if item.strip())
  if not dims:
    raise ValueError("--hidden-dims must contain at least one integer")
  if any(dim < 1 for dim in dims):
    raise ValueError("--hidden-dims must be positive")
  return dims


def make_adapter(args: argparse.Namespace) -> MeltingPotVectorAdapter:
  return MeltingPotVectorAdapter(
    substrate=args.substrate,
    num_envs=args.num_envs,
    max_cycles=args.max_cycles,
    observation_size=args.observation_size,
    include_observation_scalars=args.include_observation_scalars,
    append_agent_id=args.append_agent_id,
  )


def log_stage(args: argparse.Namespace, message: str) -> None:
  if not args.quiet:
    print(f"[state-model] {message}", flush=True)


def main() -> None:
  args = parse_args()
  if args.target_source == "checkpoint" and args.policy_checkpoint is None:
    raise SystemExit("--policy-checkpoint is required with --target-source checkpoint")

  hidden_dims = parse_hidden_dims(args.hidden_dims)
  representation_config = StateRepresentationConfig(
    pool_size=args.pool_size,
    include_channel_stats=not args.no_channel_stats,
  )
  model_config = WorldModelConfig(
    hidden_dims=hidden_dims,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    train_steps=args.train_steps,
    next_loss_weight=args.next_loss_weight,
    delta_loss_weight=args.delta_loss_weight,
    changed_loss_weight=args.changed_loss_weight,
    reward_loss_weight=args.reward_loss_weight,
    reward_event_loss_weight=args.reward_event_loss_weight,
    done_loss_weight=args.done_loss_weight,
    policy_loss_weight=args.policy_loss_weight,
    max_grad_norm=args.max_grad_norm,
    reward_oversample_factor=args.reward_oversample_factor,
    delta_oversample_factor=args.delta_oversample_factor,
    changed_feature_fraction=args.changed_feature_fraction,
    reward_event_epsilon=args.reward_event_epsilon,
  )

  run_dir = Path(args.out_dir) / f"state_model_{timestamp()}"
  log_stage(args, f"writing artifacts to {run_dir}")
  logger = RunLogger(run_dir)
  logger.write_json(
    "config.json",
    {
      "args": vars(args),
      "hidden_dims": hidden_dims,
      "representation_config": representation_config,
      "model_config": model_config,
      "purpose": (
        "Validate whether a simple supervised world model fits deterministic "
        "state representations from Melting Pot rollouts. This is state, "
        "reward, done, and behavior-policy recovery validation, not imagined "
        "policy improvement."
      ),
    },
  )
  logger.write_json("versions.json", dependency_versions())

  np_rng = np.random.default_rng(args.seed)
  source_metadata: dict[str, Any] | None = None
  source_policy = None
  adapter = make_adapter(args)
  try:
    env_metadata = {
      "substrate": adapter.substrate,
      "num_agents": adapter.num_agents,
      "action_dim": adapter.action_dim,
      "observation_shape": adapter.observation_shape,
      "raw_observation_shape": adapter.raw_observation_shape,
      "scalar_observation_keys": adapter.scalar_observation_keys,
    }
    if args.target_source == "checkpoint":
      from world_marl.scripts.train_coin_flow import load_checkpoint_policy

      log_stage(args, f"loading source checkpoint from {args.policy_checkpoint}")
      source_policy, source_metadata = load_checkpoint_policy(
        args.policy_checkpoint,
        adapter,
        deterministic=not args.source_stochastic,
        seed=args.seed + 10,
      )

    log_stage(
      args,
      (
        f"collecting {args.collect_steps} transition steps "
        f"({args.collect_steps * args.num_envs} joint transitions) "
        f"from {args.target_source}"
      ),
    )
    with tqdm(
      total=args.collect_steps,
      desc="collect transitions",
      unit="step",
      disable=args.quiet,
    ) as progress:
      dataset = collect_transition_dataset(
        adapter,
        np_rng,
        rollout_steps=args.collect_steps,
        policy_fn=source_policy,
        progress_callback=lambda _step: progress.update(1),
      )
  finally:
    adapter.close()

  logger.write_json(
    "transition_dataset.json",
    {
      **dataset.to_metadata(),
      "env": env_metadata,
      "target_source": args.target_source,
      "source_checkpoint_metadata": source_metadata,
      "validation_fraction": args.validation_fraction,
    },
  )
  log_stage(
    args,
    (
      f"collected {dataset.num_transitions} transitions; "
      f"mean reward per agent={dataset.rewards.mean(axis=0).round(4).tolist()}"
    ),
  )

  log_stage(args, "embedding observations into deterministic state features")
  prepared = prepare_transition_data(dataset, representation_config)
  train_data, validation_data = split_prepared_data(
    prepared,
    validation_fraction=args.validation_fraction,
    seed=args.seed,
  )
  normalizer = fit_feature_normalizer(
    train_data.state_features,
    train_data.next_state_features,
  )
  train_data = replace(train_data, normalizer=normalizer)
  validation_data = replace(validation_data, normalizer=normalizer)
  logger.write_json(
    "representation.json",
    {
      "feature_dim": prepared.feature_dim,
      "num_agents": prepared.num_agents,
      "pool_size": representation_config.pool_size,
      "include_channel_stats": representation_config.include_channel_stats,
      "normalizer": normalizer.to_metadata(),
      "train_transitions": train_data.num_transitions,
      "heldout_transitions": validation_data.num_transitions,
      "train_reward_event_fraction": float(
        (np.abs(train_data.rewards) > args.reward_event_epsilon).mean()
      ),
      "heldout_reward_event_fraction": float(
        (np.abs(validation_data.rewards) > args.reward_event_epsilon).mean()
      ),
      "changed_feature_fraction": args.changed_feature_fraction,
    },
  )
  log_stage(
    args,
    (
      f"feature_dim={prepared.feature_dim}; "
      f"train={train_data.num_transitions}, heldout={validation_data.num_transitions}"
    ),
  )

  rng = jax.random.PRNGKey(args.seed)
  log_stage(args, f"training state-fit world model for {args.train_steps} steps")
  with tqdm(
    total=args.train_steps,
    desc="train state model",
    unit="step",
    disable=args.quiet,
  ) as progress:
    def update_progress(step: int, losses: dict[str, float]) -> None:
      progress.update(1)
      progress.set_postfix(loss=f"{losses['loss']:.4g}", next=f"{losses['next_mse']:.4g}")
      logger.append_metrics({"step": step, **losses})

    train_state, loss_rows = train_world_model(
      rng,
      train_data,
      config=model_config,
      progress_callback=update_progress,
    )
  finite_losses = bool(np.isfinite([row["loss"] for row in loss_rows]).all())
  logger.write_json(
    "training_summary.json",
    {
      "initial_loss": loss_rows[0]["loss"],
      "final_loss": loss_rows[-1]["loss"],
      "min_loss": min(row["loss"] for row in loss_rows),
      "finite_losses": finite_losses,
      "train_steps": args.train_steps,
    },
  )

  log_stage(args, "evaluating heldout state, reward, done, and policy recovery")
  predictions = predict_world_model(train_state, validation_data)
  prediction_metrics = evaluate_state_fit(
    train_data,
    validation_data,
    predictions,
    seed=args.seed,
    changed_feature_fraction=args.changed_feature_fraction,
    reward_event_epsilon=args.reward_event_epsilon,
  )
  logger.write_json("prediction_metrics.json", prediction_metrics)

  log_stage(args, "saving checkpoint and validating reload equality")
  save_checkpoint(
    run_dir / "checkpoint",
    train_state,
    metadata=to_jsonable({
      "kind": "state_representation_fit_world_model",
      "substrate": args.substrate,
      "target_source": args.target_source,
      "source_checkpoint": args.policy_checkpoint,
      "env": env_metadata,
      "feature_dim": prepared.feature_dim,
      "num_agents": prepared.num_agents,
      "action_dim": prepared.action_dim,
      "representation_config": representation_config,
      "model_config": model_config,
      "config": vars(args),
    }),
  )
  reload_state = create_world_model_train_state(
    jax.random.PRNGKey(args.seed + 1000),
    feature_dim=validation_data.feature_dim,
    num_agents=validation_data.num_agents,
    action_dim=validation_data.action_dim,
    config=model_config,
  )
  reload_params = load_params(
    run_dir / "checkpoint" / "checkpoint.msgpack",
    reload_state.params,
  )
  reload_state = reload_state.replace(params=reload_params)
  reload_predictions = predict_world_model(reload_state, validation_data)
  reload_max_abs_diff = max(
    float(np.max(np.abs(reload_predictions.next_state_features - predictions.next_state_features))),
    float(np.max(np.abs(reload_predictions.rewards - predictions.rewards))),
    float(np.max(np.abs(reload_predictions.reward_event_logits - predictions.reward_event_logits))),
    float(np.max(np.abs(reload_predictions.done_logits - predictions.done_logits))),
    float(np.max(np.abs(reload_predictions.policy_logits - predictions.policy_logits))),
  )
  reload_passed = reload_max_abs_diff <= 1e-6
  logger.write_json(
    "reload_evaluation.json",
    {
      "reload_passed": reload_passed,
      "reload_max_abs_prediction_diff": reload_max_abs_diff,
    },
  )

  log_stage(args, "writing state recovery visual artifacts")
  plot_prediction_dashboard(
    run_dir / "prediction_dashboard.png",
    train_data,
    validation_data,
    predictions,
    prediction_metrics,
    seed=args.seed,
  )
  recovery_examples = plot_state_recoveries(
    run_dir / "state_recoveries.png",
    validation_data,
    predictions,
    num_examples=args.recovery_examples,
    seed=args.seed,
  )
  logger.write_json(
    "sample_predictions.json",
    {
      "recovery_examples": recovery_examples,
      "reward_predictions_preview": predictions.rewards[: min(10, len(predictions.rewards))].tolist(),
      "reward_event_probabilities_preview": sigmoid_np(
        predictions.reward_event_logits[: min(10, len(predictions.reward_event_logits))]
      ).tolist(),
      "reward_targets_preview": validation_data.rewards[: min(10, len(validation_data.rewards))].tolist(),
      "actions_preview": validation_data.actions[: min(10, len(validation_data.actions))].astype(int).tolist(),
    },
  )

  reward_model_has_signal = bool(
    prediction_metrics["reward"]["model_beats_mean"]
    or prediction_metrics["reward"]["model_beats_zero"]
    or prediction_metrics["reward_event"]["model_beats_prior_bce"]
  )
  transition_model_has_signal = bool(
    prediction_metrics["next_state"]["model_beats_persistence"]
    or prediction_metrics["delta_state"]["model_beats_zero_delta"]
    or prediction_metrics["changed_features"]["delta_model_beats_zero"]
  )
  passed = bool(
    finite_losses
    and reload_passed
    and transition_model_has_signal
    and prediction_metrics["policy"]["model_beats_marginal_ce"]
  )
  outcome = {
    "passed": passed,
    "criteria": {
      "finite_losses": finite_losses,
      "reload_passed": reload_passed,
      "transition_model_has_signal": transition_model_has_signal,
      "next_state_model_beats_mean": prediction_metrics["next_state"]["model_beats_mean"],
      "next_state_model_beats_persistence": prediction_metrics["next_state"][
        "model_beats_persistence"
      ],
      "delta_model_beats_zero": prediction_metrics["delta_state"][
        "model_beats_zero_delta"
      ],
      "delta_model_beats_mean_delta": prediction_metrics["delta_state"][
        "model_beats_mean_delta"
      ],
      "changed_feature_delta_model_beats_zero": prediction_metrics["changed_features"][
        "delta_model_beats_zero"
      ],
      "reward_model_beats_mean": prediction_metrics["reward"]["model_beats_mean"],
      "reward_model_beats_zero": prediction_metrics["reward"]["model_beats_zero"],
      "reward_event_model_beats_prior_bce": prediction_metrics["reward_event"][
        "model_beats_prior_bce"
      ],
      "reward_model_has_signal": reward_model_has_signal,
      "policy_model_beats_marginal_ce": prediction_metrics["policy"][
        "model_beats_marginal_ce"
      ],
    },
    "next_state": prediction_metrics["next_state"],
    "delta_state": prediction_metrics["delta_state"],
    "changed_features": prediction_metrics["changed_features"],
    "reward": prediction_metrics["reward"],
    "reward_event": prediction_metrics["reward_event"],
    "done": prediction_metrics["done"],
    "policy": prediction_metrics["policy"],
    "state_distribution": prediction_metrics["state_distribution"],
    "delta_distribution": prediction_metrics["delta_distribution"],
    "reload_max_abs_prediction_diff": reload_max_abs_diff,
    "artifacts": {
      "prediction_dashboard": str(run_dir / "prediction_dashboard.png"),
      "state_recoveries": str(run_dir / "state_recoveries.png"),
      "checkpoint": str(run_dir / "checkpoint"),
    },
  }
  logger.write_json("evaluation.json", outcome)
  log_stage(
    args,
    (
      "done; "
      f"passed={passed}, "
      f"next_mse={prediction_metrics['next_state']['model_mse']:.6g}, "
      f"persist_baseline={prediction_metrics['next_state']['persistence_baseline_mse']:.6g}, "
      f"delta_mse={prediction_metrics['delta_state']['model_mse']:.6g}, "
      f"reward_event_bce={prediction_metrics['reward_event']['model_bce']:.6g}, "
      f"reload_diff={reload_max_abs_diff:.3g}"
    ),
  )
  print(logger.write_json("outcome.json", outcome).read_text(encoding="utf-8"))


if __name__ == "__main__":
  main()
