"""Train a flow-matching joint-action sampler on JaxMARL CoinGame rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import numpy as np
from tqdm import tqdm

from flow_matching.distributions import sample_gmm
from flow_matching.models import MLPVectorField
from flow_matching.train import (
  create_conditioned_train_state,
  create_train_state as create_flow_train_state,
)
from world_marl.algs.ippo import IPPOConfig
from world_marl.algs.mappo import MAPPOConfig
from world_marl.checkpointing import load_metadata, load_params, save_checkpoint
from world_marl.coin_flow import (
  action_prediction_metrics,
  classifier_joint_action_policy,
  compare_joint_action_distributions,
  collect_policy_joint_actions,
  collect_policy_state_actions,
  collect_random_joint_actions,
  collect_random_state_actions,
  conditional_flow_joint_action_policy,
  decode_joint_actions,
  fit_feature_normalizer,
  fit_joint_action_gmm,
  flow_joint_action_policy,
  predict_action_logits,
  sample_conditional_action_flow_points,
  sample_flow_points,
  sampled_action_prediction_metrics,
  split_joint_actions,
  split_state_action_dataset,
  summarize_joint_action_distribution,
  train_action_classifier,
  train_conditional_action_flow,
  train_flow_for_gmm,
  uniform_joint_actions,
  create_action_classifier_train_state,
)
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.evaluation import evaluate_policy, random_policy
from world_marl.logging import RunLogger, dependency_versions, timestamp
from world_marl.scripts.train_e2e import (
  create_algorithm_train_state,
  policy_from_train_state,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--substrate", default="coins")
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--max-cycles", type=int, default=500)
  parser.add_argument("--observation-size", type=int, default=44)
  parser.add_argument("--include-observation-scalars", action="store_true")
  parser.add_argument("--append-agent-id", action="store_true")
  parser.add_argument(
    "--target-source",
    choices=("random", "checkpoint"),
    default="random",
    help="Source policy used to collect joint actions for the GMM target.",
  )
  parser.add_argument(
    "--policy-checkpoint",
    default=None,
    help="IPPO/MAPPO checkpoint directory used when --target-source checkpoint.",
  )
  parser.add_argument(
    "--source-stochastic",
    action="store_true",
    help="Sample checkpoint policy actions while collecting/evaluating source actions.",
  )
  parser.add_argument("--collect-steps", type=int, default=256)
  parser.add_argument("--validation-fraction", type=float, default=0.25)
  parser.add_argument("--gmm-std", type=float, default=0.10)
  parser.add_argument("--max-components", type=int, default=None)
  parser.add_argument("--train-steps", type=int, default=1000)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--hidden-dims", default="64,64,64,64")
  parser.add_argument(
    "--conditional-action",
    action="store_true",
    help=(
      "Run Milestone 1: learn p(joint_action | state) from rollout "
      "state-action pairs instead of the older unconditional p(joint_action) demo."
    ),
  )
  parser.add_argument(
    "--classifier-hidden-dims",
    default="128,128",
    help="Hidden dimensions for the categorical p(action | state) sanity baseline.",
  )
  parser.add_argument(
    "--classifier-learning-rate",
    type=float,
    default=None,
    help="Optional learning rate for the categorical baseline; defaults to --learning-rate.",
  )
  parser.add_argument("--flow-integration-steps", type=int, default=64)
  parser.add_argument("--generated-samples", type=int, default=256)
  parser.add_argument("--distribution-top-k", type=int, default=5)
  parser.add_argument("--eval-episodes", type=int, default=10)
  parser.add_argument("--eval-max-steps", type=int, default=None)
  parser.add_argument(
    "--skip-policy-eval",
    action="store_true",
    help="Only run distribution validation; skip environment return evaluation.",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out-dir", default="runs")
  parser.add_argument(
    "--quiet",
    action="store_true",
    help="Disable terminal progress output.",
  )
  return parser.parse_args()


def parse_hidden_dims(value: str) -> tuple[int, ...]:
  dims = tuple(int(item.strip()) for item in value.split(",") if item.strip())
  if not dims:
    raise ValueError("--hidden-dims must contain at least one integer")
  if any(dim < 1 for dim in dims):
    raise ValueError("--hidden-dims must be positive")
  return dims


def make_adapter(args: argparse.Namespace) -> JaxMARLCoinGameVectorAdapter:
  _ignored_meltingpot_args = (
    args.observation_size,
    args.include_observation_scalars,
    args.append_agent_id,
  )
  del _ignored_meltingpot_args
  return JaxMARLCoinGameVectorAdapter(
    num_envs=args.num_envs,
    max_cycles=args.max_cycles,
    seed=args.seed,
  )


def log_stage(args: argparse.Namespace, message: str) -> None:
  if not args.quiet:
    print(f"[coin-flow] {message}", flush=True)


def plot_distribution_validation(
  output_path: Path,
  *,
  validation_actions: np.ndarray,
  train_actions: np.ndarray,
  gmm_actions: np.ndarray,
  flow_actions: np.ndarray,
  uniform_actions: np.ndarray,
  action_dim: int,
  distribution_metrics: dict[str, Any],
) -> None:
  """Plot distribution fit, errors, and summary distances."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  from world_marl.coin_flow import joint_action_probabilities

  panels = [
    ("heldout", validation_actions),
    ("train", train_actions),
    ("gmm sample", gmm_actions),
    ("flow", flow_actions),
    ("uniform", uniform_actions),
  ]
  matrices = [joint_action_probabilities(actions, action_dim) for _, actions in panels]
  heldout = matrices[0]
  vmax = max(float(matrix.max()) for matrix in matrices)
  error_matrices = [np.abs(matrix - heldout) for matrix in matrices]
  error_vmax = max(float(matrix.max()) for matrix in error_matrices[1:]) or 1.0

  fig = plt.figure(figsize=(19, 10))
  grid = fig.add_gridspec(
    3,
    6,
    height_ratios=(1.0, 1.0, 0.9),
    width_ratios=(1.0, 1.0, 1.0, 1.0, 1.0, 0.08),
  )

  prob_image = None
  for column, ((title, _actions), matrix) in enumerate(zip(panels, matrices, strict=True)):
    ax = fig.add_subplot(grid[0, column])
    prob_image = ax.imshow(matrix, vmin=0.0, vmax=vmax, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("player_1 action")
    if column == 0:
      ax.set_ylabel("player_0 action")
    ax.set_xticks(range(action_dim))
    ax.set_yticks(range(action_dim))
  if prob_image is not None:
    cax = fig.add_subplot(grid[0, 5])
    fig.colorbar(prob_image, cax=cax, label="probability")

  error_image = None
  for column, ((title, _actions), matrix) in enumerate(
    zip(panels, error_matrices, strict=True)
  ):
    ax = fig.add_subplot(grid[1, column])
    error_image = ax.imshow(matrix, vmin=0.0, vmax=error_vmax, origin="lower")
    ax.set_title(f"|{title} - heldout|")
    ax.set_xlabel("player_1 action")
    if column == 0:
      ax.set_ylabel("player_0 action")
    ax.set_xticks(range(action_dim))
    ax.set_yticks(range(action_dim))
  if error_image is not None:
    cax = fig.add_subplot(grid[1, 5])
    fig.colorbar(error_image, cax=cax, label="abs error")

  rank_ax = fig.add_subplot(grid[2, :3])
  order = np.argsort(heldout.reshape(-1))[::-1]
  x = np.arange(order.shape[0])
  for title, matrix in zip(
    ("heldout", "train", "gmm sample", "flow", "uniform"),
    matrices,
    strict=True,
  ):
    rank_ax.plot(x, matrix.reshape(-1)[order], marker="o", linewidth=1.5, label=title)
  rank_ax.set_title("Action-pair probabilities sorted by heldout frequency")
  rank_ax.set_xlabel("joint-action rank")
  rank_ax.set_ylabel("probability")
  rank_ax.grid(True, alpha=0.25)
  rank_ax.legend(fontsize=8)

  metric_ax = fig.add_subplot(grid[2, 3:5])
  labels = ["train", "gmm", "flow", "uniform"]
  metric_keys = ["train_empirical", "gmm_sample", "flow", "uniform"]
  js_values = [
    distribution_metrics[key]["js_divergence"]
    for key in metric_keys
  ]
  tv_values = [
    distribution_metrics[key]["total_variation"]
    for key in metric_keys
  ]
  positions = np.arange(len(labels))
  width = 0.38
  metric_ax.bar(positions - width / 2, js_values, width, label="JS divergence")
  metric_ax.bar(positions + width / 2, tv_values, width, label="total variation")
  metric_ax.set_title("Distances to heldout distribution")
  metric_ax.set_xticks(positions, labels)
  metric_ax.grid(True, axis="y", alpha=0.25)
  metric_ax.legend(fontsize=8)
  fig.add_subplot(grid[2, 5]).axis("off")

  fig.suptitle("Flow Matching Joint-Action Distribution Validation", fontsize=14)
  fig.subplots_adjust(
    left=0.04,
    right=0.98,
    top=0.90,
    bottom=0.08,
    wspace=0.55,
    hspace=0.75,
  )
  fig.savefig(output_path)
  plt.close(fig)


def plot_conditional_action_validation(
  output_path: Path,
  *,
  heldout_actions: np.ndarray,
  train_actions: np.ndarray,
  classifier_actions: np.ndarray,
  flow_actions: np.ndarray,
  uniform_actions: np.ndarray,
  action_dim: int,
  metrics: dict[str, Any],
) -> None:
  """Plot Milestone 1 state-conditioned action validation."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  from world_marl.coin_flow import joint_action_probabilities

  panels = [
    ("heldout source", heldout_actions),
    ("train source", train_actions),
    ("classifier", classifier_actions),
    ("conditional flow", flow_actions),
    ("uniform", uniform_actions),
  ]
  matrices = [joint_action_probabilities(actions, action_dim) for _, actions in panels]
  heldout = matrices[0]
  vmax = max(float(matrix.max()) for matrix in matrices)
  errors = [np.abs(matrix - heldout) for matrix in matrices]
  error_vmax = max(float(error.max()) for error in errors[1:]) or 1.0

  fig = plt.figure(figsize=(19, 9))
  grid = fig.add_gridspec(2, 6, width_ratios=(1, 1, 1, 1, 1, 0.08))

  image = None
  for column, ((title, _actions), matrix) in enumerate(zip(panels, matrices, strict=True)):
    ax = fig.add_subplot(grid[0, column])
    image = ax.imshow(matrix, vmin=0.0, vmax=vmax, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("agent 1 action")
    if column == 0:
      ax.set_ylabel("agent 0 action")
    ax.set_xticks(range(action_dim))
    ax.set_yticks(range(action_dim))
  if image is not None:
    fig.colorbar(image, cax=fig.add_subplot(grid[0, 5]), label="probability")

  error_image = None
  for column, ((title, _actions), error) in enumerate(zip(panels, errors, strict=True)):
    ax = fig.add_subplot(grid[1, column])
    error_image = ax.imshow(error, vmin=0.0, vmax=error_vmax, origin="lower")
    ax.set_title(f"|{title} - heldout|")
    ax.set_xlabel("agent 1 action")
    if column == 0:
      ax.set_ylabel("agent 0 action")
    ax.set_xticks(range(action_dim))
    ax.set_yticks(range(action_dim))
  if error_image is not None:
    fig.colorbar(error_image, cax=fig.add_subplot(grid[1, 5]), label="abs error")

  fig.suptitle(
    (
      "Milestone 1: State-Conditioned Joint-Action Prediction "
      f"(flow joint acc={metrics['flow']['joint_accuracy']:.3f}, "
      f"classifier joint acc={metrics['classifier']['joint_accuracy']:.3f})"
    ),
    fontsize=14,
  )
  fig.subplots_adjust(
    left=0.04,
    right=0.98,
    top=0.86,
    bottom=0.08,
    wspace=0.55,
    hspace=0.45,
  )
  fig.savefig(output_path)
  plt.close(fig)


def load_checkpoint_policy(
  checkpoint_dir: str | Path,
  adapter: JaxMARLCoinGameVectorAdapter,
  *,
  deterministic: bool,
  seed: int,
):
  checkpoint_path = Path(checkpoint_dir)
  metadata = load_metadata(checkpoint_path)
  algorithm = metadata.get("algorithm", "ippo")
  config_payload = metadata.get("algorithm_config", metadata.get("ippo_config"))
  if config_payload is None:
    raise KeyError("checkpoint metadata missing algorithm_config")
  config = MAPPOConfig(**config_payload) if algorithm == "mappo" else IPPOConfig(**config_payload)

  expected_substrate = metadata.get("substrate")
  if expected_substrate is not None and expected_substrate != adapter.substrate:
    raise ValueError(
      f"checkpoint substrate {expected_substrate!r} does not match "
      f"adapter substrate {adapter.substrate!r}"
    )
  expected_action_dim = metadata.get("action_dim")
  if expected_action_dim is not None and int(expected_action_dim) != adapter.action_dim:
    raise ValueError(
      f"checkpoint action_dim {expected_action_dim} does not match "
      f"adapter action_dim {adapter.action_dim}"
    )
  expected_num_agents = metadata.get("num_agents")
  if expected_num_agents is not None and int(expected_num_agents) != adapter.num_agents:
    raise ValueError(
      f"checkpoint num_agents {expected_num_agents} does not match "
      f"adapter num_agents {adapter.num_agents}"
    )
  expected_observation_shape = metadata.get("observation_shape")
  if expected_observation_shape is not None:
    expected_observation_shape = tuple(int(dim) for dim in expected_observation_shape)
    if expected_observation_shape != adapter.observation_shape:
      raise ValueError(
        "checkpoint observation_shape "
        f"{expected_observation_shape} does not match JaxMARL CoinGame "
        f"adapter observation_shape {adapter.observation_shape}."
      )
  observation_mode = metadata.get("observation_mode", "vector")
  if observation_mode != "vector":
    raise ValueError(
      "JaxMARL CoinGame flow validation expects vector-mode checkpoints; "
      f"got observation_mode={observation_mode!r}"
    )

  train_state = create_algorithm_train_state(
    algorithm,
    jax.random.PRNGKey(0),
    adapter,
    config,
    observation_mode=observation_mode,
  )
  params = load_params(checkpoint_path / "checkpoint.msgpack", train_state.params)
  train_state = train_state.replace(params=params)
  return (
    policy_from_train_state(
      algorithm,
      train_state,
      adapter=adapter,
      deterministic=deterministic,
      seed=seed,
      observation_mode=observation_mode,
    ),
    metadata,
  )


def run_conditional_action_validation(
  args: argparse.Namespace,
  *,
  hidden_dims: tuple[int, ...],
  classifier_hidden_dims: tuple[int, ...],
  run_dir: Path,
  logger: RunLogger,
  np_rng: np.random.Generator,
) -> None:
  """Milestone 1: learn and validate p(joint_action | state)."""
  source_metadata: dict[str, Any] | None = None
  adapter = make_adapter(args)
  try:
    log_stage(args, "constructing JaxMARL CoinGame adapter")
    log_stage(
      args,
      (
        f"collecting {args.collect_steps} state-action rollout steps "
        f"({args.collect_steps * args.num_envs} samples) from {args.target_source}"
      ),
    )
    if args.target_source == "checkpoint":
      log_stage(args, f"loading source policy checkpoint from {args.policy_checkpoint}")
      source_policy, source_metadata = load_checkpoint_policy(
        args.policy_checkpoint,
        adapter,
        deterministic=not args.source_stochastic,
        seed=args.seed + 10,
      )
    else:
      source_policy = None

    with tqdm(
      total=args.collect_steps,
      desc="collect state-actions",
      unit="step",
      disable=args.quiet,
    ) as progress:
      if source_policy is None:
        dataset = collect_random_state_actions(
          adapter,
          np_rng,
          rollout_steps=args.collect_steps,
          progress_callback=lambda _step: progress.update(1),
        )
      else:
        dataset = collect_policy_state_actions(
          adapter,
          source_policy,
          rollout_steps=args.collect_steps,
          progress_callback=lambda _step: progress.update(1),
        )
    env_metadata = {
      "substrate": adapter.substrate,
      "num_agents": adapter.num_agents,
      "action_dim": adapter.action_dim,
      "observation_shape": adapter.observation_shape,
      "raw_observation_shape": adapter.raw_observation_shape,
      "scalar_observation_keys": adapter.scalar_observation_keys,
      "environment_family": "jaxmarl_coin_game",
    }
  finally:
    adapter.close()

  log_stage(
    args,
    (
      "collected "
      f"{dataset.joint_actions.shape[0]} state-action samples; "
      f"mean reward per agent={dataset.rewards.mean(axis=0).round(4).tolist()}"
    ),
  )
  logger.write_json(
    "conditional_action_dataset.json",
    {
      **dataset.to_metadata(),
      "env": env_metadata,
      "target_source": args.target_source,
      "source_checkpoint_metadata": source_metadata,
      "validation_fraction": args.validation_fraction,
      "target": "p(joint_action | state)",
    },
  )

  (
    train_features,
    train_actions,
    validation_features,
    validation_actions,
  ) = split_state_action_dataset(
    dataset,
    validation_fraction=args.validation_fraction,
    seed=args.seed,
  )
  normalizer = fit_feature_normalizer(train_features)
  train_features_norm = normalizer.transform(train_features)
  validation_features_norm = normalizer.transform(validation_features)
  log_stage(
    args,
    (
      f"split into {train_actions.shape[0]} train and "
      f"{validation_actions.shape[0]} heldout state-action samples"
    ),
  )
  logger.write_json(
    "conditional_action_split.json",
    {
      "train_samples": int(train_actions.shape[0]),
      "heldout_samples": int(validation_actions.shape[0]),
      "state_feature_dim": int(train_features.shape[1]),
      "validation_fraction": args.validation_fraction,
      "normalizer": normalizer.to_metadata(),
    },
  )

  classifier_lr = (
    args.learning_rate
    if args.classifier_learning_rate is None
    else args.classifier_learning_rate
  )
  log_stage(args, f"training categorical action baseline for {args.train_steps} steps")
  with tqdm(
    total=args.train_steps,
    desc="train classifier",
    unit="step",
    disable=args.quiet,
  ) as progress:
    def update_classifier_progress(_step: int, loss: float) -> None:
      progress.update(1)
      progress.set_postfix(loss=f"{loss:.4g}")

    classifier_state, classifier_losses = train_action_classifier(
      jax.random.PRNGKey(args.seed + 100),
      train_features_norm,
      train_actions,
      action_dim=dataset.action_dim,
      num_agents=dataset.num_agents,
      train_steps=args.train_steps,
      batch_size=args.batch_size,
      learning_rate=classifier_lr,
      hidden_dims=classifier_hidden_dims,
      progress_callback=update_classifier_progress,
    )
  classifier_logits = predict_action_logits(classifier_state, validation_features_norm)
  classifier_predictions = classifier_logits.argmax(axis=-1).astype(np.int32)
  classifier_metrics = action_prediction_metrics(
    logits=classifier_logits,
    reference_actions=validation_actions,
    train_actions=train_actions,
    action_dim=dataset.action_dim,
  )
  for step, loss in enumerate(classifier_losses, start=1):
    logger.append_metrics({"step": step, "classifier/loss": loss})

  log_stage(args, f"training conditional flow for {args.train_steps} steps")
  with tqdm(
    total=args.train_steps,
    desc="train conditional flow",
    unit="step",
    disable=args.quiet,
  ) as progress:
    def update_flow_progress(_step: int, loss: float) -> None:
      progress.update(1)
      progress.set_postfix(loss=f"{loss:.4g}")

    flow_state, flow_losses = train_conditional_action_flow(
      jax.random.PRNGKey(args.seed + 200),
      train_features_norm,
      train_actions,
      action_dim=dataset.action_dim,
      train_steps=args.train_steps,
      batch_size=args.batch_size,
      learning_rate=args.learning_rate,
      hidden_dims=hidden_dims,
      progress_callback=update_flow_progress,
    )
  for step, loss in enumerate(flow_losses, start=1):
    logger.append_metrics({"step": step, "conditional_flow/loss": loss})

  flow_eval_size = min(args.generated_samples, validation_actions.shape[0])
  eval_indices = np.random.default_rng(args.seed + 30).choice(
    validation_actions.shape[0],
    size=flow_eval_size,
    replace=False,
  )
  validation_features_eval = validation_features_norm[eval_indices]
  validation_actions_eval = validation_actions[eval_indices]
  classifier_predictions_eval = classifier_predictions[eval_indices]

  log_stage(
    args,
    f"sampling conditional flow actions on {flow_eval_size} heldout states",
  )
  flow_sample_key = jax.random.PRNGKey(args.seed + 300)
  flow_points = np.asarray(
    sample_conditional_action_flow_points(
      flow_state,
      flow_sample_key,
      validation_features_eval,
      integration_steps=args.flow_integration_steps,
    ),
    dtype=np.float32,
  )
  flow_actions = decode_joint_actions(flow_points, dataset.action_dim)
  uniform_actions = uniform_joint_actions(
    np.random.default_rng(args.seed + 40),
    num_samples=flow_eval_size,
    action_dim=dataset.action_dim,
  )
  flow_metrics = sampled_action_prediction_metrics(
    sampled_actions=flow_actions,
    reference_actions=validation_actions_eval,
    train_actions=train_actions,
    action_dim=dataset.action_dim,
    top_k=args.distribution_top_k,
  )
  uniform_metrics = sampled_action_prediction_metrics(
    sampled_actions=uniform_actions,
    reference_actions=validation_actions_eval,
    train_actions=train_actions,
    action_dim=dataset.action_dim,
    top_k=args.distribution_top_k,
  )
  flow_distribution_beats_uniform = (
    flow_metrics["distribution_vs_heldout"]["js_divergence"]
    < uniform_metrics["distribution_vs_heldout"]["js_divergence"]
  )
  flow_per_agent_beats_marginal = (
    flow_metrics["per_agent_accuracy"] > flow_metrics["marginal_per_agent_accuracy"]
  )

  log_stage(args, "saving and reloading conditional checkpoints")
  save_checkpoint(
    run_dir / "conditional_classifier_checkpoint",
    classifier_state,
    metadata={
      "kind": "coin_state_conditioned_action_classifier",
      "target": "p(joint_action | state)",
      "target_source": args.target_source,
      "source_checkpoint": args.policy_checkpoint,
      "environment_family": "jaxmarl_coin_game",
      "action_dim": dataset.action_dim,
      "num_agents": dataset.num_agents,
      "state_feature_dim": int(train_features.shape[1]),
      "hidden_dims": classifier_hidden_dims,
      "normalizer": normalizer.to_metadata(),
      "config": vars(args),
    },
  )
  save_checkpoint(
    run_dir / "conditional_flow_checkpoint",
    flow_state,
    metadata={
      "kind": "coin_state_conditioned_action_flow",
      "target": "p(joint_action | state)",
      "target_source": args.target_source,
      "source_checkpoint": args.policy_checkpoint,
      "environment_family": "jaxmarl_coin_game",
      "action_dim": dataset.action_dim,
      "num_agents": dataset.num_agents,
      "state_feature_dim": int(train_features.shape[1]),
      "hidden_dims": hidden_dims,
      "normalizer": normalizer.to_metadata(),
      "flow_integration_steps": args.flow_integration_steps,
      "config": vars(args),
    },
  )

  classifier_reload_state = create_action_classifier_train_state(
    jax.random.PRNGKey(args.seed + 400),
    feature_dim=train_features.shape[1],
    action_dim=dataset.action_dim,
    num_agents=dataset.num_agents,
    hidden_dims=classifier_hidden_dims,
    learning_rate=classifier_lr,
  )
  classifier_reload_params = load_params(
    run_dir / "conditional_classifier_checkpoint" / "checkpoint.msgpack",
    classifier_reload_state.params,
  )
  classifier_reload_state = classifier_reload_state.replace(
    params=classifier_reload_params
  )
  classifier_reload_logits = predict_action_logits(
    classifier_reload_state,
    validation_features_norm,
  )
  classifier_reload_max_abs_diff = float(
    np.max(np.abs(classifier_reload_logits - classifier_logits))
  )

  flow_reload_state = create_conditioned_train_state(
    jax.random.PRNGKey(args.seed + 500),
    MLPVectorField(hidden_dims=hidden_dims),
    args.learning_rate,
    dim=2,
    cond_dim=train_features.shape[1],
  )
  flow_reload_params = load_params(
    run_dir / "conditional_flow_checkpoint" / "checkpoint.msgpack",
    flow_reload_state.params,
  )
  flow_reload_state = flow_reload_state.replace(params=flow_reload_params)
  flow_reload_points = np.asarray(
    sample_conditional_action_flow_points(
      flow_reload_state,
      flow_sample_key,
      validation_features_eval,
      integration_steps=args.flow_integration_steps,
    ),
    dtype=np.float32,
  )
  flow_reload_max_abs_diff = float(np.max(np.abs(flow_reload_points - flow_points)))
  reload_passed = (
    classifier_reload_max_abs_diff <= 1e-6 and flow_reload_max_abs_diff <= 1e-6
  )

  finite_losses = bool(
    np.isfinite(classifier_losses).all() and np.isfinite(flow_losses).all()
  )
  criteria = {
    "finite_losses": finite_losses,
    "classifier_beats_marginal_ce": classifier_metrics["model_beats_marginal_ce"],
    "flow_distribution_beats_uniform": bool(flow_distribution_beats_uniform),
    "flow_per_agent_accuracy_beats_marginal": bool(flow_per_agent_beats_marginal),
    "reload_passed": bool(reload_passed),
  }
  passed = bool(all(criteria.values()))

  validation_payload = {
    "milestone": "conditional_action_imitation",
    "target": "p(joint_action | state)",
    "passed": passed,
    "criteria": criteria,
    "classifier": classifier_metrics,
    "flow": flow_metrics,
    "uniform": uniform_metrics,
    "reload": {
      "classifier_max_abs_logit_diff": classifier_reload_max_abs_diff,
      "flow_max_abs_point_diff": flow_reload_max_abs_diff,
    },
    "training": {
      "classifier_initial_loss": classifier_losses[0],
      "classifier_final_loss": classifier_losses[-1],
      "classifier_min_loss": min(classifier_losses),
      "flow_initial_loss": flow_losses[0],
      "flow_final_loss": flow_losses[-1],
      "flow_min_loss": min(flow_losses),
      "train_steps": args.train_steps,
    },
  }
  logger.write_json("conditional_action_validation.json", validation_payload)
  logger.write_json(
    "conditional_action_samples.json",
    {
      "heldout_actions": validation_actions_eval.astype(int).tolist(),
      "classifier_actions": classifier_predictions_eval.astype(int).tolist(),
      "flow_points": flow_points.astype(float).tolist(),
      "flow_actions": flow_actions.astype(int).tolist(),
      "uniform_actions": uniform_actions.astype(int).tolist(),
    },
  )
  plot_conditional_action_validation(
    run_dir / "conditional_action_validation.png",
    heldout_actions=validation_actions_eval,
    train_actions=train_actions,
    classifier_actions=classifier_predictions_eval,
    flow_actions=flow_actions,
    uniform_actions=uniform_actions,
    action_dim=dataset.action_dim,
    metrics={"classifier": classifier_metrics, "flow": flow_metrics},
  )

  random_eval = None
  source_eval = None
  classifier_eval = None
  flow_eval = None
  if not args.skip_policy_eval:
    log_stage(args, f"evaluating policies for {args.eval_episodes} episodes")
    eval_adapter = make_adapter(args)
    try:
      random_eval = evaluate_policy(
        eval_adapter,
        random_policy(eval_adapter, np.random.default_rng(args.seed + 1)),
        episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
      )
      if args.target_source == "checkpoint":
        source_policy, _ = load_checkpoint_policy(
          args.policy_checkpoint,
          eval_adapter,
          deterministic=not args.source_stochastic,
          seed=args.seed + 3,
        )
        source_eval = evaluate_policy(
          eval_adapter,
          source_policy,
          episodes=args.eval_episodes,
          max_steps=args.eval_max_steps,
        )
      classifier_eval = evaluate_policy(
        eval_adapter,
        classifier_joint_action_policy(
          classifier_state,
          normalizer,
          deterministic=True,
          seed=args.seed + 4,
        ),
        episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
      )
      flow_eval = evaluate_policy(
        eval_adapter,
        conditional_flow_joint_action_policy(
          flow_state,
          normalizer,
          action_dim=eval_adapter.action_dim,
          seed=args.seed + 5,
          integration_steps=args.flow_integration_steps,
        ),
        episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
      )
    finally:
      eval_adapter.close()

  outcome: dict[str, Any] = {
    "milestone": "conditional_action_imitation",
    "target": "p(joint_action | state)",
    "passed": passed,
    "criteria": criteria,
    "prediction_validation": {
      "classifier_cross_entropy": classifier_metrics["cross_entropy"],
      "classifier_marginal_cross_entropy": classifier_metrics[
        "marginal_cross_entropy"
      ],
      "classifier_per_agent_accuracy": classifier_metrics["per_agent_accuracy"],
      "classifier_joint_accuracy": classifier_metrics["joint_accuracy"],
      "flow_per_agent_accuracy": flow_metrics["per_agent_accuracy"],
      "flow_joint_accuracy": flow_metrics["joint_accuracy"],
      "flow_js_divergence": flow_metrics["distribution_vs_heldout"][
        "js_divergence"
      ],
      "uniform_js_divergence": uniform_metrics["distribution_vs_heldout"][
        "js_divergence"
      ],
      "plot": str(run_dir / "conditional_action_validation.png"),
    },
    "random": random_eval.to_dict() if random_eval is not None else None,
    "source": source_eval.to_dict() if source_eval is not None else None,
    "classifier": (
      classifier_eval.to_dict() if classifier_eval is not None else None
    ),
    "flow": flow_eval.to_dict() if flow_eval is not None else None,
    "classifier_minus_random_mean_return_per_agent": (
      classifier_eval.mean_return_per_agent - random_eval.mean_return_per_agent
      if classifier_eval is not None and random_eval is not None
      else None
    ),
    "flow_minus_random_mean_return_per_agent": (
      flow_eval.mean_return_per_agent - random_eval.mean_return_per_agent
      if flow_eval is not None and random_eval is not None
      else None
    ),
    "flow_minus_source_mean_return_per_agent": (
      flow_eval.mean_return_per_agent - source_eval.mean_return_per_agent
      if flow_eval is not None and source_eval is not None
      else None
    ),
  }
  logger.write_json("evaluation.json", outcome)
  log_stage(
    args,
    (
      "conditional validation complete; "
      f"flow_acc={flow_metrics['per_agent_accuracy']:.4g}, "
      f"classifier_acc={classifier_metrics['per_agent_accuracy']:.4g}, "
      f"passed={passed}"
    ),
  )
  log_stage(args, "done")
  print(logger.write_json("outcome.json", outcome).read_text(encoding="utf-8"))


def main() -> None:
  args = parse_args()
  if args.substrate != "coins":
    raise SystemExit("world-marl-train-coin-flow currently targets --substrate coins")
  if args.target_source == "checkpoint" and args.policy_checkpoint is None:
    raise SystemExit("--policy-checkpoint is required with --target-source checkpoint")

  hidden_dims = parse_hidden_dims(args.hidden_dims)
  classifier_hidden_dims = parse_hidden_dims(args.classifier_hidden_dims)
  run_dir = Path(args.out_dir) / f"coin_flow_{timestamp()}"
  log_stage(args, f"writing artifacts to {run_dir}")
  logger = RunLogger(run_dir)
  logger.write_json(
    "config.json",
    {
      "args": vars(args),
      "hidden_dims": hidden_dims,
      "classifier_hidden_dims": classifier_hidden_dims,
      "target_source": args.target_source,
      "purpose": (
        "Validate whether flow matching can learn p(joint_action | state) "
        "from JaxMARL CoinGame rollouts."
        if args.conditional_action
        else (
          "Validate whether flow matching can learn two-agent joint-action "
          "distributions from JaxMARL CoinGame rollouts. This is distribution "
          "prediction/sampling validation, not world modeling."
        )
      ),
    },
  )
  logger.write_json("versions.json", dependency_versions())

  if args.conditional_action:
    run_conditional_action_validation(
      args,
      hidden_dims=hidden_dims,
      classifier_hidden_dims=classifier_hidden_dims,
      run_dir=run_dir,
      logger=logger,
      np_rng=np.random.default_rng(args.seed),
    )
    return

  log_stage(args, "constructing JaxMARL CoinGame adapter")
  np_rng = np.random.default_rng(args.seed)
  source_metadata: dict[str, Any] | None = None
  adapter = make_adapter(args)
  try:
    log_stage(
      args,
      (
        f"collecting {args.collect_steps} rollout steps "
        f"({args.collect_steps * args.num_envs} joint-action samples) "
        f"from {args.target_source}"
      ),
    )
    if args.target_source == "checkpoint":
      log_stage(args, f"loading source policy checkpoint from {args.policy_checkpoint}")
      source_policy, source_metadata = load_checkpoint_policy(
        args.policy_checkpoint,
        adapter,
        deterministic=not args.source_stochastic,
        seed=args.seed + 10,
      )
    else:
      source_policy = None

    with tqdm(
      total=args.collect_steps,
      desc="collect rollouts",
      unit="step",
      disable=args.quiet,
    ) as progress:
      if source_policy is None:
        dataset = collect_random_joint_actions(
          adapter,
          np_rng,
          rollout_steps=args.collect_steps,
          progress_callback=lambda _step: progress.update(1),
        )
      else:
        dataset = collect_policy_joint_actions(
          adapter,
          source_policy,
          rollout_steps=args.collect_steps,
          progress_callback=lambda _step: progress.update(1),
        )
    env_metadata = {
      "substrate": adapter.substrate,
      "num_agents": adapter.num_agents,
      "action_dim": adapter.action_dim,
      "observation_shape": adapter.observation_shape,
      "raw_observation_shape": adapter.raw_observation_shape,
      "scalar_observation_keys": adapter.scalar_observation_keys,
      "environment_family": "jaxmarl_coin_game",
    }
  finally:
    adapter.close()

  log_stage(
    args,
    (
      "collected "
      f"{dataset.joint_actions.shape[0]} joint actions; "
      f"mean reward per agent={dataset.rewards.mean(axis=0).round(4).tolist()}"
    ),
  )
  logger.write_json(
    "rollout_dataset.json",
    {
      **dataset.to_metadata(),
      "env": env_metadata,
      "target_source": args.target_source,
      "source_checkpoint_metadata": source_metadata,
      "validation_fraction": args.validation_fraction,
    },
  )

  train_actions, validation_actions = split_joint_actions(
    dataset.joint_actions,
    validation_fraction=args.validation_fraction,
    seed=args.seed,
  )
  log_stage(
    args,
    (
      f"split joint actions into {train_actions.shape[0]} train and "
      f"{validation_actions.shape[0]} heldout samples"
    ),
  )
  logger.write_json(
    "distribution_split.json",
    {
      "train_samples": int(train_actions.shape[0]),
      "heldout_samples": int(validation_actions.shape[0]),
      "validation_fraction": args.validation_fraction,
      "seed": args.seed,
    },
  )

  log_stage(args, "fitting empirical GMM over train joint actions")
  fitted = fit_joint_action_gmm(
    train_actions,
    action_dim=dataset.action_dim,
    std=args.gmm_std,
    max_components=args.max_components,
  )
  log_stage(args, f"fitted {fitted.action_pairs.shape[0]} GMM components")
  logger.write_json("gmm.json", fitted.to_metadata())

  log_stage(args, f"training flow model for {args.train_steps} steps")
  rng = jax.random.PRNGKey(args.seed)
  with tqdm(
    total=args.train_steps,
    desc="train flow",
    unit="step",
    disable=args.quiet,
  ) as progress:
    def update_training_progress(_step: int, loss: float) -> None:
      progress.update(1)
      progress.set_postfix(loss=f"{loss:.4g}")

    train_state, losses = train_flow_for_gmm(
      rng,
      fitted.gmm,
      train_steps=args.train_steps,
      batch_size=args.batch_size,
      learning_rate=args.learning_rate,
      hidden_dims=hidden_dims,
      progress_callback=update_training_progress,
    )
  log_stage(
    args,
    (
      f"flow training complete; initial_loss={losses[0]:.6g}, "
      f"final_loss={losses[-1]:.6g}, min_loss={min(losses):.6g}"
    ),
  )
  for step, loss in enumerate(losses, start=1):
    logger.append_metrics({"step": step, "flow/loss": loss})
  logger.write_json(
    "training_summary.json",
    {
      "initial_loss": losses[0],
      "final_loss": losses[-1],
      "min_loss": min(losses),
      "train_steps": args.train_steps,
    },
  )

  log_stage(args, f"sampling {args.generated_samples} points from learned flow")
  rng, sample_key = jax.random.split(rng)
  generated_points = np.asarray(
    sample_flow_points(
      train_state,
      sample_key,
      num_samples=args.generated_samples,
      integration_steps=args.flow_integration_steps,
    )
  )
  generated_actions = decode_joint_actions(generated_points, dataset.action_dim)
  unique_actions, generated_counts = np.unique(
    generated_actions,
    axis=0,
    return_counts=True,
  )
  rng, gmm_sample_key = jax.random.split(rng)
  gmm_actions = decode_joint_actions(
    np.asarray(sample_gmm(gmm_sample_key, fitted.gmm, args.generated_samples)),
    dataset.action_dim,
  )
  uniform_actions = uniform_joint_actions(
    np.random.default_rng(args.seed + 20),
    num_samples=args.generated_samples,
    action_dim=dataset.action_dim,
  )
  logger.write_json(
    "generated_action_samples.json",
    {
      "points": generated_points.tolist(),
      "actions": generated_actions.astype(int).tolist(),
      "unique_action_pairs": unique_actions.astype(int).tolist(),
      "unique_action_counts": generated_counts.astype(int).tolist(),
      "gmm_actions": gmm_actions.astype(int).tolist(),
      "uniform_actions": uniform_actions.astype(int).tolist(),
    },
  )

  log_stage(args, "saving flow checkpoint")
  save_checkpoint(
    run_dir / "checkpoint",
    train_state,
    metadata={
      "kind": "coin_joint_action_flow",
      "target_source": args.target_source,
      "source_checkpoint": args.policy_checkpoint,
      "substrate": args.substrate,
      "environment_family": "jaxmarl_coin_game",
      "action_dim": dataset.action_dim,
      "num_agents": dataset.num_agents,
      "gmm": fitted.to_metadata(),
      "hidden_dims": hidden_dims,
      "flow_integration_steps": args.flow_integration_steps,
      "validation_fraction": args.validation_fraction,
      "config": vars(args),
    },
  )

  log_stage(args, "reloading flow checkpoint and validating sampled distribution")
  reload_state = create_flow_train_state(
    jax.random.PRNGKey(args.seed + 1000),
    MLPVectorField(hidden_dims=hidden_dims),
    learning_rate=args.learning_rate,
    dim=fitted.gmm.dim,
  )
  reload_params = load_params(
    run_dir / "checkpoint" / "checkpoint.msgpack",
    reload_state.params,
  )
  reload_state = reload_state.replace(params=reload_params)
  reload_points = np.asarray(
    sample_flow_points(
      reload_state,
      sample_key,
      num_samples=args.generated_samples,
      integration_steps=args.flow_integration_steps,
    )
  )
  reload_actions = decode_joint_actions(reload_points, dataset.action_dim)
  reload_max_abs_point_diff = float(np.max(np.abs(reload_points - generated_points)))

  distribution_metrics = {
    "train_empirical": compare_joint_action_distributions(
      validation_actions,
      train_actions,
      action_dim=dataset.action_dim,
      top_k=args.distribution_top_k,
    ),
    "flow": compare_joint_action_distributions(
      validation_actions,
      generated_actions,
      action_dim=dataset.action_dim,
      top_k=args.distribution_top_k,
    ),
    "gmm_sample": compare_joint_action_distributions(
      validation_actions,
      gmm_actions,
      action_dim=dataset.action_dim,
      top_k=args.distribution_top_k,
    ),
    "uniform": compare_joint_action_distributions(
      validation_actions,
      uniform_actions,
      action_dim=dataset.action_dim,
      top_k=args.distribution_top_k,
    ),
    "reload_flow": compare_joint_action_distributions(
      validation_actions,
      reload_actions,
      action_dim=dataset.action_dim,
      top_k=args.distribution_top_k,
    ),
  }
  strict_flow_beats_uniform = (
    distribution_metrics["flow"]["js_divergence"]
    < distribution_metrics["uniform"]["js_divergence"]
  )
  reload_passed = reload_max_abs_point_diff <= 1e-6
  distribution_payload = {
    "passed": bool(strict_flow_beats_uniform and reload_passed),
    "strict_flow_beats_uniform": bool(strict_flow_beats_uniform),
    "reload_passed": bool(reload_passed),
    "reload_max_abs_point_diff": reload_max_abs_point_diff,
    "train_distribution": summarize_joint_action_distribution(
      train_actions,
      dataset.action_dim,
      top_k=args.distribution_top_k,
    ),
    "heldout_distribution": summarize_joint_action_distribution(
      validation_actions,
      dataset.action_dim,
      top_k=args.distribution_top_k,
    ),
    "flow_distribution": summarize_joint_action_distribution(
      generated_actions,
      dataset.action_dim,
      top_k=args.distribution_top_k,
    ),
    "gmm_sample_distribution": summarize_joint_action_distribution(
      gmm_actions,
      dataset.action_dim,
      top_k=args.distribution_top_k,
    ),
    "uniform_distribution": summarize_joint_action_distribution(
      uniform_actions,
      dataset.action_dim,
      top_k=args.distribution_top_k,
    ),
    "metrics_vs_heldout": distribution_metrics,
  }
  logger.write_json("distribution_validation.json", distribution_payload)
  plot_distribution_validation(
    run_dir / "distribution_validation.png",
    validation_actions=validation_actions,
    train_actions=train_actions,
    gmm_actions=gmm_actions,
    flow_actions=generated_actions,
    uniform_actions=uniform_actions,
    action_dim=dataset.action_dim,
    distribution_metrics=distribution_metrics,
  )
  log_stage(
    args,
    (
      "distribution validation complete; "
      f"flow_js={distribution_metrics['flow']['js_divergence']:.6g}, "
      f"uniform_js={distribution_metrics['uniform']['js_divergence']:.6g}, "
      f"reload_diff={reload_max_abs_point_diff:.3g}"
    ),
  )

  random_eval = None
  source_eval = None
  flow_eval = None
  if not args.skip_policy_eval:
    log_stage(args, f"evaluating random and flow policies for {args.eval_episodes} episodes")
    eval_adapter = make_adapter(args)
    try:
      random_eval = evaluate_policy(
        eval_adapter,
        random_policy(eval_adapter, np.random.default_rng(args.seed + 1)),
        episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
      )
      if args.target_source == "checkpoint":
        source_policy, _ = load_checkpoint_policy(
          args.policy_checkpoint,
          eval_adapter,
          deterministic=not args.source_stochastic,
          seed=args.seed + 3,
        )
        source_eval = evaluate_policy(
          eval_adapter,
          source_policy,
          episodes=args.eval_episodes,
          max_steps=args.eval_max_steps,
        )
      flow_eval = evaluate_policy(
        eval_adapter,
        flow_joint_action_policy(
          train_state,
          num_envs=eval_adapter.num_envs,
          action_dim=eval_adapter.action_dim,
          seed=args.seed + 2,
          integration_steps=args.flow_integration_steps,
        ),
        episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
      )
    finally:
      eval_adapter.close()

  outcome: dict[str, Any] = {
    "distribution_validation": {
      "passed": distribution_payload["passed"],
      "strict_flow_beats_uniform": distribution_payload["strict_flow_beats_uniform"],
      "reload_passed": distribution_payload["reload_passed"],
      "flow_js_divergence": distribution_metrics["flow"]["js_divergence"],
      "uniform_js_divergence": distribution_metrics["uniform"]["js_divergence"],
      "gmm_sample_js_divergence": distribution_metrics["gmm_sample"]["js_divergence"],
      "train_empirical_js_divergence": distribution_metrics[
        "train_empirical"
      ]["js_divergence"],
      "reload_max_abs_point_diff": reload_max_abs_point_diff,
      "plot": str(run_dir / "distribution_validation.png"),
    },
    "random": random_eval.to_dict() if random_eval is not None else None,
    "source": source_eval.to_dict() if source_eval is not None else None,
    "flow": flow_eval.to_dict() if flow_eval is not None else None,
    "flow_minus_random_mean_return_per_agent": (
      flow_eval.mean_return_per_agent - random_eval.mean_return_per_agent
      if flow_eval is not None and random_eval is not None
      else None
    ),
    "flow_minus_source_mean_return_per_agent": (
      flow_eval.mean_return_per_agent - source_eval.mean_return_per_agent
      if flow_eval is not None and source_eval is not None
      else None
    ),
  }
  logger.write_json("evaluation.json", outcome)
  if flow_eval is not None and random_eval is not None:
    source_fragment = (
      f"source={source_eval.mean_return_per_agent:.4g}, "
      if source_eval is not None
      else ""
    )
    log_stage(
      args,
      (
        "evaluation complete; "
        f"random={random_eval.mean_return_per_agent:.4g}, "
        f"{source_fragment}"
        f"flow={flow_eval.mean_return_per_agent:.4g}, "
        f"delta={outcome['flow_minus_random_mean_return_per_agent']:.4g}"
      ),
    )
  log_stage(args, "done")
  print(logger.write_json("outcome.json", outcome).read_text(encoding="utf-8"))


if __name__ == "__main__":
  main()
