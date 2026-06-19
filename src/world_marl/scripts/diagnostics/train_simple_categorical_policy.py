"""Train a simple categorical CoinGame next-state diagnostic baseline."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from baselines.softmax_model import (
    SoftmaxBaselineConfig,
    create_softmax_train_state,
    decode_coin_positions,
    evaluate_softmax_baseline,
    predict_softmax_baseline,
    prepare_softmax_data,
    sample_predictions,
    split_softmax_data,
    summarize_softmax_outcome,
    train_softmax_baseline,
)
from world_marl.checkpoint.train_state import load_params, save_checkpoint
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.logging import RunLogger, dependency_versions, timestamp
from world_marl.visualize import (
    build_next_state_comparison,
    plot_next_state_comparison,
)
from world_marl.world_model import (
    VectorTransitionBatch,
    VectorWorldModelConfig,
    make_world_model_train_state,
    predict_next,
)
from world_marl.world_model_training import (
    collect_random_transition_batch,
    fit_world_model_steps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate", default="coins")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--collect-steps", type=int, default=2048)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", default="256,256")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--stochastic-target-weight",
        type=float,
        default=32.0,
        help="Loss weight for respawn/reset distribution targets.",
    )
    parser.add_argument(
        "--min-deterministic-exact",
        type=float,
        default=0.95,
        help="Pass threshold for exact next-state accuracy on deterministic transitions.",
    )
    parser.add_argument(
        "--max-respawn-uniform-kl",
        type=float,
        default=0.25,
        help="Pass threshold for KL(uniform respawn target || model distribution).",
    )
    parser.add_argument("--sample-predictions", type=int, default=16)
    parser.add_argument(
        "--flow-samples",
        type=int,
        default=8,
        help="Flow next-state samples drawn per validation transition.",
    )
    parser.add_argument(
        "--wm-fit-steps",
        type=int,
        default=800,
        help="Full-batch fitting steps for the flow world model.",
    )
    parser.add_argument("--wm-hidden-dim", type=int, default=128)
    parser.add_argument("--wm-integration-steps", type=int, default=8)
    parser.add_argument("--wm-learning-rate", type=float, default=1e-3)
    parser.add_argument("--wm-flow-type", default="gaussian")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.substrate != "coins":
        parser.error(
            "simple categorical diagnostic currently targets --substrate coins"
        )
    if args.collect_steps < 1:
        parser.error("--collect-steps must be >= 1")
    if args.train_steps < 1:
        parser.error("--train-steps must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    if not 0.0 <= args.min_deterministic_exact <= 1.0:
        parser.error("--min-deterministic-exact must be in [0, 1]")
    if args.max_respawn_uniform_kl < 0.0:
        parser.error("--max-respawn-uniform-kl must be non-negative")
    if args.stochastic_target_weight <= 0.0:
        parser.error("--stochastic-target-weight must be positive")
    if args.flow_samples < 1:
        parser.error("--flow-samples must be >= 1")
    if args.wm_fit_steps < 1:
        parser.error("--wm-fit-steps must be >= 1")
    if args.wm_hidden_dim < 1:
        parser.error("--wm-hidden-dim must be >= 1")
    if args.wm_integration_steps < 1:
        parser.error("--wm-integration-steps must be >= 1")
    return args


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not dims:
        raise ValueError("--hidden-dims must contain at least one integer")
    if any(dim < 1 for dim in dims):
        raise ValueError("--hidden-dims must be positive")
    return dims


def log_stage(args: argparse.Namespace, message: str) -> None:
    if not args.quiet:
        print(f"[coin-softmax] {message}", flush=True)


def main() -> None:
    args = parse_args()
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    config = SoftmaxBaselineConfig(
        hidden_dims=hidden_dims,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        max_grad_norm=args.max_grad_norm,
        stochastic_target_weight=args.stochastic_target_weight,
    )
    run_dir = Path(args.out_dir) / f"coin_softmax_{timestamp()}"
    log_stage(args, f"writing artifacts to {run_dir}")
    logger = RunLogger(run_dir)
    logger.write_json(
        "config.json",
        {
            "args": vars(args),
            "model_config": dataclasses.asdict(config),
            "target": "p(next_joint_state | state, joint_action)",
            "purpose": (
                "Diagnostic categorical next-state baseline for CoinGame. "
                "Stochastic coin respawns and terminal resets are trained as "
                "distributions."
            ),
        },
    )
    logger.write_json("versions.json", dependency_versions())

    adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=args.seed,
    )
    try:
        observations = adapter.reset()
        log_stage(
            args,
            (
                f"collecting {args.collect_steps} random transition steps "
                f"({args.collect_steps * args.num_envs} samples)"
            ),
        )
        batch, _, _ = collect_random_transition_batch(
            adapter,
            observations,
            np.random.default_rng(args.seed),
            rollout_steps=args.collect_steps,
        )
    finally:
        adapter.close()

    data = prepare_softmax_data(batch)
    train_data, validation_data = split_softmax_data(
        data,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    logger.write_json(
        "transition_dataset.json",
        {
            "num_transitions": data.num_transitions,
            "train_transitions": train_data.num_transitions,
            "validation_transitions": validation_data.num_transitions,
            "num_envs": args.num_envs,
            "rollout_steps": args.collect_steps,
            "action_dim": data.action_dim,
            "num_agents": data.num_agents,
            "mean_reward_by_agent": data.rewards.mean(axis=0).astype(float).tolist(),
            "done_fraction": float(np.any(data.dones > 0.0, axis=1).mean()),
        },
    )

    log_stage(args, f"training categorical baseline for {args.train_steps} steps")
    train_bar = (
        tqdm(total=args.train_steps, desc="train softmax", unit="step")
        if not args.quiet
        else None
    )

    def on_train(step: int, row: dict[str, float]) -> None:
        logger.append_metrics({"step": step, **row})
        if train_bar is not None:
            train_bar.set_postfix(
                loss=f"{row['loss']:.4g}",
                exact=f"{row['full_state_exact_accuracy']:.3f}",
            )
            train_bar.update(1)

    train_state, rows = train_softmax_baseline(
        jax.random.PRNGKey(args.seed),
        train_data,
        config=config,
        progress_callback=on_train,
    )
    if train_bar is not None:
        train_bar.close()

    finite_losses = bool(np.isfinite([row["loss"] for row in rows]).all())
    logger.write_json(
        "training_summary.json",
        {
            "initial_loss": rows[0]["loss"],
            "final_loss": rows[-1]["loss"],
            "min_loss": min(row["loss"] for row in rows),
            "finite_losses": finite_losses,
        },
    )
    plot_training(run_dir / "softmax_training.png", rows)

    log_stage(args, "evaluating heldout next-state predictions")
    predictions = predict_softmax_baseline(train_state, validation_data)
    metrics = evaluate_softmax_baseline(train_data, validation_data, predictions)
    logger.write_json("prediction_metrics.json", metrics)
    logger.write_json(
        "sample_predictions.json",
        sample_predictions(
            validation_data,
            predictions,
            count=args.sample_predictions,
        ),
    )

    checkpoint_metadata = {
        "kind": "coingame_softmax_baseline",
        "target": "p(next_joint_state | state, joint_action)",
        "config": dataclasses.asdict(config),
        "action_dim": data.action_dim,
        "num_agents": data.num_agents,
    }
    save_checkpoint(run_dir / "checkpoint", train_state, metadata=checkpoint_metadata)

    reload_state = create_softmax_train_state(
        jax.random.PRNGKey(args.seed + 1),
        config=config,
        num_agents=data.num_agents,
        action_dim=data.action_dim,
    )
    reload_params = load_params(
        run_dir / "checkpoint" / "checkpoint.msgpack",
        reload_state.params,
    )
    reload_state = reload_state.replace(params=reload_params)
    reload_predictions = predict_softmax_baseline(reload_state, validation_data)
    reload_max_abs_diff = float(
        np.max(
            np.abs(
                reload_predictions.next_position_logits
                - predictions.next_position_logits
            )
        )
    )
    reload_passed = reload_max_abs_diff == 0.0
    logger.write_json(
        "reload_evaluation.json",
        {
            "reload_max_abs_logit_diff": reload_max_abs_diff,
            "reload_passed": reload_passed,
        },
    )

    log_stage(
        args,
        f"fitting flow world model ({args.wm_fit_steps} steps) for next-state comparison",
    )
    comparison = run_flow_comparison(
        args, logger, run_dir, train_data, validation_data, predictions
    )
    log_stage(
        args,
        (
            "next-state comparison: deterministic exact "
            f"softmax={comparison.det_exact_softmax:.4f} "
            f"flow={comparison.det_exact_flow:.4f}; "
            "respawn KL(uniform->model) "
            f"softmax={comparison.respawn_kl_softmax:.4f} "
            f"flow={comparison.respawn_kl_flow:.4f} "
            f"empirical={comparison.respawn_kl_empirical:.4f}"
        ),
    )

    passed, criteria = summarize_softmax_outcome(
        metrics,
        finite_losses=finite_losses,
        reload_passed=reload_passed,
        min_deterministic_exact=args.min_deterministic_exact,
        max_respawn_uniform_kl=args.max_respawn_uniform_kl,
    )
    outcome: dict[str, Any] = {
        "milestone": "coingame_softmax_next_state_diagnostic",
        "target": "p(next_joint_state | state, joint_action)",
        "passed": passed,
        "criteria": criteria,
        "prediction_metrics": metrics,
        "artifacts": {
            "checkpoint": str(run_dir / "checkpoint"),
            "training_plot": str(run_dir / "softmax_training.png"),
            "next_state_comparison": str(run_dir / "next_state_comparison.json"),
            "next_state_comparison_plot": str(run_dir / "next_state_comparison.png"),
        },
    }
    logger.write_json("outcome.json", outcome)
    log_stage(
        args,
        (
            "done; deterministic exact="
            f"{metrics['deterministic_full_state_exact_accuracy']}, "
            f"respawn KL={metrics['respawn']['uniform_target_kl']}, "
            f"full exact={metrics['full_state_exact_accuracy']:.4f}"
        ),
    )
    print_json(outcome)


def run_flow_comparison(
    args: argparse.Namespace,
    logger: RunLogger,
    run_dir: Path,
    train_data: Any,
    validation_data: Any,
    softmax_predictions: Any,
):
    """Fit a flow world model on the same transitions and compare next-state predictions.

    The flow is trained on ``train_data`` and sampled on ``validation_data`` so it is
    scored on the identical heldout split as the softmax baseline. ``predict_next``
    yields continuous next-state samples that we decode back to entity cell ids,
    giving softmax and flow a common regime-split (deterministic move accuracy vs.
    stochastic respawn calibration against uniform).
    """
    flow_config = VectorWorldModelConfig(
        state_dim=int(train_data.states.shape[-1]),
        num_agents=train_data.num_agents,
        action_dim=train_data.action_dim,
        hidden_dims=(args.wm_hidden_dim, args.wm_hidden_dim),
        learning_rate=args.wm_learning_rate,
        integration_steps=args.wm_integration_steps,
        flow_type=args.wm_flow_type,
    )
    train_batch = VectorTransitionBatch(
        states=jnp.asarray(train_data.states, dtype=jnp.float32),
        actions=jnp.asarray(train_data.actions, dtype=jnp.int32),
        next_states=jnp.asarray(train_data.next_states, dtype=jnp.float32),
        rewards=jnp.asarray(train_data.rewards, dtype=jnp.float32),
        dones=jnp.asarray(train_data.dones, dtype=jnp.float32),
    )
    flow_key = jax.random.PRNGKey(args.seed + 2)
    flow_state = make_world_model_train_state(flow_key, flow_config)
    flow_state, sample_rng, _, loss_history = fit_world_model_steps(
        flow_state,
        flow_key,
        train_batch,
        flow_config,
        steps=args.wm_fit_steps,
    )
    loss_history = np.asarray(loss_history, dtype=np.float64)

    val_states = jnp.asarray(validation_data.states, dtype=jnp.float32)
    val_actions = jnp.asarray(validation_data.actions, dtype=jnp.int32)
    target_shape = validation_data.next_states.shape
    samples = []
    for _ in range(args.flow_samples):
        sample_rng, draw_key = jax.random.split(sample_rng)
        next_states = predict_next(
            flow_state, draw_key, val_states, val_actions, flow_config
        )
        next_states = np.asarray(next_states, dtype=np.float32).reshape(target_shape)
        samples.append(decode_coin_positions(next_states))
    flow_position_samples = np.stack(samples, axis=0)

    comparison = build_next_state_comparison(
        validation_data, softmax_predictions, flow_position_samples
    )
    plot_next_state_comparison(comparison, run_dir / "next_state_comparison.png")
    logger.write_json(
        "next_state_comparison.json",
        {
            "artifact": str(run_dir / "next_state_comparison.png"),
            "comparison": comparison.to_metrics(),
            "flow_config": dataclasses.asdict(flow_config),
            "flow_fit": {
                "steps": args.wm_fit_steps,
                "initial_loss": float(loss_history[0]),
                "final_loss": float(loss_history[-1]),
                "eval_samples": args.flow_samples,
            },
        },
    )
    return comparison


def plot_training(path: Path, rows: list[dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(1, len(rows) + 1)
    losses = np.asarray([row["loss"] for row in rows], dtype=np.float32)
    exact = np.asarray(
        [row["full_state_exact_accuracy"] for row in rows],
        dtype=np.float32,
    )
    entity = np.asarray([row["entity_accuracy"] for row in rows], dtype=np.float32)

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(11, 4.2))
    loss_ax.plot(steps, losses)
    if np.all(losses > 0.0):
        loss_ax.set_yscale("log")
    loss_ax.set_title("Training loss")
    loss_ax.set_xlabel("step")
    loss_ax.set_ylabel("cross entropy")
    loss_ax.grid(True, alpha=0.25)

    acc_ax.plot(steps, entity, label="entity")
    acc_ax.plot(steps, exact, label="full state")
    acc_ax.set_title("Minibatch accuracy")
    acc_ax.set_xlabel("step")
    acc_ax.set_ylabel("accuracy")
    acc_ax.set_ylim(-0.02, 1.02)
    acc_ax.legend()
    acc_ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def print_json(payload: Any) -> None:
    import json

    from world_marl.logging import to_jsonable

    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
