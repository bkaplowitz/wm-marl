"""Train and validate a discrete CoinGame next-state world model."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import jax
import numpy as np
from tqdm import tqdm

from world_marl.checkpointing import load_params, save_checkpoint
from world_marl.coingame_dynamics import (
    CoinDynamicsConfig,
    collect_coin_dynamics_dataset,
    create_coin_dynamics_train_state,
    evaluate_coin_dynamics,
    predict_coin_dynamics,
    prepare_coin_dynamics_data,
    sample_predictions,
    split_coin_dynamics_data,
    summarize_coin_dynamics_outcome,
    train_coin_dynamics_model,
)
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.logging import RunLogger, dependency_versions, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate", default="coins")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument(
        "--target-source",
        choices=("random", "checkpoint"),
        default="random",
        help="Policy used to collect transition actions. Random is best for coverage.",
    )
    parser.add_argument("--policy-checkpoint", default=None)
    parser.add_argument("--source-stochastic", action="store_true")
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
        "--min-reward-exact",
        type=float,
        default=0.99,
        help="Pass threshold for exact reward accuracy on non-terminal transitions.",
    )
    parser.add_argument(
        "--max-respawn-uniform-kl",
        type=float,
        default=0.25,
        help="Pass threshold for KL(uniform respawn target || model respawn distribution).",
    )
    parser.add_argument("--sample-predictions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.substrate != "coins":
        parser.error(
            "world-marl-train-coin-dynamics currently targets --substrate coins"
        )
    if args.target_source == "checkpoint" and args.policy_checkpoint is None:
        parser.error("--policy-checkpoint is required with --target-source checkpoint")
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
    if not 0.0 <= args.min_reward_exact <= 1.0:
        parser.error("--min-reward-exact must be in [0, 1]")
    if args.max_respawn_uniform_kl < 0.0:
        parser.error("--max-respawn-uniform-kl must be non-negative")
    if args.stochastic_target_weight <= 0.0:
        parser.error("--stochastic-target-weight must be positive")
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
        print(f"[coin-dynamics] {message}", flush=True)


def main() -> None:
    args = parse_args()
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    config = CoinDynamicsConfig(
        hidden_dims=hidden_dims,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        max_grad_norm=args.max_grad_norm,
        stochastic_target_weight=args.stochastic_target_weight,
    )
    run_dir = Path(args.out_dir) / f"coin_dynamics_{timestamp()}"
    log_stage(args, f"writing artifacts to {run_dir}")
    logger = RunLogger(run_dir)
    logger.write_json(
        "config.json",
        {
            "args": vars(args),
            "model_config": dataclasses.asdict(config),
            "target": "p(next_joint_state, reward | state, joint_action)",
            "purpose": (
                "Validate a discrete categorical CoinGame dynamics model before using "
                "learned dynamics for model-based policy improvement. Stochastic "
                "coin respawns and terminal resets are trained as distributions."
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
        policy_fn, source_metadata = _make_source_policy(args, adapter)
        logger.write_json(
            "source_policy.json",
            {
                "target_source": args.target_source,
                "policy_checkpoint": args.policy_checkpoint,
                "source_metadata": source_metadata,
            },
        )

        log_stage(
            args,
            (
                f"collecting {args.collect_steps} transition steps "
                f"({args.collect_steps * args.num_envs} samples) from {args.target_source}"
            ),
        )
        collect_bar = (
            tqdm(total=args.collect_steps, desc="collect transitions", unit="step")
            if not args.quiet
            else None
        )

        def on_collect(_step: int) -> None:
            if collect_bar is not None:
                collect_bar.update(1)

        dataset = collect_coin_dynamics_dataset(
            adapter,
            np.random.default_rng(args.seed),
            rollout_steps=args.collect_steps,
            policy_fn=policy_fn,
            progress_callback=on_collect,
        )
        if collect_bar is not None:
            collect_bar.close()
    finally:
        adapter.close()

    data = prepare_coin_dynamics_data(dataset)
    train_data, validation_data = split_coin_dynamics_data(
        data,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    logger.write_json(
        "transition_dataset.json",
        {
            "num_transitions": dataset.num_transitions,
            "train_transitions": train_data.num_transitions,
            "validation_transitions": validation_data.num_transitions,
            "num_envs": dataset.num_envs,
            "rollout_steps": dataset.rollout_steps,
            "action_dim": dataset.action_dim,
            "num_agents": dataset.num_agents,
            "mean_reward_by_agent": dataset.rewards.mean(axis=0).astype(float).tolist(),
            "done_fraction": float(np.any(dataset.dones > 0.0, axis=1).mean()),
        },
    )

    log_stage(args, f"training discrete dynamics model for {args.train_steps} steps")
    train_bar = (
        tqdm(total=args.train_steps, desc="train dynamics", unit="step")
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

    train_state, rows = train_coin_dynamics_model(
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
    plot_training(run_dir / "dynamics_training.png", rows)

    log_stage(args, "evaluating heldout next-state predictions")
    predictions = predict_coin_dynamics(train_state, validation_data)
    metrics = evaluate_coin_dynamics(train_data, validation_data, predictions)
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
        "kind": "coingame_discrete_dynamics",
        "target": "p(next_joint_state, reward | state, joint_action)",
        "config": dataclasses.asdict(config),
        "action_dim": data.action_dim,
        "num_agents": data.num_agents,
    }
    save_checkpoint(run_dir / "checkpoint", train_state, metadata=checkpoint_metadata)

    reload_state = create_coin_dynamics_train_state(
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
    reload_predictions = predict_coin_dynamics(reload_state, validation_data)
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

    passed, criteria = summarize_coin_dynamics_outcome(
        metrics,
        finite_losses=finite_losses,
        reload_passed=reload_passed,
        min_deterministic_exact=args.min_deterministic_exact,
        min_reward_exact=args.min_reward_exact,
        max_respawn_uniform_kl=args.max_respawn_uniform_kl,
    )
    outcome: dict[str, Any] = {
        "milestone": "discrete_coingame_next_state_dynamics",
        "target": "p(next_joint_state, reward | state, joint_action)",
        "passed": passed,
        "criteria": criteria,
        "prediction_metrics": metrics,
        "artifacts": {
            "checkpoint": str(run_dir / "checkpoint"),
            "training_plot": str(run_dir / "dynamics_training.png"),
        },
    }
    logger.write_json("outcome.json", outcome)
    log_stage(
        args,
        (
            "done; deterministic exact="
            f"{metrics['deterministic_full_state_exact_accuracy']}, "
            f"reward exact={metrics['reward']['nonterminal_transition_exact_accuracy']}, "
            f"respawn KL={metrics['respawn']['uniform_target_kl']}, "
            f"full exact={metrics['full_state_exact_accuracy']:.4f}"
        ),
    )
    print_json(outcome)


def _make_source_policy(
    args: argparse.Namespace, adapter: JaxMARLCoinGameVectorAdapter
):
    if args.target_source == "random":
        return None, None
    from world_marl.policy_loading import load_checkpoint_policy

    return load_checkpoint_policy(
        args.policy_checkpoint,
        adapter,
        deterministic=not args.source_stochastic,
        seed=args.seed + 10,
    )


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
