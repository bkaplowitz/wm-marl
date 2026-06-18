"""Verify a fitted vector-state world model against held-out env transitions."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.algs.ippo import IPPOConfig, create_train_state as create_ippo_state
from world_marl.algs.mappo import MAPPOConfig, create_train_state as create_mappo_state
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.logging import timestamp, to_jsonable
from world_marl.training import central_observation_shape
from world_marl.world_model import (
    VectorTransitionBatch,
    VectorWorldModelConfig,
    _pack_discrete_tokens,
    create_world_model_state,
    predict_next,
    train_world_model_step,
)
from world_marl.world_model_training import (
    collect_policy_transition_batch,
    collect_random_transition_batch,
    concatenate_transition_batches,
)

TrainingAdapter = MeltingPotVectorAdapter | JaxMARLCoinGameVectorAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ippo", "mappo"), default="ippo")
    parser.add_argument("--substrate", default="coins")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--observation-size", type=int, default=None)
    parser.add_argument("--append-agent-id", action="store_true")
    parser.add_argument("--include-observation-scalars", action="store_true")
    parser.add_argument("--train-random-rollouts", type=int, default=64)
    parser.add_argument("--train-initial-rollouts", type=int, default=64)
    parser.add_argument("--heldout-random-rollouts", type=int, default=16)
    parser.add_argument("--heldout-initial-rollouts", type=int, default=16)
    parser.add_argument("--fit-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--integration-steps", type=int, default=8)
    parser.add_argument(
        "--flow-type",
        choices=("gaussian", "linear", "discrete"),
        default="gaussian",
    )
    parser.add_argument(
        "--num-categories",
        type=int,
        default=9,
        help=(
            "Per-factor category count (coins = 9). Selects the discrete model "
            "when --flow-type discrete, and is also the cell cardinality used to "
            "decode per-factor categorical accuracy for every flow type."
        ),
    )
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args()
    for name in (
        "train_random_rollouts",
        "train_initial_rollouts",
        "heldout_random_rollouts",
        "heldout_initial_rollouts",
        "fit_steps",
        "hidden_dim",
        "integration_steps",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    run_dir = Path(args.out_dir) / f"verify_fitted_env_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.substrate == "coins":
        # Mirror train_e2e: "coins" is the JaxMARL CoinGame vector env (state_dim
        # 36, strided (3,3,4) one-hots), the only substrate where the discrete
        # flow's per-factor categorical structure applies.
        adapter = JaxMARLCoinGameVectorAdapter(
            num_envs=args.num_envs,
            max_cycles=args.max_cycles,
            seed=args.seed,
        )
    else:
        adapter = MeltingPotVectorAdapter(
            substrate=args.substrate,
            num_envs=args.num_envs,
            max_cycles=args.max_cycles,
            observation_size=args.observation_size,
            include_observation_scalars=args.include_observation_scalars,
            append_agent_id=args.append_agent_id,
        )
    try:
        rng = jax.random.PRNGKey(args.seed)
        rng, policy_key = jax.random.split(rng)
        policy_state = _create_initial_policy_state(args, adapter, policy_key)

        observations = adapter.reset()
        train_batch, observations, rng = _collect_combined_batch(
            args,
            adapter,
            policy_state,
            observations,
            rng,
            random_seed=args.seed + 1,
            random_rollouts=args.train_random_rollouts,
            initial_rollouts=args.train_initial_rollouts,
        )

        observations = adapter.reset()
        heldout_batch, _, rng = _collect_combined_batch(
            args,
            adapter,
            policy_state,
            observations,
            rng,
            random_seed=args.seed + 2,
            random_rollouts=args.heldout_random_rollouts,
            initial_rollouts=args.heldout_initial_rollouts,
        )

        config = VectorWorldModelConfig(
            state_dim=int(train_batch.states.shape[-1]),
            num_agents=adapter.num_agents,
            action_dim=adapter.action_dim,
            hidden_dims=(args.hidden_dim, args.hidden_dim),
            learning_rate=args.learning_rate,
            integration_steps=args.integration_steps,
            flow_type=args.flow_type,
            num_categories=(args.num_categories if args.flow_type == "discrete" else 0),
        )
        rng, model_key = jax.random.split(rng)
        model_state = create_world_model_state(model_key, config)
        model_state, rng, losses = _fit_with_loss_curve(
            model_state,
            rng,
            train_batch,
            config,
            steps=args.fit_steps,
        )

        rng, predict_key = jax.random.split(rng)
        predicted_next_states = predict_next(
            model_state,
            predict_key,
            heldout_batch.states,
            heldout_batch.actions,
            config,
        )
        summary = _summarize_fit(
            args=args,
            config=config,
            train_batch=train_batch,
            heldout_batch=heldout_batch,
            losses=losses,
            predicted_next_states=predicted_next_states,
        )

        _write_loss_csv(run_dir / "loss_curve.csv", losses)
        _plot_loss_curve(run_dir / "loss_curve.png", losses)
        _plot_histogram_overlay(
            run_dir / "next_state_delta_norm_distribution.png",
            _delta_norms(heldout_batch.next_states, heldout_batch.states),
            _delta_norms(predicted_next_states, heldout_batch.states),
            title="Held-out next-state delta norm distribution",
            xlabel="||next_state - state||",
        )
        summary["artifacts"] = {
            "loss_csv": str(run_dir / "loss_curve.csv"),
            "loss_png": str(run_dir / "loss_curve.png"),
            "next_state_delta_norm_distribution_png": str(
                run_dir / "next_state_delta_norm_distribution.png"
            ),
            "summary_json": str(run_dir / "summary.json"),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(to_jsonable(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    finally:
        adapter.close()


def _create_initial_policy_state(
    args: argparse.Namespace,
    adapter: TrainingAdapter,
    key: jax.Array,
):
    observation_shape = (int(np.prod(adapter.observation_shape)),)
    if args.algorithm == "mappo":
        return create_mappo_state(
            key,
            observation_shape,
            central_observation_shape(
                observation_shape,
                adapter.num_agents,
                observation_mode="vector",
            ),
            adapter.action_dim,
            MAPPOConfig(network_arch="mlp", num_minibatches=1),
        )
    return create_ippo_state(
        key,
        observation_shape,
        adapter.action_dim,
        IPPOConfig(network_arch="mlp", num_minibatches=1),
    )


def _collect_combined_batch(
    args: argparse.Namespace,
    adapter: TrainingAdapter,
    policy_state,
    observations: np.ndarray,
    rng: jax.Array,
    *,
    random_seed: int,
    random_rollouts: int,
    initial_rollouts: int,
) -> tuple[VectorTransitionBatch, np.ndarray, jax.Array]:
    random_batch, observations, _ = collect_random_transition_batch(
        adapter,
        observations,
        np.random.default_rng(random_seed),
        rollout_steps=random_rollouts,
    )
    rng, policy_key = jax.random.split(rng)
    policy_batch, observations, rng, _ = collect_policy_transition_batch(
        adapter,
        policy_state,
        observations,
        policy_key,
        rollout_steps=initial_rollouts,
        algorithm=args.algorithm,
    )
    return (
        concatenate_transition_batches([random_batch, policy_batch]),
        observations,
        rng,
    )


def _fit_with_loss_curve(
    model_state,
    rng: jax.Array,
    batch: VectorTransitionBatch,
    config: VectorWorldModelConfig,
    *,
    steps: int,
) -> tuple[Any, jax.Array, list[float]]:
    losses = []
    for _ in range(steps):
        rng, fit_key = jax.random.split(rng)
        model_state, loss = train_world_model_step(
            model_state,
            fit_key,
            batch,
            config,
        )
        loss = jax.block_until_ready(loss)
        losses.append(float(loss))
    return model_state, rng, losses


def _summarize_fit(
    *,
    args: argparse.Namespace,
    config: VectorWorldModelConfig,
    train_batch: VectorTransitionBatch,
    heldout_batch: VectorTransitionBatch,
    losses: list[float],
    predicted_next_states: jnp.ndarray,
) -> dict[str, Any]:
    true_next = np.asarray(heldout_batch.next_states)
    pred_next = np.asarray(predicted_next_states)
    true_rewards = np.asarray(heldout_batch.rewards)
    true_dones = np.asarray(heldout_batch.dones)
    true_delta_norms = _delta_norms(heldout_batch.next_states, heldout_batch.states)
    pred_delta_norms = _delta_norms(predicted_next_states, heldout_batch.states)

    return {
        "args": vars(args),
        "config": dataclasses.asdict(config),
        "train_transition_count": int(train_batch.states.shape[0]),
        "heldout_transition_count": int(heldout_batch.states.shape[0]),
        "loss": {
            "first": losses[0],
            "last": losses[-1],
            "min": float(np.min(losses)),
            "max": float(np.max(losses)),
            "mean_first_10": float(np.mean(losses[: min(10, len(losses))])),
            "mean_last_10": float(np.mean(losses[-min(10, len(losses)) :])),
        },
        "next_state": {
            "mse": float(np.mean(np.square(pred_next - true_next))),
            "mae": float(np.mean(np.abs(pred_next - true_next))),
            "categorical_accuracy": _categorical_accuracy(
                predicted_next_states,
                heldout_batch.next_states,
                config,
                args.num_categories,
            ),
            "true_delta_norm": _numeric_summary(true_delta_norms),
            "predicted_delta_norm": _numeric_summary(pred_delta_norms),
        },
        "reward": {
            "true": _numeric_summary(true_rewards.reshape(-1)),
        },
        "done": {
            "true": _numeric_summary(true_dones.reshape(-1)),
        },
    }


def _categorical_accuracy(
    predicted_next_states: jnp.ndarray,
    true_next_states: jnp.ndarray,
    config: VectorWorldModelConfig,
    num_categories: int,
) -> float | None:
    """Per-factor cell accuracy via the strided coin decode, flow-type agnostic.

    Decode both the prediction and the truth to per-channel cell tokens with the
    same strided argmax (``_pack_discrete_tokens`` at cardinality
    ``num_categories``) and report the fraction that match. Because the decode is
    independent of how the prediction was produced, discrete and the continuous
    baselines (whose soft outputs are argmax-ed to the most likely cell) are
    measured on identical factors. Returns ``None`` when the state is not a clean
    multiple of ``num_categories`` (e.g. observation scalars appended), where the
    coin layout no longer holds.
    """
    transition_dim = config.num_agents * config.state_dim
    if num_categories <= 0 or transition_dim % num_categories != 0:
        return None
    decode_config = dataclasses.replace(config, num_categories=num_categories)
    pred_tokens = _pack_discrete_tokens(
        jnp.asarray(predicted_next_states), decode_config
    )
    true_tokens = _pack_discrete_tokens(jnp.asarray(true_next_states), decode_config)
    return float(np.mean(np.asarray(pred_tokens) == np.asarray(true_tokens)))


def _delta_norms(next_states: jnp.ndarray, states: jnp.ndarray) -> np.ndarray:
    deltas = np.asarray(next_states) - np.asarray(states)
    return np.linalg.norm(deltas.reshape((deltas.shape[0], -1)), axis=-1)


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _write_loss_csv(path: Path, losses: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "loss"))
        writer.writeheader()
        for step, loss in enumerate(losses, start=1):
            writer.writerow({"step": step, "loss": loss})


def _plot_loss_curve(path: Path, losses: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(losses) + 1), np.asarray(losses))
    ax.set_xlabel("fit step")
    ax.set_ylabel("flow matching loss")
    ax.set_title("World model fit loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_histogram_overlay(
    path: Path,
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    *,
    title: str,
    xlabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.asarray(true_values), bins=40, alpha=0.5, density=True, label="true")
    ax.hist(
        np.asarray(predicted_values),
        bins=40,
        alpha=0.5,
        density=True,
        label="model",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
