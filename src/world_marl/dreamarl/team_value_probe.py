"""Frozen local-versus-joint probes for team reward and return prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax


Mode = Literal["local", "joint"]


@dataclass(frozen=True, slots=True)
class ProbeDataset:
    state: np.ndarray
    reward: np.ndarray
    return_: np.ndarray
    episode: np.ndarray
    train_episodes: np.ndarray
    validation_episodes: np.ndarray
    test_episodes: np.ndarray
    manifest: dict[str, object]


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    hidden: int = 256
    learning_rate: float = 3e-4
    steps: int = 2_000
    batch_size: int = 128
    grad_clip: float = 10.0


def _complete_episodes(first: np.ndarray, last: np.ndarray) -> list[tuple[int, int]]:
    first = np.asarray(first, bool)
    last = np.asarray(last, bool)
    episodes = []
    start = None
    for index, (is_first, is_last) in enumerate(zip(first, last, strict=True)):
        if is_first:
            start = index
        if is_last and start is not None:
            episodes.append((start, index + 1))
            start = None
    return episodes


def _future_return(
    reward: np.ndarray,
    terminal: np.ndarray,
    start: int,
    stop: int,
    gamma: float,
) -> np.ndarray:
    """Return after the current state, matching replay-value alignment."""

    result = np.zeros((stop - start,), np.float32)
    running = np.float32(0)
    for index in range(stop - 1, start - 1, -1):
        result[index - start] = running
        continuation = np.float32(0 if terminal[index] else gamma)
        running = np.float32(reward[index]) + continuation * running
    return result


def _split_episodes(count: int, seed: int) -> tuple[np.ndarray, ...]:
    if count < 10:
        raise ValueError(f"at least 10 complete episodes are required, got {count}")
    order = np.random.default_rng(seed).permutation(count)
    validation = max(2, round(0.2 * count))
    test = max(2, round(0.2 * count))
    return (
        np.sort(order[validation + test :]).astype(np.int32),
        np.sort(order[:validation]).astype(np.int32),
        np.sort(order[validation : validation + test]).astype(np.int32),
    )


def load_replay_dataset(
    replay_dir: Path,
    *,
    states_per_episode: int = 64,
    gamma: float = 332 / 333,
    seed: int = 0,
) -> ProbeDataset:
    """Load sampled states and exact returns from complete replay episodes."""

    replay_dir = replay_dir.expanduser().resolve()
    files = sorted(replay_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no replay chunks under {replay_dir}")
    required = {
        "dyn/deter",
        "dyn/stoch",
        "dyn/memory",
        "reward",
        "is_first",
        "is_last",
        "is_terminal",
    }
    chunks = []
    rewards = []
    first = []
    last = []
    terminal = []
    offset = 0
    agents = None
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{path} is missing {sorted(missing)}")
            length = len(archive["reward"])
            current_agents = archive["dyn/deter"].shape[1]
            if agents is None:
                agents = current_agents
            elif agents != current_agents:
                raise ValueError("agent count changes across replay chunks")
            chunks.append((path, offset, offset + length))
            offset += length
            rewards.append(np.asarray(archive["reward"], np.float32))
            first.append(np.asarray(archive["is_first"], bool))
            last.append(np.asarray(archive["is_last"], bool))
            terminal.append(np.asarray(archive["is_terminal"], bool))
    rewards = np.concatenate(rewards)
    first = np.concatenate(first)
    last = np.concatenate(last)
    terminal = np.concatenate(terminal)
    episodes = _complete_episodes(first, last)

    generator = np.random.default_rng(seed)
    selected = []
    reward_targets = []
    return_targets = []
    episode_ids = []
    for episode_id, (start, stop) in enumerate(episodes):
        # The first row is the reset observation and carries no transition reward.
        eligible = np.arange(start + 1, stop, dtype=np.int64)
        if not len(eligible):
            continue
        count = min(states_per_episode, len(eligible))
        chosen = np.sort(generator.choice(eligible, count, replace=False))
        returns = _future_return(rewards, terminal, start, stop, gamma)
        selected.extend(chosen.tolist())
        reward_targets.extend(rewards[chosen].tolist())
        return_targets.extend(returns[chosen - start].tolist())
        episode_ids.extend([episode_id] * count)
    selected = np.asarray(selected, np.int64)
    if not len(selected):
        raise ValueError("replay contains no eligible states")

    state_parts = []
    cursor = 0
    for path, begin, end in chunks:
        stop = np.searchsorted(selected, end, side="left")
        if stop == cursor:
            continue
        local_indices = selected[cursor:stop] - begin
        with np.load(path, allow_pickle=False) as archive:
            deter = np.asarray(archive["dyn/deter"][local_indices], np.float32)
            stoch = np.asarray(archive["dyn/stoch"][local_indices], np.float32)
            memory = np.asarray(archive["dyn/memory"][local_indices], np.float32)
        state_parts.append(
            np.concatenate(
                [
                    deter,
                    stoch.reshape((*stoch.shape[:2], -1)),
                    memory.reshape((*memory.shape[:2], -1)),
                ],
                axis=-1,
            ).astype(np.float16)
        )
        cursor = stop
    if cursor != len(selected):
        raise AssertionError((cursor, len(selected)))
    state = np.concatenate(state_parts)
    episode_ids = np.asarray(episode_ids, np.int32)
    train, validation, test = _split_episodes(len(episodes), seed)
    return ProbeDataset(
        state=state,
        reward=np.asarray(reward_targets, np.float32),
        return_=np.asarray(return_targets, np.float32),
        episode=episode_ids,
        train_episodes=train,
        validation_episodes=validation,
        test_episodes=test,
        manifest={
            "replay_dir": str(replay_dir),
            "chunks": len(files),
            "transitions": int(offset),
            "complete_episodes": len(episodes),
            "sampled_states": len(selected),
            "states_per_episode": states_per_episode,
            "agents": int(agents),
            "state_width": int(state.shape[-1]),
            "gamma": gamma,
            "seed": seed,
            "reward_alignment": "post_transition_state_t_to_reward_t",
            "return_alignment": "state_t_to_discounted_rewards_t_plus_1_onward",
        },
    )


def init_predictor(
    key: jax.Array, input_width: int, hidden: int
) -> dict[str, jax.Array]:
    keys = jax.random.split(key, 3)
    return {
        "adapter_kernel": jax.random.normal(keys[0], (input_width, hidden))
        / jnp.sqrt(input_width),
        "adapter_bias": jnp.zeros((hidden,)),
        "mix_kernel": jax.random.normal(keys[1], (2 * hidden, hidden))
        / jnp.sqrt(2 * hidden),
        "mix_bias": jnp.zeros((hidden,)),
        "output_kernel": jax.random.normal(keys[2], (hidden, 2))
        / jnp.sqrt(hidden),
        "output_bias": jnp.zeros((2,)),
    }


def predictor(
    params: Mapping[str, jax.Array], state: jax.Array, mode: Mode
) -> jax.Array:
    """Predict per-agent targets with identical capacity in both modes."""

    hidden = jax.nn.silu(
        jnp.einsum("...ad,dh->...ah", state, params["adapter_kernel"])
        + params["adapter_bias"]
    )
    if mode == "local":
        pooled = jnp.concatenate([hidden, hidden], -1)
    elif mode == "joint":
        mean = hidden.mean(-2, keepdims=True)
        maximum = hidden.max(-2, keepdims=True)
        pooled = jnp.broadcast_to(
            jnp.concatenate([mean, maximum], -1),
            (*hidden.shape[:-2], hidden.shape[-2], 2 * hidden.shape[-1]),
        )
    else:
        raise ValueError(mode)
    fused = jax.nn.silu(
        jnp.einsum("...ad,dh->...ah", pooled, params["mix_kernel"])
        + params["mix_bias"]
    )
    return (
        jnp.einsum("...ah,ho->...ao", fused, params["output_kernel"])
        + params["output_bias"]
    )


def _mask(dataset: ProbeDataset, episodes: np.ndarray) -> np.ndarray:
    return np.isin(dataset.episode, episodes)


def _normalization(dataset: ProbeDataset) -> tuple[np.ndarray, ...]:
    train = _mask(dataset, dataset.train_episodes)
    flat = np.asarray(dataset.state[train], np.float32).reshape(
        -1, dataset.state.shape[-1]
    )
    state_mean = flat.mean(0)
    state_std = np.maximum(flat.std(0), 1e-4)
    targets = np.stack([dataset.reward, dataset.return_], -1)
    target_mean = targets[train].mean(0)
    target_std = np.maximum(targets[train].std(0), 1e-4)
    return state_mean, state_std, target_mean, target_std


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    names = ("reward", "return")
    result = {}
    repeated = np.broadcast_to(target[:, None], prediction.shape)
    for aggregation, pred, tar in (
        ("per_agent", prediction, repeated),
        ("team_mean", prediction.mean(1), target),
    ):
        values = {}
        for index, name in enumerate(names):
            residual = np.asarray(pred[..., index] - tar[..., index], np.float64)
            variance = np.var(tar[..., index], dtype=np.float64)
            mse = np.mean(np.square(residual))
            values[name] = {
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(mse)),
                "r2": float(1 - mse / max(variance, 1e-12)),
            }
        result[aggregation] = values
    return result


def train_probe(
    dataset: ProbeDataset,
    mode: Mode,
    config: ProbeConfig,
    *,
    seed: int,
) -> dict[str, object]:
    state_mean, state_std, target_mean, target_std = _normalization(dataset)
    targets = np.stack([dataset.reward, dataset.return_], -1)
    train_indices = np.flatnonzero(_mask(dataset, dataset.train_episodes))
    validation_indices = np.flatnonzero(
        _mask(dataset, dataset.validation_episodes)
    )
    test_indices = np.flatnonzero(_mask(dataset, dataset.test_episodes))
    params = init_predictor(
        jax.random.key(seed), dataset.state.shape[-1], config.hidden
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.learning_rate),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, state, target):
        def objective(current):
            prediction = predictor(current, state, mode)
            repeated = jnp.broadcast_to(target[:, None], prediction.shape)
            return jnp.square(prediction - repeated).mean()

        loss, gradients = jax.value_and_grad(objective)(params)
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    generator = np.random.default_rng(seed)
    history = []
    for step in range(config.steps):
        selected = generator.choice(
            train_indices,
            min(config.batch_size, len(train_indices)),
            replace=len(train_indices) < config.batch_size,
        )
        state = (
            np.asarray(dataset.state[selected], np.float32) - state_mean
        ) / state_std
        target = (targets[selected] - target_mean) / target_std
        params, opt_state, loss = update(
            params, opt_state, jnp.asarray(state), jnp.asarray(target)
        )
        if step == 0 or step == config.steps - 1 or (step + 1) % 250 == 0:
            history.append({"step": step + 1, "loss": float(loss)})

    def evaluate(indices: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        outputs = []
        for start in range(0, len(indices), config.batch_size):
            selected = indices[start : start + config.batch_size]
            state = (
                np.asarray(dataset.state[selected], np.float32) - state_mean
            ) / state_std
            normalized = predictor(params, jnp.asarray(state), mode)
            prediction = np.asarray(normalized) * target_std + target_mean
            outputs.append(prediction)
        prediction = np.concatenate(outputs)
        return prediction, _metrics(prediction, targets[indices])

    validation_prediction, validation_metrics = evaluate(validation_indices)
    test_prediction, test_metrics = evaluate(test_indices)
    del validation_prediction
    parameter_count = int(
        sum(np.prod(value.shape) for value in jax.tree.leaves(params))
    )
    return {
        "mode": mode,
        "seed": seed,
        "parameter_count": parameter_count,
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_indices": test_indices,
        "test_prediction": test_prediction,
    }


def paired_episode_bootstrap(
    dataset: ProbeDataset,
    local: Mapping[str, object],
    joint: Mapping[str, object],
    *,
    seed: int,
    samples: int = 2_000,
) -> dict[str, object]:
    local_indices = np.asarray(local["test_indices"])
    joint_indices = np.asarray(joint["test_indices"])
    if not np.array_equal(local_indices, joint_indices):
        raise ValueError("paired probes must use identical test rows")
    target = np.stack([dataset.reward, dataset.return_], -1)[local_indices]
    local_prediction = np.asarray(local["test_prediction"]).mean(1)
    joint_prediction = np.asarray(joint["test_prediction"]).mean(1)
    episode = dataset.episode[local_indices]
    generator = np.random.default_rng(seed)
    result = {}
    for target_index, name in enumerate(("reward", "return")):
        local_error = np.square(local_prediction[:, target_index] - target[:, target_index])
        joint_error = np.square(joint_prediction[:, target_index] - target[:, target_index])
        unique = np.unique(episode)
        deltas = np.asarray(
            [
                np.mean(local_error[episode == item] - joint_error[episode == item])
                for item in unique
            ]
        )
        boot = deltas[
            generator.integers(0, len(deltas), (samples, len(deltas)))
        ].mean(1)
        result[name] = {
            "mse_improvement_local_minus_joint": float(deltas.mean()),
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
            "test_episodes": int(len(unique)),
        }
    return result
