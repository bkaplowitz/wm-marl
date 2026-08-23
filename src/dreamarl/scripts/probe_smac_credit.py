"""Frozen local/joint outcome probes and an all-action combat critic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax


REGRESSION_TARGETS = (
    "damage_h5",
    "damage_h15",
    "corrected_h5",
    "corrected_h15",
    "episode_enemy_deaths",
    "episode_ally_survivors",
)
BINARY_TARGETS = (
    "enemy_death_h5",
    "enemy_death_h15",
    "ally_death_h15",
    "episode_win",
    "episode_timeout",
)


def _load(paths):
    arrays = []
    episode_offset = 0
    for source, path in enumerate(paths):
        with np.load(path) as data:
            item = {key: np.asarray(data[key]) for key in data.files}
        episodes = int(item.pop("num_episodes"))
        item.pop("checkpoint")
        item.pop("policy_mode")
        item["episode_id"] = item["episode_id"].astype(np.int64) + episode_offset
        item["source"] = np.full(item["episode_id"].shape, source, np.int16)
        episode_offset += episodes
        arrays.append(item)
    keys = arrays[0].keys()
    return {key: np.concatenate([item[key] for item in arrays]) for key in keys}


def _episode_split(data, seed):
    rng = np.random.default_rng(seed)
    split = {}
    for source in np.unique(data["source"]):
        episodes = np.unique(data["episode_id"][data["source"] == source])
        rng.shuffle(episodes)
        first = max(1, int(0.7 * len(episodes)))
        second = max(first + 1, int(0.85 * len(episodes)))
        split.setdefault("train", []).extend(episodes[:first])
        split.setdefault("validation", []).extend(episodes[first:second])
        split.setdefault("test", []).extend(episodes[second:])
    return {
        key: np.isin(data["episode_id"], np.asarray(episodes))
        for key, episodes in split.items()
    }


def _latent_table(seed, variables, categories, width=4):
    rng = np.random.default_rng(seed)
    table = rng.normal(size=(variables, categories, width)).astype(np.float32)
    table /= np.maximum(np.linalg.norm(table, axis=-1, keepdims=True), 1e-6)
    return table


def _state_features(data, table):
    indices = data["latent_index"].astype(np.int64)
    variables = np.arange(indices.shape[-1])[None, None, :]
    embedded = table[variables, indices].reshape((*indices.shape[:2], -1))
    # The pinned 3s_vs_4z horizon is 200. Never normalize by the realized
    # episode length because that would leak the future termination time.
    timestep = (data["timestep"] / 200.0).astype(np.float32)
    timestep = np.broadcast_to(timestep[:, None, None], (*indices.shape[:2], 1))
    return np.concatenate(
        [embedded, data["observation"].astype(np.float32), timestep], axis=-1
    )


def _onehot(actions, count):
    return np.eye(count, dtype=np.float32)[actions.astype(np.int64)]


def _focal_views(data, states):
    count, agents, feature_dim = states.shape
    actions = _onehot(data["action"], data["action_mask"].shape[-1])
    labels = np.stack(
        [data[key].astype(np.float32) for key in (*REGRESSION_TARGETS, *BINARY_TARGETS)],
        axis=-1,
    )
    outputs = {key: [] for key in ("local", "joint", "critic", "action", "episode")}
    outputs["labels"] = []
    outputs["mask"] = []
    for focal in range(agents):
        order = [focal, *[index for index in range(agents) if index != focal]]
        ordered_states = states[:, order]
        ordered_actions = actions[:, order]
        local = np.concatenate([ordered_states[:, 0], ordered_actions[:, 0]], -1)
        joint = np.concatenate(
            [ordered_states.reshape(count, -1), ordered_actions.reshape(count, -1)],
            -1,
        )
        critic_actions = ordered_actions.copy()
        critic_actions[:, 0] = 0.0
        critic = np.concatenate(
            [ordered_states.reshape(count, -1), critic_actions.reshape(count, -1)],
            -1,
        )
        outputs["local"].append(local)
        outputs["joint"].append(joint)
        outputs["critic"].append(critic)
        outputs["action"].append(data["action"][:, focal])
        outputs["episode"].append(data["episode_id"])
        outputs["labels"].append(labels)
        outputs["mask"].append(data["action_mask"][:, focal])
    return {key: np.concatenate(value, axis=0) for key, value in outputs.items()}


def _normalize(train, *arrays):
    mean = train.mean(0, keepdims=True)
    std = train.std(0, keepdims=True)
    std = np.where(std < 1e-4, 1.0, std)
    return mean, std, [((array - mean) / std).astype(np.float32) for array in arrays]


def _init_mlp(key, input_dim, output_dim, width=256):
    dims = (input_dim, width, width, output_dim)
    keys = jax.random.split(key, len(dims) - 1)
    params = []
    for inner, outer, layer_key in zip(dims[:-1], dims[1:], keys):
        limit = np.sqrt(6.0 / (inner + outer))
        params.append(
            {
                "kernel": jax.random.uniform(
                    layer_key, (inner, outer), minval=-limit, maxval=limit
                ),
                "bias": jnp.zeros((outer,), jnp.float32),
            }
        )
    return params


def _mlp(params, inputs):
    value = inputs
    for layer in params[:-1]:
        value = jax.nn.silu(value @ layer["kernel"] + layer["bias"])
    layer = params[-1]
    return value @ layer["kernel"] + layer["bias"]


def _fit_outcome_model(
    train_x,
    train_y,
    validation_x,
    validation_y,
    *,
    seed,
    epochs,
    batch_size,
):
    regression = len(REGRESSION_TARGETS)
    target_mean = train_y[:, :regression].mean(0, keepdims=True)
    target_std = train_y[:, :regression].std(0, keepdims=True)
    target_std = np.where(target_std < 1e-4, 1.0, target_std)
    encoded_train = train_y.copy()
    encoded_validation = validation_y.copy()
    encoded_train[:, :regression] = (
        encoded_train[:, :regression] - target_mean
    ) / target_std
    encoded_validation[:, :regression] = (
        encoded_validation[:, :regression] - target_mean
    ) / target_std
    positives = encoded_train[:, regression:].sum(0)
    negatives = len(encoded_train) - positives
    positive_weight = np.clip(negatives / np.maximum(positives, 1.0), 1.0, 20.0)

    params = _init_mlp(
        jax.random.PRNGKey(seed), train_x.shape[-1], train_y.shape[-1]
    )
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)
    positive_weight_jax = jnp.asarray(positive_weight, jnp.float32)

    def lossfn(parameters, inputs, targets):
        prediction = _mlp(parameters, inputs)
        regression_loss = jnp.square(
            prediction[:, :regression] - targets[:, :regression]
        ).mean()
        logits = prediction[:, regression:]
        binary = targets[:, regression:]
        weight = 1.0 + binary * (positive_weight_jax - 1.0)
        binary_loss = (
            optax.sigmoid_binary_cross_entropy(logits, binary) * weight
        ).mean()
        return regression_loss + binary_loss

    @jax.jit
    def update(parameters, state, inputs, targets):
        loss, grads = jax.value_and_grad(lossfn)(parameters, inputs, targets)
        updates, state = optimizer.update(grads, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss

    rng = np.random.default_rng(seed)
    best = None
    best_loss = np.inf
    patience = 0
    for _ in range(epochs):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            params, opt_state, _ = update(
                params,
                opt_state,
                jnp.asarray(train_x[indices]),
                jnp.asarray(encoded_train[indices]),
            )
        validation_loss = float(
            lossfn(
                params,
                jnp.asarray(validation_x),
                jnp.asarray(encoded_validation),
            )
        )
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best = jax.tree.map(lambda value: np.asarray(value), params)
            patience = 0
        else:
            patience += 1
        if patience >= 5:
            break
    return best, target_mean, target_std, best_loss


def _predict_outcomes(params, inputs, mean, std):
    raw = np.asarray(_mlp(params, jnp.asarray(inputs))).copy()
    regression = len(REGRESSION_TARGETS)
    raw[:, :regression] = raw[:, :regression] * std + mean
    raw[:, regression:] = 1.0 / (1.0 + np.exp(-raw[:, regression:]))
    return raw


def _auc(target, score):
    target = np.asarray(target, bool)
    positive = int(target.sum())
    negative = len(target) - positive
    if not positive or not negative:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), np.float64)
    sorted_scores = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return float((ranks[target].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _metrics(target, prediction):
    result = {}
    regression = len(REGRESSION_TARGETS)
    for index, name in enumerate(REGRESSION_TARGETS):
        error = prediction[:, index] - target[:, index]
        variance = np.var(target[:, index])
        result[name] = {
            "mae": float(np.mean(np.abs(error))),
            "r2": float(1.0 - np.mean(np.square(error)) / max(variance, 1e-8)),
        }
    for offset, name in enumerate(BINARY_TARGETS):
        index = regression + offset
        result[name] = {
            "auc": _auc(target[:, index], prediction[:, index]),
            "brier": float(np.mean(np.square(prediction[:, index] - target[:, index]))),
            "positive_rate": float(np.mean(target[:, index])),
        }
    return result


def _fit_action_critic(
    train_x,
    train_action,
    train_target,
    validation,
    *,
    action_count,
    seed,
    epochs,
    batch_size,
):
    target_mean = float(train_target.mean())
    target_std = float(max(train_target.std(), 1e-4))
    encoded = ((train_target - target_mean) / target_std).astype(np.float32)
    validation_x, validation_action, validation_target = validation
    validation_encoded = ((validation_target - target_mean) / target_std).astype(np.float32)
    params = _init_mlp(jax.random.PRNGKey(seed), train_x.shape[-1], action_count)
    optimizer = optax.adam(3e-4)
    state = optimizer.init(params)

    def selected(parameters, inputs, action):
        values = _mlp(parameters, inputs)
        return jnp.take_along_axis(values, action[:, None], axis=-1)[:, 0]

    def lossfn(parameters, inputs, action, target):
        return jnp.square(selected(parameters, inputs, action) - target).mean()

    @jax.jit
    def update(parameters, opt_state, inputs, action, target):
        loss, grads = jax.value_and_grad(lossfn)(parameters, inputs, action, target)
        updates, opt_state = optimizer.update(grads, opt_state, parameters)
        return optax.apply_updates(parameters, updates), opt_state, loss

    rng = np.random.default_rng(seed)
    best = None
    best_loss = np.inf
    patience = 0
    for _ in range(epochs):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            params, state, _ = update(
                params,
                state,
                jnp.asarray(train_x[indices]),
                jnp.asarray(train_action[indices]),
                jnp.asarray(encoded[indices]),
            )
        value = float(
            lossfn(
                params,
                jnp.asarray(validation_x),
                jnp.asarray(validation_action),
                jnp.asarray(validation_encoded),
            )
        )
        if value < best_loss - 1e-4:
            best_loss = value
            best = jax.tree.map(lambda item: np.asarray(item), params)
            patience = 0
        else:
            patience += 1
        if patience >= 5:
            break
    return best, target_mean, target_std, best_loss


def _save_model(path, models, metadata):
    arrays = {"metadata": np.asarray(json.dumps(metadata, sort_keys=True))}
    for name, params in models.items():
        for index, layer in enumerate(params):
            arrays[f"{name}_{index}_kernel"] = np.asarray(layer["kernel"])
            arrays[f"{name}_{index}_bias"] = np.asarray(layer["bias"])
    np.savez_compressed(path, **arrays)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args(argv)

    data = _load(args.datasets)
    split = _episode_split(data, args.seed)
    table = _latent_table(
        args.seed,
        data["latent_index"].shape[-1],
        max(64, int(data["latent_index"].max()) + 1),
    )
    views = _focal_views(data, _state_features(data, table))
    episode_split = {
        key: np.isin(views["episode"], np.unique(data["episode_id"][mask]))
        for key, mask in split.items()
    }

    models = {}
    result = {"samples": int(len(views["episode"])), "models": {}}
    normalizers = {}
    for model_name in ("local", "joint"):
        train_mask = episode_split["train"]
        validation_mask = episode_split["validation"]
        test_mask = episode_split["test"]
        mean, std, normalized = _normalize(
            views[model_name][train_mask],
            views[model_name][train_mask],
            views[model_name][validation_mask],
            views[model_name][test_mask],
        )
        train_x, validation_x, test_x = normalized
        params, target_mean, target_std, validation_loss = _fit_outcome_model(
            train_x,
            views["labels"][train_mask],
            validation_x,
            views["labels"][validation_mask],
            seed=args.seed + len(models),
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        prediction = _predict_outcomes(params, test_x, target_mean, target_std)
        result["models"][model_name] = {
            "validation_loss": validation_loss,
            "test": _metrics(views["labels"][test_mask], prediction),
        }
        models[model_name] = params
        normalizers[model_name] = {"mean": mean, "std": std}

    test_mask = episode_split["test"]
    joint = views["joint"][test_mask].copy()
    feature_dim = (joint.shape[-1] - data["action_mask"].shape[-1] * 3) // 3
    action_start = 3 * feature_dim
    rng = np.random.default_rng(args.seed + 99)
    for peer in (1, 2):
        block = slice(action_start + peer * 10, action_start + (peer + 1) * 10)
        joint[:, block] = joint[rng.permutation(len(joint)), block]
    mean = normalizers["joint"]["mean"]
    std = normalizers["joint"]["std"]
    shuffled_prediction = _predict_outcomes(
        models["joint"], ((joint - mean) / std).astype(np.float32), target_mean, target_std
    )
    result["models"]["joint_shuffled_peer_actions"] = {
        "test": _metrics(views["labels"][test_mask], shuffled_prediction)
    }

    critic_mean, critic_std, critic_normalized = _normalize(
        views["critic"][episode_split["train"]],
        views["critic"][episode_split["train"]],
        views["critic"][episode_split["validation"]],
        views["critic"][episode_split["test"]],
    )
    critic_train, critic_validation, critic_test = critic_normalized
    target_index = REGRESSION_TARGETS.index("corrected_h15")
    critic, qmean, qstd, validation_loss = _fit_action_critic(
        critic_train,
        views["action"][episode_split["train"]],
        views["labels"][episode_split["train"], target_index],
        (
            critic_validation,
            views["action"][episode_split["validation"]],
            views["labels"][episode_split["validation"], target_index],
        ),
        action_count=10,
        seed=args.seed + 10,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    selected = np.take_along_axis(
        np.asarray(_mlp(critic, jnp.asarray(critic_test))),
        views["action"][episode_split["test"], None],
        axis=-1,
    )[:, 0]
    selected = selected * qstd + qmean
    target = views["labels"][episode_split["test"], target_index]
    result["all_action_critic"] = {
        "validation_loss": validation_loss,
        "test_mae": float(np.mean(np.abs(selected - target))),
        "test_r2": float(1.0 - np.mean(np.square(selected - target)) / max(np.var(target), 1e-8)),
        "action_counts": np.bincount(
            views["action"][episode_split["train"]], minlength=10
        ).tolist(),
    }
    models["critic"] = critic
    metadata = {
        "agent_feature_dim": int(feature_dim),
        "action_count": 10,
        "latent_table": table.tolist(),
        "critic_input_mean": critic_mean.reshape(-1).tolist(),
        "critic_input_std": critic_std.reshape(-1).tolist(),
        "critic_target_mean": qmean,
        "critic_target_std": qstd,
        "datasets": [str(path) for path in args.datasets],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "probe_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _save_model(args.output_dir / "probe_models.npz", models, metadata)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
