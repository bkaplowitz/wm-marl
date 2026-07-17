"""Jasmine schedules, losses, and scanned stage-training helpers.

Adapted from ``p-doom/jasmine`` at commit
``420859bc99eecf6b07a7e9edf65d5d145935f1e1``, paths
``jasmine/utils/train_utils.py``,
``jasmine/baselines/diffusion/train_tokenizer_mae.py``,
``jasmine/baselines/train_lam.py``, and
``jasmine/baselines/diffusion/train_dynamics_diffusion.py``.
Integration changes: NNX optimizer mutation becomes Linen ``TrainState`` updates, stage
updates run through ``jax.lax.scan``, and optimizer labels make the frozen MAE
and optional LAM stop boundaries explicit.
"""

from collections.abc import Mapping

from flax.core import FrozenDict, freeze, unfreeze
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from world_marl.jasmine.dynamics import ramp_weight
from world_marl.jasmine.lam import LatentActionModel
from world_marl.jasmine.model import JasmineWorldModel
from world_marl.jasmine.tokenizer import TokenizerMAE


def wsd_schedule(
    initial_learning_rate: float,
    peak_learning_rate: float,
    decay_end: float,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
) -> optax.Schedule:
    if warmup_steps + decay_steps > total_steps:
        raise ValueError("warmup and decay periods exceed total steps")
    schedules = (
        optax.linear_schedule(
            init_value=initial_learning_rate,
            end_value=peak_learning_rate,
            transition_steps=warmup_steps,
        ),
        optax.constant_schedule(peak_learning_rate),
        optax.linear_schedule(
            init_value=peak_learning_rate,
            end_value=decay_end,
            transition_steps=decay_steps,
        ),
    )
    return optax.join_schedules(
        schedules,
        boundaries=(warmup_steps, total_steps - decay_steps),
    )


def tokenizer_loss(
    targets: jax.Array,
    outputs: Mapping[str, jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    mse = jnp.square(
        targets.astype(jnp.float32) - outputs["recon"].astype(jnp.float32)
    ).mean()
    return mse, {"loss": mse, "mse": mse}


def vq_loss(
    targets: jax.Array,
    outputs: Mapping[str, jax.Array],
    beta: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    mse = jnp.square(
        targets.astype(jnp.float32) - outputs["recon"].astype(jnp.float32)
    ).mean()
    q_loss = jnp.square(jax.lax.stop_gradient(outputs["emb"]) - outputs["z"]).mean()
    commitment_loss = jnp.square(
        outputs["emb"] - jax.lax.stop_gradient(outputs["z"])
    ).mean()
    loss = mse + q_loss + beta * commitment_loss
    return loss, {
        "loss": loss,
        "mse": mse,
        "q_loss": q_loss,
        "commitment_loss": commitment_loss,
    }


def diffusion_loss(
    outputs: Mapping[str, jax.Array],
    num_actions: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    squared_error = jnp.square(
        outputs["x_pred"].astype(jnp.float32) - outputs["x_gt"].astype(jnp.float32)
    )
    mse_per_frame = squared_error.mean(axis=(2, 3))
    mse = mse_per_frame.mean()
    loss = (mse_per_frame * ramp_weight(outputs["signal_level"])).mean()
    count_fn = jax.vmap(lambda index: (outputs["lam_indices"] == index).sum())
    index_counts = count_fn(jnp.arange(num_actions))
    return loss, {
        "loss": loss,
        "mse": mse,
        "codebook_usage_lam": (index_counts != 0).mean(),
    }


def reset_inactive_codes(
    rng: jax.Array,
    codebook: jax.Array,
    index_counts: jax.Array,
    action_last_active: jax.Array,
    threshold: int,
) -> tuple[jax.Array, jax.Array]:
    active_codes = index_counts != 0
    action_last_active = jnp.where(active_codes, 0, action_last_active + 1)
    probabilities = active_codes / active_codes.sum()
    reset_indices = jax.random.choice(
        rng,
        len(codebook),
        shape=(len(codebook),),
        p=probabilities,
    )
    do_reset = action_last_active >= threshold
    codebook = jnp.where(do_reset[:, None], codebook[reset_indices], codebook)
    action_last_active = jnp.where(do_reset, 0, action_last_active)
    return codebook, action_last_active


def _adamw(
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.adamw(
        learning_rate=learning_rate,
        b1=0.9,
        b2=0.9,
        weight_decay=1e-4,
        mu_dtype=jnp.float32,
    )


def create_tokenizer_train_state(
    rng: jax.Array,
    tokenizer: TokenizerMAE,
    example_videos: jax.Array,
    learning_rate: optax.ScalarOrSchedule,
) -> TrainState:
    init_rng, mask_rng = jax.random.split(rng)
    variables = tokenizer.init(
        init_rng,
        {"videos": example_videos, "rng": mask_rng},
        training=True,
    )
    return TrainState.create(
        apply_fn=tokenizer.apply,
        params=variables["params"],
        tx=_adamw(learning_rate),
    )


def tokenizer_train_step(
    state: TrainState,
    videos: jax.Array,
    rng: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    def loss_fn(
        params: Mapping[str, object],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        outputs = state.apply_fn(
            {"params": params},
            {"videos": videos, "rng": rng},
            training=True,
        )
        return tokenizer_loss(videos, outputs)

    (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=gradients), metrics


def scan_tokenizer_updates(
    state: TrainState,
    batches: jax.Array,
    rngs: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    def update(
        current_state: TrainState,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        videos, rng = inputs
        return tokenizer_train_step(current_state, videos, rng)

    return jax.lax.scan(update, state, (batches, rngs))


def create_lam_train_state(
    rng: jax.Array,
    lam: LatentActionModel,
    example_videos: jax.Array,
    learning_rate: optax.ScalarOrSchedule,
) -> TrainState:
    variables = lam.init(rng, {"videos": example_videos}, training=True)
    return TrainState.create(
        apply_fn=lam.apply,
        params=variables["params"],
        tx=_adamw(learning_rate),
    )


def lam_train_step(
    state: TrainState,
    action_last_active: jax.Array,
    videos: jax.Array,
    rng: jax.Array,
    beta: float,
    reset_threshold: int,
) -> tuple[TrainState, jax.Array, dict[str, jax.Array]]:
    def loss_fn(
        params: Mapping[str, object],
    ) -> tuple[jax.Array, tuple[dict[str, jax.Array], jax.Array]]:
        outputs = state.apply_fn(
            {"params": params},
            {"videos": videos},
            training=True,
        )
        loss, metrics = vq_loss(videos[:, 1:], outputs, beta)
        num_codes = len(params["vq"]["codebook"])
        count_fn = jax.vmap(lambda index: (outputs["indices"] == index).sum())
        index_counts = count_fn(jnp.arange(num_codes))
        metrics["codebook_usage"] = (index_counts != 0).mean()
        return loss, (metrics, index_counts)

    (_, (metrics, index_counts)), gradients = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(state.params)
    state = state.apply_gradients(grads=gradients)
    mutable_params = unfreeze(state.params)
    codebook, action_last_active = reset_inactive_codes(
        rng,
        mutable_params["vq"]["codebook"],
        index_counts,
        action_last_active,
        reset_threshold,
    )
    mutable_params["vq"]["codebook"] = codebook
    params = (
        freeze(mutable_params)
        if isinstance(state.params, FrozenDict)
        else mutable_params
    )
    return state.replace(params=params), action_last_active, metrics


def scan_lam_updates(
    state: TrainState,
    action_last_active: jax.Array,
    batches: jax.Array,
    rngs: jax.Array,
    beta: float,
    reset_threshold: int,
) -> tuple[TrainState, jax.Array, dict[str, jax.Array]]:
    def update(
        carry: tuple[TrainState, jax.Array],
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[TrainState, jax.Array], dict[str, jax.Array]]:
        current_state, last_active = carry
        videos, rng = inputs
        current_state, last_active, metrics = lam_train_step(
            current_state,
            last_active,
            videos,
            rng,
            beta,
            reset_threshold,
        )
        return (current_state, last_active), metrics

    (state, action_last_active), metrics = jax.lax.scan(
        update,
        (state, action_last_active),
        (batches, rngs),
    )
    return state, action_last_active, metrics


def create_dynamics_train_state(
    rng: jax.Array,
    model: JasmineWorldModel,
    example_batch: dict[str, jax.Array],
    learning_rate: optax.ScalarOrSchedule,
) -> TrainState:
    variables = model.init(rng, example_batch)
    params = variables["params"]
    labels = {
        component: jax.tree.map(
            lambda _: (
                "train"
                if component == "dynamics"
                or (component == "lam" and model.lam_co_train)
                else "frozen"
            ),
            component_params,
        )
        for component, component_params in params.items()
    }
    optimizer = optax.multi_transform(
        {"train": _adamw(learning_rate), "frozen": optax.set_to_zero()},
        labels,
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


def dynamics_train_step(
    state: TrainState,
    batch: dict[str, jax.Array],
) -> tuple[TrainState, dict[str, jax.Array]]:
    def loss_fn(
        params: Mapping[str, object],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        outputs = state.apply_fn({"params": params}, batch)
        num_actions = len(params["lam"]["vq"]["codebook"])
        return diffusion_loss(outputs, num_actions)

    (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=gradients), metrics


def scan_dynamics_updates(
    state: TrainState,
    batches: dict[str, jax.Array],
) -> tuple[TrainState, dict[str, jax.Array]]:
    def update(
        current_state: TrainState,
        batch: dict[str, jax.Array],
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        return dynamics_train_step(current_state, batch)

    return jax.lax.scan(update, state, batches)
