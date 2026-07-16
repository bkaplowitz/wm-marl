"""Jafar stage losses and functional training helpers.

Adapted from ``FLAIROx/jafar`` at commit
``5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665``, paths
``train_tokenizer.py``, ``train_lam.py``, and ``train_dynamics.py``.
Integration changes: pure typed loss/reset helpers prepare the source
equations for immutable state and ``jax.lax.scan`` training loops.
"""

from collections.abc import Mapping

from flax.core import FrozenDict, freeze, unfreeze
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from world_marl.jafar.model import JafarWorldModel
from world_marl.jafar.lam import LatentActionModel
from world_marl.jafar.tokenizer import TokenizerVQVAE


def vqvae_loss(
    targets: jax.Array,
    outputs: Mapping[str, jax.Array],
    beta: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    mse = jnp.square(targets - outputs["recon"]).mean()
    q_loss = jnp.square(jax.lax.stop_gradient(outputs["emb"]) - outputs["z"]).mean()
    commitment_loss = jnp.square(
        outputs["emb"] - jax.lax.stop_gradient(outputs["z"])
    ).mean()
    total = mse + q_loss + beta * commitment_loss
    return total, {
        "loss": total,
        "mse": mse,
        "q_loss": q_loss,
        "commitment_loss": commitment_loss,
    }


def masked_token_metrics(
    logits: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    cross_entropy = optax.softmax_cross_entropy_with_integer_labels(
        logits,
        labels,
    )
    denominator = mask.sum()
    loss = (mask * cross_entropy).sum() / denominator
    accuracy = (mask * (logits.argmax(-1) == labels)).sum() / denominator
    return loss, accuracy


def reset_inactive_codes(
    rng: jax.Array,
    codebook: jax.Array,
    index_counts: jax.Array,
    action_last_active: jax.Array,
    threshold: int,
) -> tuple[jax.Array, jax.Array]:
    active_codes = index_counts != 0
    action_last_active = jnp.where(
        active_codes,
        0,
        action_last_active + 1,
    )
    probabilities = active_codes / active_codes.sum()
    reset_indices = jax.random.choice(
        rng,
        len(codebook),
        shape=(len(codebook),),
        p=probabilities,
    )
    do_reset = action_last_active >= threshold
    new_codebook = jnp.where(
        do_reset[:, None],
        codebook[reset_indices],
        codebook,
    )
    new_last_active = jnp.where(do_reset, 0, action_last_active)
    return new_codebook, new_last_active


def _adamw(learning_rate: optax.ScalarOrSchedule) -> optax.GradientTransformation:
    return optax.adamw(
        learning_rate=learning_rate,
        b1=0.9,
        b2=0.9,
        weight_decay=1e-4,
    )


def create_tokenizer_train_state(
    rng: jax.Array,
    tokenizer: TokenizerVQVAE,
    example_videos: jax.Array,
    learning_rate: optax.ScalarOrSchedule,
) -> TrainState:
    variables = tokenizer.init(
        rng,
        {"videos": example_videos},
        training=False,
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
    beta: float,
) -> tuple[TrainState, dict[str, jax.Array]]:
    def loss_fn(
        params: Mapping[str, object],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        outputs = state.apply_fn(
            {"params": params},
            {"videos": videos},
            training=True,
            rngs={"dropout": rng},
        )
        return vqvae_loss(videos, outputs, beta)

    (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=gradients), metrics


def scan_tokenizer_updates(
    state: TrainState,
    batches: jax.Array,
    rngs: jax.Array,
    beta: float,
) -> tuple[TrainState, dict[str, jax.Array]]:
    def update(
        current_state: TrainState,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        videos, rng = inputs
        return tokenizer_train_step(current_state, videos, rng, beta)

    return jax.lax.scan(update, state, (batches, rngs))


def create_lam_train_state(
    rng: jax.Array,
    lam: LatentActionModel,
    example_videos: jax.Array,
    learning_rate: optax.ScalarOrSchedule,
) -> TrainState:
    variables = lam.init(
        rng,
        {"videos": example_videos},
        training=False,
    )
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
            rngs={"dropout": rng},
        )
        loss, metrics = vqvae_loss(videos[:, 1:], outputs, beta)
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
    model: JafarWorldModel,
    example_batch: dict[str, jax.Array],
    learning_rate: optax.ScalarOrSchedule,
) -> TrainState:
    variables = model.init(rng, example_batch, training=True)
    params = variables["params"]
    labels = {
        component: jax.tree.map(
            lambda _: "train" if component == "dynamics" else "frozen",
            component_params,
        )
        for component, component_params in params.items()
    }
    optimizer = optax.multi_transform(
        {
            "train": _adamw(learning_rate),
            "frozen": optax.set_to_zero(),
        },
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
        outputs = state.apply_fn({"params": params}, batch, training=True)
        mask = outputs["mask"]
        if mask is None:
            raise ValueError("training dynamics requires a token mask")
        loss, accuracy = masked_token_metrics(
            outputs["token_logits"],
            outputs["video_tokens"],
            mask,
        )
        probabilities = jax.nn.softmax(outputs["token_logits"])
        metrics = {
            "loss": loss,
            "cross_entropy_loss": loss,
            "masked_token_accuracy": accuracy,
            "select_logit": outputs["token_logits"].max(-1).mean(),
            "select_p": probabilities.max(-1).mean(),
            "entropy": jax.scipy.special.entr(probabilities).sum(-1).mean(),
        }
        return loss, metrics

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
