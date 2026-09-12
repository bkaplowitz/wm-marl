"""Truncated recurrent supervision of the deployed joint simulator.

Recorded actions determine targets even outside predicted support. Joint-state
history is differentiated through two-step chunks. Local parameters are frozen;
trajectory posterior alignment keeps its input Jacobian. Joint transition inputs
retain the established stop-gradient boundary into local states.
"""

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj

from ..models.heads import binary_vector_loss
from .ctde import sample_two_step_anchors, shared_team_outcomes


def endpoint_validity(first, last, present, alive, horizon):
    """Return team/root-live masks [B,T,H,A], retaining terminal arrivals.

    A source needs two observed transitions to be sampled, but individual later
    endpoints may be invalid. This retains near-terminal H2 examples when H4/H5
    would cross a reset or run off the batch. Individual death is never itself
    a trajectory boundary; disappearing roster slots are.
    """
    first, last = jnp.asarray(first, bool), jnp.asarray(last, bool)
    present, alive = jnp.asarray(present, bool), jnp.asarray(alive, bool)
    if (
        first.ndim != 2
        or last.shape != first.shape
        or present.shape[:2] != first.shape
        or present.ndim != 3
        or alive.shape != present.shape
        or horizon < 2
    ):
        raise ValueError("Self-fed boundaries require [B,T] flags, [B,T,A] masks, H>=2")
    length = first.shape[1]
    if length < 3:
        raise ValueError("Self-fed training requires at least three replay states")
    index = jnp.arange(length)[:, None] + jnp.arange(1, horizon + 1)[None]
    target = jnp.minimum(index, length - 1)
    source = jnp.minimum(index - 1, length - 1)
    valid = present[:, :, None] & jnp.take(present, target, axis=1)
    valid &= (
        ~jnp.take(first, target, axis=1)
        & ~jnp.take(last, source, axis=1)
        & (index < length)[None]
    )[..., None]
    valid = jnp.cumprod(valid.astype(jnp.int32), axis=2).astype(bool)
    return valid, valid & alive[:, :, None]


def gather_at(values, anchors, offset=0):
    """Safe gather; endpoint_validity supplies the independent out-of-bounds mask."""
    return jax.tree.map(
        lambda value: value[
            anchors.batch, jnp.minimum(anchors.time + offset, value.shape[1] - 1)
        ],
        values,
    )


def scatter_sample_mean(value, anchors, valid, destination_valid, *, team_loss=False):
    """Keep the learner mean equal to the sampled valid-head mean.

    Team heads include dead but present slots. The learner's outer loss mask may
    exclude those slots, so place their *summed loss* on eligible rows at the same
    root instead. This changes only the sparse loss placement, never which head
    outputs receive a gradient or the sampled-head mean being optimized.
    """
    recipients = destination_valid[anchors.batch, anchors.time]
    valid = valid & anchors.valid[:, None] & recipients.any(-1, keepdims=True)
    if not team_loss:
        valid &= recipients
    scale = destination_valid.sum() / jnp.maximum(valid.sum(), 1)
    value = jnp.where(valid, value.astype(jnp.float32), 0.0) * scale
    if team_loss:
        value = (
            value.sum(-1, keepdims=True)
            / jnp.maximum(recipients.sum(-1, keepdims=True), 1)
        ) * recipients
    grid = jnp.zeros(destination_valid.shape, jnp.float32)
    return grid.at[anchors.batch, anchors.time].add(value)


def frozen_posterior(dynamics, embedding, deter, *, history_gradient=False):
    """Frozen parameter Jacobian, live input Jacobian, no state creation/writes."""
    prefix = dynamics.path + "/"
    params = {
        key: jax.lax.stop_gradient(value)
        for key, value in nj.context().items()
        if key.startswith(prefix)
    }
    _, logits = nj.pure(dynamics.posterior, nested=True)(
        params,
        nn.cast(embedding),
        nn.cast(deter if history_gradient else jax.lax.stop_gradient(deter)),
        create=False,
        modify=False,
    )
    return logits


def mixed_posterior_kl(predicted, target, unimix):
    """Actual executable categorical KL, summed over stochastic variables."""

    def probability(logits):
        return (1 - unimix) * jax.nn.softmax(
            logits.astype(jnp.float32), axis=-1
        ) + unimix / logits.shape[-1]

    p = jax.lax.stop_gradient(probability(target))
    q = probability(predicted)
    return (p * (jnp.log(jnp.maximum(p, 1e-30)) - jnp.log(jnp.maximum(q, 1e-30)))).sum(
        axis=(-1, -2)
    )


def frozen_local_transition(dynamics, local, action, embedding, active, *, logits=None):
    """Keep the local input Jacobian while stopping all local parameters."""

    def advance(local, action, embedding, active, logits):
        cache, deter = dynamics.advance(local, action, training=False, active=active)
        if logits is not None:
            return dynamics.complete(cache, deter, logit=logits, sample=True)[0]
        return dynamics.complete_from_observation(cache, deter, embedding, sample=True)[
            0
        ]

    if nj.creating():
        return advance(local, action, embedding, active, logits)
    params = {
        key: jax.lax.stop_gradient(value)
        for key, value in nj.context().items()
        if key.startswith(dynamics.path + "/")
    }
    _, output = nj.pure(advance, nested=True)(
        params,
        local,
        action,
        embedding,
        active,
        logits,
        seed=nj.seed(),
        create=False,
        modify=False,
    )
    return output


def truncate_self_fed_state(state, offset, bptt_steps):
    """Start each truncated chunk from a detached recurrent state."""
    return jax.lax.cond(
        (offset - 1) % bptt_steps == 0,
        jax.lax.stop_gradient,
        lambda value: value,
        state,
    )


def self_fed_losses(agent, online, features, entries, ema, obs, prevact):
    """Produce sparse root-aligned auxiliary grids using existing model heads."""
    cfg = agent.config.marl.ctde.self_fed
    bptt_steps = int(cfg.get("bptt_steps", 1))
    trajectory_kl = float(cfg.get("trajectory_kl_scale", 0.0)) > 0
    horizons = tuple(int(value) for value in cfg.horizons)
    maximum = max(horizons)
    group = agent.team.unfold_sequence
    stop = jax.lax.stop_gradient
    present = group(agent._present(obs)).astype(bool)
    alive = group(agent._controllable(obs)).astype(bool)
    first = group(obs["is_first"]).any(-1)
    last = group(obs["is_last"]).any(-1)
    actions = group(prevact[agent.ctde_action_key]).astype(jnp.int32)
    targets = stop(
        {
            "online": group(online),
            "ema": group(ema),
            "reward": group(obs["reward"]),
            "terminal": group(obs["is_terminal"]),
            "mask": group(obs["action_mask"]),
            "alive": alive,
        }
    )
    if trajectory_kl:
        targets["factual_deter"] = stop(group(features["deter"]))
    team_valid, local_valid = endpoint_validity(first, last, present, alive, maximum)
    destination = group(agent.validity(obs)).astype(bool)
    eligible = local_valid[:, :, 1].any(-1) & destination.any(-1)
    anchors = sample_two_step_anchors(
        nj.seed(), eligible, min(int(cfg.anchors), eligible.size)
    )
    team_valid = team_valid[anchors.batch, anchors.time] & anchors.valid[:, None, None]
    local_valid = (
        local_valid[anchors.batch, anchors.time] & anchors.valid[:, None, None]
    )

    # Existing factual prediction/snapshots were made with training=True dropout.
    # Rebuild only joint snapshots in inference mode from these same detached
    # factual local states; PPO also uses training=False for every joint step.
    initial_joint = stop(entries["ctde_joint_carry"])
    _, _, snapshots = agent.ctde_joint.sequence(
        initial_joint,
        stop(group(agent.feat2tensor(features)))[:, :-1],
        actions[:, 1:],
        present[:, :-1],
        alive[:, :-1],
        first[:, :-1],
        training=False,
    )
    before_root = {
        key: jnp.concatenate([initial_joint[key][:, None], value], axis=1)
        for key, value in snapshots.items()
    }
    joint = agent.team.fold_tree_batch(
        gather_at(jax.tree.map(group, before_root), anchors)
    )
    local = agent.team.fold_tree_batch(
        gather_at(
            {
                key: group(entries[key])
                for key in ("deter", "stoch", "keys", "values", "valid", "position")
            },
            anchors,
        )
    )
    root_present = gather_at(present, anchors)
    root_alive = gather_at(alive, anchors)
    root_mask = gather_at(group(obs["action_mask"]), anchors).astype(bool)
    root_reset = gather_at(first, anchors)

    def transition(state, offset):
        state = truncate_self_fed_state(state, offset, bptt_steps)
        local, joint, current_alive, current_mask, reset = state
        action = gather_at(actions, anchors, offset)
        local_state = agent.team.unfold_batch(
            agent.feat2tensor({key: local[key] for key in ("deter", "stoch")})
        )
        joint, prediction = agent.ctde_joint.step(
            joint,
            local_state,
            action,
            root_present,
            current_alive,
            reset,
            training=False,
            **({}),
        )
        local_action = {agent.ctde_action_key: agent.team.fold_batch(action)}
        local = frozen_local_transition(
            agent.dyn,
            local,
            local_action,
            agent.team.fold_batch(prediction["embedding"]),
            agent.team.fold_batch(root_present),
            **({}),
        )
        hidden = prediction["hidden"]
        reward_output = agent.ctde_rew(hidden, 2)
        continuation_output = agent.ctde_con(hidden, 2)
        mask_output = agent.ctde_mask(hidden, 2)
        alive_output = agent.ctde_alive(hidden, 2)
        next_alive = current_alive & root_present & (alive_output.prob(1) >= 0.5)
        binary = mask_output.output if hasattr(mask_output, "output") else mask_output
        next_mask = jax.nn.sigmoid(binary.logit) >= 0.5
        noop = jnp.zeros_like(next_mask).at[..., 0].set(True)
        next_mask = jnp.where(next_mask.any(-1, keepdims=True), next_mask, noop)
        next_mask = jnp.where(next_alive[..., None], next_mask, noop)
        target = gather_at(targets, anchors, offset)
        embedding = prediction["embedding"].astype(jnp.float32)

        def normalize(value):
            return value.astype(jnp.float32) / jnp.maximum(
                jnp.linalg.norm(value.astype(jnp.float32), axis=-1, keepdims=True), 1e-8
            )

        cosine = (normalize(embedding) * normalize(target["ema"])).sum(-1)
        difference = jnp.abs(embedding - target["online"].astype(jnp.float32))
        continuation_target = (~target["terminal"]).astype(jnp.float32)
        if agent.config.contdisc:
            continuation_target *= 1 - 1 / float(agent.config.horizon)
        losses = {
            "embedding": 1 - cosine,
            "interface": jnp.where(
                difference < 1, 0.5 * difference**2, difference - 0.5
            ).mean(-1),
            "reward": reward_output.loss(target["reward"]),
            "continuation": continuation_output.loss(continuation_target),
            "action_mask": binary_vector_loss(
                mask_output, target["mask"], agent.action_mask_reduction
            ),
            "alive": alive_output.loss(target["alive"]),
        }
        if trajectory_kl:
            # Compare the actually induced recurrent posterior with the factual
            # history posterior, rather than evaluating both on predicted history.
            # Local parameters and factual targets stay frozen; the live history
            # Jacobian credits the joint producer through the existing BPTT window.
            prediction_logits = frozen_posterior(
                agent.dyn,
                agent.team.fold_batch(embedding),
                local["deter"],
                history_gradient=True,
            )
            target_logits = frozen_posterior(
                agent.dyn,
                agent.team.fold_batch(target["online"]),
                agent.team.fold_batch(target["factual_deter"]),
            )
            losses["trajectory_kl"] = agent.team.unfold_batch(
                mixed_posterior_kl(
                    prediction_logits,
                    target_logits,
                    agent.dyn.unimix,
                )
            )
        reward, continuation = shared_team_outcomes(
            reward_output.pred(),
            continuation_output.prob(1),
            root_present,
            current_alive,
            next_alive,
        )
        metrics = {
            "embedding_cosine": cosine,
            "reward_squared_error": (reward - target["reward"]) ** 2,
            "continuation_brier": (continuation - continuation_target) ** 2,
            "alive_brier": (next_alive.astype(jnp.float32) - target["alive"]) ** 2,
            "recorded_action_outside_support": ~jnp.take_along_axis(
                current_mask, action[..., None], axis=-1
            )[..., 0],
            "action_mask_error": (next_mask != target["mask"])
            .astype(jnp.float32)
            .mean(-1),
        }
        state = (local, joint, next_alive, next_mask, jnp.zeros_like(reset))
        return (state), (losses, stop(metrics))

    initial = nn.cast(stop((local, joint, root_alive, root_mask, root_reset)))
    _, (steps, step_metrics) = nj.scan(
        transition,
        initial,
        jnp.arange(1, maximum + 1, dtype=jnp.int32),
        axis=0,
    )
    output = {name: jnp.zeros_like(destination, jnp.float32) for name in steps}
    metrics = {
        "ctde/self_fed_train/anchors": anchors.valid.astype(jnp.float32).sum(),
        "ctde/self_fed_train/candidate_roots": eligible.astype(jnp.float32).sum(),
        "ctde/self_fed_train/joint_snapshot_transitions": jnp.asarray(
            present.shape[0] * (present.shape[1] - 1), jnp.float32
        ),
        "ctde/self_fed_train/rollout_transitions": jnp.asarray(
            len(anchors.batch) * maximum, jnp.float32
        ),
    }
    for horizon in horizons:
        masks = {
            name: (team_valid if name in {"reward", "continuation"} else local_valid)[
                :, horizon - 1
            ]
            for name in steps
        }
        for name, value in steps.items():
            selected = value[horizon - 1]
            mask = masks[name]
            output[name] += scatter_sample_mean(
                selected,
                anchors,
                mask,
                destination,
                team_loss=name in {"reward", "continuation"},
            ) / len(horizons)
            metrics[f"ctde/self_fed_train_h{horizon}/{name}_loss"] = jnp.where(
                mask, selected, 0.0
            ).sum() / jnp.maximum(mask.sum(), 1)
        for name, value in step_metrics.items():
            mask = (
                team_valid
                if name in {"reward_squared_error", "continuation_brier"}
                else local_valid
            )[:, horizon - 1]
            metrics[f"ctde/self_fed_train_h{horizon}/{name}"] = jnp.where(
                mask, value[horizon - 1], 0.0
            ).sum() / jnp.maximum(mask.sum(), 1)
        metrics[f"ctde/self_fed_train_h{horizon}/team_count"] = (
            team_valid[:, horizon - 1].sum().astype(jnp.float32)
        )
        metrics[f"ctde/self_fed_train_h{horizon}/root_live_count"] = (
            local_valid[:, horizon - 1].sum().astype(jnp.float32)
        )
    return {
        f"ctde_self_fed_{name}": agent.team.fold_sequence(value)
        for name, value in output.items()
    }, stop(metrics)
