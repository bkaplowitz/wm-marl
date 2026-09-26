"""Input-only gradients from factual joint outcomes, with unchanged head updates."""

import jax
import jax.numpy as jnp
import ninjax as nj


def outcome_input_losses(agent, joint_args, reward, continuation, alive, rng,
                         embedding_target=None):
    """Replay the factual pass with frozen parameters and identical dropout.

    The caller adds ``loss - stop_gradient(loss)`` to the ordinary loss. Its
    forward value is zero and only the local input state receives gradients.
    The ordinary pass still trains the joint model and heads exactly once.
    This nested context restores the saved RNG reserve without advancing the
    caller's stream. No additional sampling is introduced into training.
    An optional stopped EMA target also supervises the same pass's embedding;
    its input gradient is weighted separately by the caller.
    """
    modules = (agent.ctde_joint, agent.ctde_rew, agent.ctde_con, agent.ctde_alive)
    prefixes = tuple(module.path + "/" for module in modules)
    frozen = {
        key: jax.lax.stop_gradient(value)
        for key, value in nj.context().items()
        if key.startswith(prefixes)
    }

    def forward():
        nj.context().reserve = list(rng[1])
        cache, *args = joint_args
        _, prediction, _ = agent.ctde_joint.sequence(
            jax.lax.stop_gradient(cache), *args, stop_state_gradient=False)
        hidden = prediction["hidden"]
        losses = (
            agent.ctde_rew(hidden, 3).loss(reward),
            agent.ctde_con(hidden, 3).loss(continuation),
            agent.ctde_alive(hidden, 3).loss(alive),
        )
        if embedding_target is not None:
            embedding = prediction["embedding"].astype(jnp.float32)
            target = jax.lax.stop_gradient(embedding_target.astype(jnp.float32))
            unit = embedding / jnp.maximum(
                jnp.linalg.norm(embedding, axis=-1, keepdims=True), 1e-8)
            target_unit = target / jnp.maximum(
                jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-8)
            losses += (1.0 - jnp.sum(unit * target_unit, axis=-1),)
        return losses

    _, losses = nj.pure(forward, nested=True)(
        frozen, seed=rng[0], create=False, modify=False)
    return losses
