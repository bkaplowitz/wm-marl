"""Recorded-reward targets and team probability correction."""

import jax
import jax.numpy as jnp
import numpy as np

from majepa.marl.axes import TeamAxis
from majepa.training.factual_value import factual_vtrace_return, joint_action_logratio


def test_factual_trace_respects_terminal_truncation_reset_and_stops_gradients():
    reward = jnp.array([[0.0, 2.0, 3.0, 0.0, 5.0]])
    first = jnp.array([[True, False, False, True, False]])
    last = jnp.array([[False, False, True, False, True]])
    value = jnp.array([[10.0, 20.0, 30.0, 40.0, 50.0]])
    present = jnp.ones_like(first)

    def target(bootstrap, logratio, terminal=last, membership=present):
        return factual_vtrace_return(
            reward,
            first,
            last,
            terminal,
            membership,
            bootstrap,
            logratio,
            discount=0.9,
            lam=1.0,
            rho_clip=1.0,
            c_clip=1.0,
        )

    returns, valid, _ = target(value, jnp.zeros_like(value))
    np.testing.assert_allclose(returns, [[4.7, 3.0, 30.0, 5.0]], atol=1e-5)
    np.testing.assert_array_equal(valid, [[True, True, False, True]])
    truncated, _, _ = target(value, jnp.zeros_like(value), last.at[0, 2].set(False))
    np.testing.assert_allclose(truncated, [[29.0, 30.0, 30.0, 5.0]], atol=1e-5)
    _, missing, _ = target(
        value, jnp.zeros_like(value), membership=present.at[0, 1].set(False)
    )
    np.testing.assert_array_equal(missing, [[False, False, False, True]])
    for gradient in jax.grad(lambda v, r: target(v, r)[0].sum(), argnums=(0, 1))(
        value, jnp.zeros_like(value)
    ):
        np.testing.assert_array_equal(gradient, 0.0)


def test_joint_ratios_include_all_acting_agents_and_clip_before_exponentiation():
    team = TeamAxis(2)
    current = jnp.log(jnp.array([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]]))
    behavior = jnp.log(jnp.full((2, 3), 0.5))
    active = jnp.array([[True, True, True], [True, False, True]])
    ratio = joint_action_logratio(current, behavior, active, team)
    np.testing.assert_allclose(jnp.exp(ratio), [[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]])
    zeros = jnp.zeros((2, 3))
    flags = zeros.astype(bool)
    present = jnp.ones_like(flags)
    value, valid, metrics = factual_vtrace_return(
        jnp.ones_like(zeros),
        flags,
        flags,
        flags,
        present,
        zeros,
        jnp.full_like(zeros, 1000.0),
        discount=0.9,
        lam=0.95,
        rho_clip=1.0,
        c_clip=1.0,
    )
    assert bool(jnp.isfinite(value).all()) and bool(valid.all())
    assert float(metrics["rho_mean"]) == 1.0
    corrected, _, _ = factual_vtrace_return(
        jnp.ones_like(zeros),
        flags,
        flags,
        flags,
        present,
        zeros,
        ratio,
        discount=0.9,
        lam=0.0,
        rho_clip=1.0,
        c_clip=1.0,
    )
    np.testing.assert_allclose(corrected, [[0.25, 0.5], [0.25, 0.5]])
