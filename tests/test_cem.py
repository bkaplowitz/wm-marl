"""Tests for the LeWM-style CEM-MPC planner (world_marl.genwm.cem)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import functools

from world_marl.genwm import (
    CEMConfig,
    CEMPlanner,
    GenWMConfig,
    cem_solve,
    create_genwm_state,
    create_head_state,
    discounted_return,
    fit_quantile_tokenizer,
    make_genwm_plan_fn,
    sample_candidates,
)


def test_sample_candidates_forces_mean_first():
    key = jax.random.PRNGKey(0)
    mean = jnp.arange(2 * 3 * 2, dtype=jnp.float32).reshape(2, 3, 2)
    std = jnp.ones_like(mean)
    candidates = sample_candidates(key, mean, std, num_samples=5)
    assert candidates.shape == (5, 2, 3, 2)
    np.testing.assert_allclose(np.asarray(candidates[0]), np.asarray(mean))
    assert not np.allclose(np.asarray(candidates[1]), np.asarray(mean))


def test_cem_solve_converges_on_quadratic_cost():
    config = CEMConfig(num_samples=64, topk=8, num_iters=20, horizon=3)
    target = jnp.asarray(np.linspace(-0.5, 0.5, 2 * 3 * 2), dtype=jnp.float32).reshape(
        2, 3, 2
    )

    def cost_fn(candidates, key):
        del key
        return jnp.sum((candidates - target[None]) ** 2, axis=(2, 3))

    mean_init = jnp.zeros((2, 3, 2), dtype=jnp.float32)
    mean, std, topk_cost = cem_solve(cost_fn, jax.random.PRNGKey(1), mean_init, config)
    assert mean.shape == (2, 3, 2)
    assert std.shape == (2, 3, 2)
    assert topk_cost.shape == (2,)
    np.testing.assert_allclose(np.asarray(mean), np.asarray(target), atol=0.05)
    assert float(topk_cost.mean()) < 0.01


def test_cem_solve_batches_envs_independently():
    config = CEMConfig(num_samples=64, topk=8, num_iters=20, horizon=1)
    targets = jnp.asarray([[[0.7]], [[-0.7]]], dtype=jnp.float32)  # (2, 1, 1)

    def cost_fn(candidates, key):
        del key
        return jnp.sum((candidates - targets[None]) ** 2, axis=(2, 3))

    mean, _, _ = cem_solve(cost_fn, jax.random.PRNGKey(2), jnp.zeros((2, 1, 1)), config)
    np.testing.assert_allclose(np.asarray(mean), np.asarray(targets), atol=0.05)


def test_discounted_return():
    gamma = 0.9
    H = 5
    S, N = 1, 1
    rewards = jnp.ones((S, N, H), dtype=jnp.float32)
    continue_probs = jnp.ones((S, N, H), dtype=jnp.float32)
    result = discounted_return(rewards, continue_probs, gamma)
    assert result.shape == (S, N)
    expected = sum(gamma**t for t in range(H))  # 1 + 0.9 + 0.81 + 0.729 + 0.6561
    np.testing.assert_allclose(float(result[0, 0]), expected, rtol=1e-5)


def test_discounted_return_zero_continue():
    gamma = 0.99
    H = 4
    S, N = 2, 3
    rewards = jnp.ones((S, N, H), dtype=jnp.float32)
    # continue_prob=0 after step 0 means episode terminates; only t=0 contributes
    continue_probs = jnp.zeros((S, N, H), dtype=jnp.float32)
    result = discounted_return(rewards, continue_probs, gamma)
    # t=0: gamma^0 * prod_{s<0}() * r_0 = 1 (empty product = 1)
    # t>=1: gamma^t * 0 * r_t = 0 (first continue_prob = 0 kills the cumprod)
    np.testing.assert_allclose(np.asarray(result), np.ones((S, N)), rtol=1e-5)


def _genwm_config() -> GenWMConfig:
    return GenWMConfig(
        arm="continuous-transformer",
        obs_dim=4,
        action_dim=2,
        action_mode="continuous",
        obs_bins=5,
        action_bins=3,
        model_dim=16,
        num_heads=2,
        num_layers=1,
        integration_steps=3,
        block_size=2,
        steps_per_block=2,
    )


def test_make_genwm_plan_fn():
    rng = np.random.default_rng(0)
    config = _genwm_config()
    obs_tokenizer = fit_quantile_tokenizer(
        rng.normal(size=(256, config.obs_dim)).astype(np.float32), config.obs_bins
    )
    action_tokenizer = fit_quantile_tokenizer(
        rng.uniform(-1.0, 1.0, size=(256, config.action_dim)).astype(np.float32),
        config.action_bins,
    )
    wm_state = create_genwm_state(jax.random.PRNGKey(1), config)
    head_state = create_head_state(jax.random.PRNGKey(2), config)

    S, N, H, A = 4, 2, 3, config.action_dim
    cem_config = CEMConfig(
        num_samples=S, topk=2, num_iters=2, horizon=H, receding_horizon=H
    )
    start_obs = jnp.zeros((N, config.float_obs_dim), dtype=jnp.float32)
    cost_fn = make_genwm_plan_fn(
        wm_state=wm_state,
        head_state=head_state,
        start_observations=start_obs,
        obs_tokenizer=obs_tokenizer,
        action_tokenizer=action_tokenizer,
        config=config,
        cem_config=cem_config,
        gamma=0.99,
    )
    candidates = jnp.zeros((S, N, H, A), dtype=jnp.float32)
    costs = cost_fn(candidates, jax.random.PRNGKey(3))
    assert costs.shape == (S, N), f"Expected ({S}, {N}), got {costs.shape}"
    assert bool(jnp.all(jnp.isfinite(costs))), "Costs must be finite"


# ---------------------------------------------------------------------------
# CEMPlanner tests
# ---------------------------------------------------------------------------


def _tiny_cem_config() -> CEMConfig:
    return CEMConfig(
        num_samples=4,
        topk=2,
        num_iters=2,
        horizon=2,
        receding_horizon=2,
        init_std=1.0,
        action_low=-1.0,
        action_high=1.0,
    )


def test_cem_planner_reset():
    config = _tiny_cem_config()
    A = 2

    def dummy_make_plan_fn(**_):
        def cost_fn(c, k):
            return jnp.zeros(c.shape[:2])

        return cost_fn

    planner = CEMPlanner(
        dummy_make_plan_fn, jax.random.PRNGKey(0), config, action_dim=A
    )
    planner._ensure_buffers(N=3)

    planner._step_index = 1
    planner._needs_replan = False
    planner._action_buffer = jnp.ones_like(planner._action_buffer)

    planner.reset()

    assert planner._step_index == 0
    assert planner._needs_replan is True
    np.testing.assert_allclose(np.asarray(planner._action_buffer), 0.0)


def test_cem_planner_warm_start_shift():
    config = _tiny_cem_config()
    H, N, A = config.horizon, 2, 3

    def dummy_make_plan_fn(**_):
        def cost_fn(c, k):
            return jnp.zeros(c.shape[:2])

        return cost_fn

    planner = CEMPlanner(
        dummy_make_plan_fn, jax.random.PRNGKey(1), config, action_dim=A
    )
    planner._ensure_buffers(N=N)

    known = jnp.arange(N * H * A, dtype=jnp.float32).reshape(N, H, A)
    planner._action_buffer = known

    planner._warm_start_shift()

    buf = np.asarray(planner._action_buffer)
    assert buf.shape == (N, H, A)
    np.testing.assert_allclose(buf[:, : H - 1, :], np.asarray(known[:, 1:, :]))
    np.testing.assert_allclose(buf[:, H - 1, :], 0.0)


def test_cem_planner_act_smoke():
    rng = np.random.default_rng(42)
    config = _genwm_config()

    obs_tokenizer = fit_quantile_tokenizer(
        rng.normal(size=(256, config.obs_dim)).astype(np.float32), config.obs_bins
    )
    action_tokenizer = fit_quantile_tokenizer(
        rng.uniform(-1.0, 1.0, size=(256, config.action_dim)).astype(np.float32),
        config.action_bins,
    )
    wm_state = create_genwm_state(jax.random.PRNGKey(10), config)
    head_state = create_head_state(jax.random.PRNGKey(11), config)

    N = 1
    cem_config = CEMConfig(
        num_samples=4, topk=2, num_iters=2, horizon=2, receding_horizon=2
    )
    start_obs = jnp.zeros((N, config.float_obs_dim), dtype=jnp.float32)

    make_plan_fn = functools.partial(
        make_genwm_plan_fn,
        config=config,
        cem_config=cem_config,
    )

    planner = CEMPlanner(
        make_plan_fn, jax.random.PRNGKey(20), cem_config, action_dim=config.action_dim
    )

    key = jax.random.PRNGKey(21)
    actions = planner.act(
        flat_obs=start_obs,
        key=key,
        wm_state=wm_state,
        head_state=head_state,
        start_obs_for_horizon=start_obs,
        current_obs_tokenizer=obs_tokenizer,
        current_action_tokenizer=action_tokenizer,
        gamma=0.99,
    )

    assert actions.shape == (N, config.action_dim), (
        f"Expected ({N}, {config.action_dim}), got {actions.shape}"
    )
    assert bool(jnp.all(jnp.isfinite(actions))), "Actions must be finite"
    assert planner._topk_costs is not None
    assert planner._solve_seconds > 0.0
