"""Tests for the LeWM-style CEM-MPC planner (world_marl.genwm.cem)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
from world_marl.scripts.train_single_genwm import (
    _evaluate_policy,
    parse_args,
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


def test_discounted_return_matches_hand_computation():
    rewards = jnp.asarray([[1.0, 1.0, 1.0]])
    continues = jnp.asarray([[1.0, 0.5, 1.0]])
    # weights: t0 -> 1, t1 -> gamma * c0 = 0.9, t2 -> gamma^2 * c0 * c1 = 0.405
    value = discounted_return(rewards, continues, gamma=0.9)
    np.testing.assert_allclose(np.asarray(value), [1.0 + 0.9 + 0.405], rtol=1e-6)


def _tiny_setup():
    config = GenWMConfig(
        arm="continuous-transformer",
        obs_dim=4,
        action_dim=2,
        action_mode="continuous",
        obs_bins=5,
        action_bins=3,
        model_dim=16,
        num_heads=2,
        num_layers=1,
        integration_steps=2,
    )
    rng = np.random.default_rng(0)
    obs_tokenizer = fit_quantile_tokenizer(
        rng.normal(size=(64, config.obs_dim)), config.obs_bins
    )
    action_tokenizer = fit_quantile_tokenizer(
        rng.uniform(-1.0, 1.0, size=(64, config.action_dim)), config.action_bins
    )
    wm_state = create_genwm_state(jax.random.PRNGKey(0), config)
    head_state = create_head_state(jax.random.PRNGKey(1), config)
    return config, wm_state, head_state, obs_tokenizer, action_tokenizer


def test_make_genwm_plan_fn_shapes_and_determinism():
    config, wm_state, head_state, obs_tok, act_tok = _tiny_setup()
    cem_config = CEMConfig(num_samples=6, topk=2, num_iters=2, horizon=2)
    plan_fn = make_genwm_plan_fn(config, cem_config, gamma=0.99)
    observations = jnp.zeros((3, config.obs_dim), dtype=jnp.float32)
    mean_init = jnp.zeros((3, cem_config.horizon, config.action_dim))
    key = jax.random.PRNGKey(7)
    plan, topk_cost = plan_fn(
        wm_state, head_state, obs_tok, act_tok, observations, mean_init, key
    )
    plan2, topk_cost2 = plan_fn(
        wm_state, head_state, obs_tok, act_tok, observations, mean_init, key
    )
    assert plan.shape == (3, cem_config.horizon, config.action_dim)
    assert topk_cost.shape == (3,)
    assert np.all(np.isfinite(np.asarray(plan)))
    assert np.all(np.isfinite(np.asarray(topk_cost)))
    np.testing.assert_allclose(np.asarray(plan), np.asarray(plan2))
    np.testing.assert_allclose(np.asarray(topk_cost), np.asarray(topk_cost2))


# ---------------------------------------------------------------------------
# CEMPlanner tests
# ---------------------------------------------------------------------------


class _FakePlanFn:
    """Deterministic plan_fn stub: plan[t] = call_index + t, records calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, wm, head, obs_tok, act_tok, observations, mean_init, key):
        self.calls.append(np.asarray(mean_init).copy())
        n, horizon, action_dim = mean_init.shape
        call_index = len(self.calls) - 1
        plan = (
            jnp.arange(horizon, dtype=jnp.float32)[None, :, None] * 0.1
            + call_index * 0.001
        )
        plan = jnp.broadcast_to(plan, (n, horizon, action_dim))
        return plan, jnp.zeros((n,), dtype=jnp.float32)


def _planner(fake, horizon, receding):
    cem_config = CEMConfig(
        horizon=horizon, receding_horizon=receding, action_low=-1.0, action_high=1.0
    )
    return CEMPlanner(
        fake,
        wm_state=None,
        head_state=None,
        obs_tokenizer=None,
        action_tokenizer=None,
        cem_config=cem_config,
        num_envs=2,
        action_dim=3,
        key=jax.random.PRNGKey(0),
    )


def test_planner_replans_every_receding_horizon_steps():
    fake = _FakePlanFn()
    planner = _planner(fake, horizon=4, receding=2)
    obs = np.zeros((2, 5), dtype=np.float32)
    for _ in range(4):
        planner.act(obs)
    assert len(fake.calls) == 2  # steps 0 and 2


def test_planner_returns_buffered_plan_slices():
    fake = _FakePlanFn()
    planner = _planner(fake, horizon=3, receding=3)
    obs = np.zeros((2, 5), dtype=np.float32)
    a0 = planner.act(obs)
    a1 = planner.act(obs)
    np.testing.assert_allclose(a0, np.full((2, 3), 0.0), atol=1e-6)
    np.testing.assert_allclose(a1, np.full((2, 3), 0.1), atol=1e-6)


def test_planner_warm_starts_shifted_mean():
    fake = _FakePlanFn()
    planner = _planner(fake, horizon=4, receding=2)
    obs = np.zeros((2, 5), dtype=np.float32)
    for _ in range(3):
        planner.act(obs)
    assert len(fake.calls) == 2
    second_init = fake.calls[1]  # (2, 4, 3)
    # first plan was [0.0, 0.1, 0.2, 0.3] per dim; shifted by receding=2 -> [0.2, 0.3, 0, 0]
    np.testing.assert_allclose(second_init[:, 0], np.full((2, 3), 0.2), atol=1e-5)
    np.testing.assert_allclose(second_init[:, 1], np.full((2, 3), 0.3), atol=1e-5)
    np.testing.assert_allclose(second_init[:, 2:], 0.0, atol=1e-6)


def test_planner_reset_clears_buffer_and_warm_start():
    fake = _FakePlanFn()
    planner = _planner(fake, horizon=4, receding=2)
    obs = np.zeros((2, 5), dtype=np.float32)
    planner.act(obs)
    planner.reset()
    planner.act(obs)
    assert len(fake.calls) == 2
    np.testing.assert_allclose(fake.calls[1], 0.0, atol=1e-6)  # fresh zero mean


def test_planner_clips_actions_to_bounds():
    class _BigPlanFn(_FakePlanFn):
        def __call__(self, *args):
            plan, cost = super().__call__(*args)
            return plan + 10.0, cost

    planner = _planner(_BigPlanFn(), horizon=2, receding=2)
    actions = planner.act(np.zeros((2, 5), dtype=np.float32))
    assert np.all(actions <= 1.0) and np.all(actions >= -1.0)


# ---------------------------------------------------------------------------
# train_single_genwm CLI/host-loop integration tests
# ---------------------------------------------------------------------------


def test_parse_args_cem_defaults_and_resolution():
    args = parse_args(
        [
            "--env",
            "brax:reacher",
            "--arm",
            "continuous-transformer",
            "--policy-optimizer",
            "cem",
        ]
    )
    assert args.policy_optimizer == "cem"
    assert args.cem_samples == 300
    assert args.cem_topk == 30
    assert args.cem_iters == 30
    assert args.cem_horizon is None  # resolved to imag_horizon in run_one
    assert args.cem_receding_horizon is None


def test_parse_args_rejects_cem_model_free():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--env",
                "brax:reacher",
                "--arm",
                "model-free",
                "--policy-optimizer",
                "cem",
            ]
        )


def test_parse_args_rejects_cem_discrete_env():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--env",
                "gymnax:CartPole-v1",
                "--arm",
                "continuous-transformer",
                "--policy-optimizer",
                "cem",
            ]
        )


class _CountingActFnAdapter:
    """Minimal host-loop adapter: 2 envs, episodes end every 4 steps."""

    num_envs = 2
    max_cycles = 4
    action_dim = 3
    action_low = -1.0
    action_high = 1.0
    observation_shape = (5,)

    def __init__(self):
        self._t = 0

    def reset(self):
        self._t = 0
        return np.zeros((self.num_envs, 1, 5), dtype=np.float32)

    def step(self, actions):
        actions = np.asarray(actions).reshape((self.num_envs, self.action_dim))
        assert np.all(actions >= -1.0) and np.all(actions <= 1.0)
        self._t += 1
        done = self._t % self.max_cycles == 0
        from world_marl.envs.meltingpot_adapter import VectorStep

        return VectorStep(
            observations=np.zeros((self.num_envs, 1, 5), dtype=np.float32),
            rewards=np.full((self.num_envs, 1), 0.25, dtype=np.float32),
            dones=np.full((self.num_envs, 1), float(done), dtype=np.float32),
            completed_returns=(
                tuple((1.0,) for _ in range(self.num_envs)) if done else ()
            ),
            completed_lengths=(
                tuple(self.max_cycles for _ in range(self.num_envs)) if done else ()
            ),
            step_infos=tuple({} for _ in range(self.num_envs)),
            infos=tuple({} for _ in range(self.num_envs)),
        )


def test_evaluate_policy_uses_act_fn_override():
    adapter = _CountingActFnAdapter()
    calls = []

    def act_fn(flat_obs):
        calls.append(flat_obs.shape)
        return np.zeros((adapter.num_envs, adapter.action_dim), dtype=np.float32)

    value = _evaluate_policy(
        adapter,
        None,
        episodes=2,
        action_mode="continuous",
        rng=np.random.default_rng(0),
        act_fn=act_fn,
    )
    assert np.isfinite(value)
    assert len(calls) > 0
    assert all(shape == (2, 5) for shape in calls)
