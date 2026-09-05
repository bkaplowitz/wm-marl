"""Frozen real-policy diagnostic gates; no SMAC engine or GPU is needed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import threading

import numpy as np
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "evaluate_value_calibration",
    Path(__file__).parents[1] / "scripts" / "evaluate_value_calibration.py",
)
calibration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calibration)


def episode(reward=(0, 1, 2, 4), terminal=True):
    length = len(reward)
    alive = np.ones((length, 2), bool)
    alive[2:, 0] = False
    values = np.zeros((length, 2), np.float32)
    values[:, 0] = np.arange(length) + 10
    values[:, 1] = np.arange(length) + 20
    return {
        "observation": np.zeros((length, 2, 3), np.float32),
        "reward": np.broadcast_to(np.asarray(reward, np.float32)[:, None], (length, 2)),
        "action": np.zeros((length, 2), np.int32),
        "action_mask": np.ones((length, 2, 3), bool),
        "agent_present": np.ones((length, 2), bool),
        "agent_alive": np.ones((length, 2), bool),
        "controllable_alive": alive,
        "is_first": np.arange(length) == 0,
        "is_last": np.arange(length) == length - 1,
        "is_terminal": (np.arange(length) == length - 1) & terminal,
        "calibration/live": values,
        "calibration/slow": values + 5,
    }


def test_returns_align_to_successor_reward_and_keep_team_reward_after_death():
    result = calibration.episode_targets(episode(), 0.5, horizons=(1, 2, 5))
    np.testing.assert_allclose(result["monte_carlo"], [[3, 3], [4, 4], [4, 4], [0, 0]])
    # Unit 0 is dead at t=2 but its team still receives reward 4.
    assert result["monte_carlo"][2, 0] == 4
    np.testing.assert_allclose(result["h2/live_bootstrap"][0], [5, 7.5])
    np.testing.assert_allclose(result["h2/slow_bootstrap"][0], [6.25, 8.75])
    np.testing.assert_allclose(result["h2/live_bootstrap"][1:3], [[4, 4], [4, 4]])
    np.testing.assert_allclose(
        result["h5/live_bootstrap"][:3], result["monte_carlo"][:3]
    )


@pytest.mark.parametrize(
    "kind", ("reset", "last", "terminal", "initial_reward", "unshared_reward")
)
def test_episode_labels_reject_internal_boundaries_and_reward_mismatch(kind):
    data = {key: value.copy() for key, value in episode().items()}
    if kind in {"reset", "last", "terminal"}:
        data[{"reset": "is_first", "last": "is_last", "terminal": "is_terminal"}[kind]][
            1
        ] = True
    elif kind == "initial_reward":
        data["reward"][0] = 7
    else:
        data["reward"][2, 1] = 99
    with pytest.raises(ValueError):
        calibration.episode_targets(data, 0.5)


def test_truncated_episodes_do_not_claim_observed_mc_or_unavailable_horizon():
    result = calibration.episode_targets(episode(terminal=False), 0.5, horizons=(2, 5))
    assert np.isnan(result["monte_carlo"]).all()
    assert np.isfinite(result["h2/live_bootstrap"][:2]).all()
    assert np.isnan(result["h2/live_bootstrap"][2:]).all()
    assert np.isnan(result["h5/live_bootstrap"]).all()


def test_summaries_keep_episode_boundaries_death_samples_and_compact_raw():
    records = [
        {"worker": 0, "worker_index": 100, "worker_episode": 0, "data": episode()},
        {
            "worker": 0,
            "worker_index": 100,
            "worker_episode": 1,
            "data": episode((0, 100, 200)),
        },
    ]
    summary, raw = calibration.summarize(records, gamma=0.5, sample_stride=10)
    assert summary["episodes"] == 2
    assert summary["return_mean"] == 153.5  # Never double broadcast team rewards.
    np.testing.assert_array_equal(raw["episode_offsets"], [0, 4, 7])
    np.testing.assert_allclose(raw["monte_carlo_return"], [3, 4, 4, 0, 200, 200, 0])
    np.testing.assert_array_equal(raw["sampled_state"], [1, 0, 1, 0, 1, 0, 0])
    dead = summary["calibration"]["live"]["monte_carlo"]["dead"]
    assert dead["count"] == 1 and dead["target_mean"] == 4
    assert summary["calibration"]["live"]["monte_carlo"]["terminal"]["target_mean"] == 0
    assert not any("deter" in key or "stoch" in key for key in raw)


class Counter:
    def __init__(self, value):
        self.value = value
        self.lock = threading.Lock()


class Agent:
    def __init__(self):
        self.n_actions = Counter(37)
        self.policy_lock = threading.Lock()
        self.pending_sync = {"preserve": True}
        self.actions = []

    def init_policy(self, batch_size):
        return np.zeros(batch_size, np.int32)

    def policy(self, carry, observation, *, mode):
        assert mode in {"eval", "eval_sample"}
        rng = np.random.default_rng([123, self.n_actions.value])
        action = rng.integers(0, 3, observation["agent_present"].shape, np.int32)
        self.n_actions.value += 1
        carry = np.where(observation["is_first"], 0, carry) + 1
        self.actions.append(action.copy())
        return carry, {"action": action}, {}


class Environment:
    def __init__(self, worker):
        import elements

        self.worker = worker
        self.act_space = {
            "action": elements.Space(np.int32, (2,), 0, 3),
            "reset": elements.Space(bool),
        }
        self.rows = episode((0, 1, 2, 4) if worker % 2 == 0 else (0, 3, 5))
        self.position = -1
        self.closed = False

    def step(self, action):
        self.position = 0 if action["reset"] else self.position + 1
        assert self.position < len(self.rows["reward"])
        return {
            key: value[self.position]
            for key, value in self.rows.items()
            if key not in (*calibration.VALUE_FIELDS, "action")
        }

    def close(self):
        self.closed = True


def test_canonical_driver_exact_quotas_and_diagnostics_do_not_change_actor():
    first, second = Agent(), Agent()
    environments = []

    def make_env(worker):
        env = Environment(worker)
        environments.append(env)
        return env

    def zero(carry, observation):
        del carry
        values = np.zeros_like(observation["reward"])
        return {"live": values, "slow": values}

    def noisy(carry, observation):
        # Deliberately consume lots of an unrelated random stream.
        values = np.random.default_rng(int(carry.sum())).normal(
            size=observation["reward"].shape
        )
        return {"live": values, "slow": values + 1}

    kwargs = dict(episodes=5, envs=2, worker_offset=100, max_driver_steps=100)
    left = calibration.collect_episodes(first, make_env, zero, **kwargs)
    right = calibration.collect_episodes(second, make_env, noisy, **kwargs)
    assert len(left) == len(right) == 5
    assert sum(row["worker"] == 0 for row in left) == 3
    assert sum(row["worker"] == 1 for row in left) == 2
    for x, y in zip(first.actions, second.actions, strict=True):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(left, right, strict=True):
        for key in calibration.RAW_FIELDS:
            np.testing.assert_array_equal(x["data"][key], y["data"][key])
    assert first.n_actions.value == second.n_actions.value == 37
    assert first.pending_sync == second.pending_sync == {"preserve": True}
    assert all(env.closed for env in environments)


def test_collector_closes_and_restores_actor_rng_on_budget_failure():
    agent = Agent()
    env = Environment(0)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        calibration.collect_episodes(
            agent,
            lambda _: env,
            lambda carry, obs: {
                key: np.zeros_like(obs["reward"]) for key in ("live", "slow")
            },
            episodes=2,
            max_driver_steps=1,
        )
    assert agent.n_actions.value == 37 and env.closed


def test_output_refuses_to_overwrite(tmp_path):
    path = tmp_path / "summary.json"
    calibration.write_json(path, {"int": np.int64(1), "array": np.ones(2)})
    with pytest.raises(FileExistsError):
        calibration.write_json(path, {"replacement": True})


def test_full_cli_frozen_checkpoint_protocol_and_raw_archive(monkeypatch, tmp_path):
    import elements
    from majepa import main as entrypoint

    config = entrypoint._resolve_config_profiles(
        entrypoint._load_configs(), ("smac_vector", "ma_jepa", "debug")
    )
    config_path = tmp_path / "config.yaml"
    config.save(elements.Path(config_path))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "done").touch()
    (checkpoint / "agent.pkl").write_bytes(b"stub checkpoint payload")
    output = tmp_path / "calibration"
    agent = Agent()
    agent.params = {"weights": np.arange(3), "opt/moments": np.ones(3)}
    agent.n_updates, agent.n_batches = Counter(5), Counter(6)
    loaded = []

    class Checkpoint:
        def load(self, path, keys):
            assert self.agent is agent
            loaded.append((path, keys))

    class Diagnostic:
        def __init__(self, frozen_agent, seed):
            assert frozen_agent is agent and seed == 9001
            self.calls = 0

        def __call__(self, carry, obs):
            self.calls += 1
            return {key: np.zeros_like(obs["reward"]) for key in ("live", "slow")}

    monkeypatch.setattr(entrypoint, "make_agent", lambda config: agent)
    monkeypatch.setattr(
        entrypoint, "make_env", lambda config, worker: Environment(worker)
    )
    monkeypatch.setattr(elements, "Checkpoint", Checkpoint)
    monkeypatch.setattr(calibration, "FrozenCentralCritic", Diagnostic)
    arguments = [
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--episodes",
        "32",
        "--platform",
        "cpu",
    ]
    assert calibration.main(arguments) == 0
    assert loaded == [(str(checkpoint), ["agent"])]
    summary = json.loads((output / "summary.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    assert summary["episodes"] == summary["complete_terminal_episodes"] == 32
    assert summary["frozen_state_before"] == summary["frozen_state_after"]
    assert summary["frozen_state_before"]["counters"] == {
        "n_updates": 5,
        "n_batches": 6,
        "n_actions": 37,
    }
    assert provenance["protocol"]["policy_mode"] == "eval_sample"
    assert provenance["protocol"]["environment_seeds"] == [150000]
    assert provenance["checkpoint_files"]["agent.pkl"][
        "sha256"
    ] == calibration.digest_file(checkpoint / "agent.pkl")
    with np.load(output / "episodes.npz", allow_pickle=False) as raw:
        assert len(raw["episode_offsets"]) == 33
        assert raw["observation"].shape == (128, 2, 3)
        assert raw["is_terminal"][raw["episode_offsets"][1:] - 1].all()
    assert (output / "done.json").is_file()
    with pytest.raises(FileExistsError):
        calibration.main(arguments)
    with pytest.raises(ValueError, match="at least 32"):
        calibration.main(arguments + ["--episodes", "31"])


def test_real_policy_critic_jit_keeps_carry_actions_params_and_rng_unchanged():
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from test_majepa_learner_smoke import _tiny_learner

    learner, obs_space, _ = _tiny_learner()
    observation = {
        key: np.zeros((2, *space.shape), space.dtype)
        for key, space in obs_space.items()
    }
    observation["observation"] = (
        np.random.default_rng(7).normal(size=(2, 2, 6)).astype(np.float32)
    )
    for key in (
        "agent_present",
        "agent_alive",
        "controllable_alive",
        "action_mask",
        "is_first",
    ):
        observation[key][...] = True
    observation["controllable_alive"][0, 0] = False
    observation["action_mask"][0, 0, 1:] = False
    carry = learner.init_policy(2)

    def initialize(carry, obs):
        carry, action, out = learner.policy(carry, obs, "eval_sample")
        value = calibration.critic_forward(
            learner,
            {key: carry[1][key] for key in ("deter", "stoch")},
            obs["agent_present"],
            obs["controllable_alive"],
        )
        return carry, action, out, value

    state = nj.init(initialize)({}, carry, observation, seed=74)
    # Only policy/critic modules were initialized; no learner or optimizer needed.
    assert not any(key.startswith("opt/") for key in state)
    pure = nj.pure(learner.policy)

    def policy_call(params, carry, observation, seed):
        return pure(
            params,
            carry,
            observation,
            "eval_sample",
            seed=seed,
            create=False,
            modify=False,
        )[1]

    policy_call = jax.jit(policy_call)

    def split(tree):
        return jax.tree.map(lambda value: list(value), tree)

    def stack(tree):
        return jax.tree.map(
            jnp.stack, tree, is_leaf=lambda value: isinstance(value, list)
        )

    class Runtime(Agent):
        def __init__(self):
            super().__init__()
            self.model, self.params = learner, state

        def policy(self, carry, obs, *, mode):
            assert mode == "eval_sample"
            result = policy_call(
                self.params, stack(carry), obs, jax.random.PRNGKey(self.n_actions.value)
            )
            self.n_actions.value += 1
            carry, actions, outs = result
            return split(carry), jax.device_get(actions), jax.device_get(outs)

    baseline, measured = Runtime(), Runtime()
    digest = calibration.parameter_digest(measured)
    diagnostic = calibration.FrozenCentralCritic(measured, seed=944)
    policy = calibration.instrument_policy(measured, diagnostic, "eval_sample")
    direct_carry, instrumented_carry = split(carry), split(carry)
    for step in range(3):
        observation["is_first"][...] = step == 0
        direct_carry, direct_action, _ = baseline.policy(
            direct_carry, observation, mode="eval_sample"
        )
        instrumented_carry, action, output = policy(instrumented_carry, observation)
        for left, right in zip(
            jax.tree.leaves(direct_carry),
            jax.tree.leaves(instrumented_carry),
            strict=True,
        ):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(direct_action["action"], action["action"])
        assert action["action"][0, 0] == 0  # Dead unit's forced no-op is preserved.
        for key in calibration.VALUE_FIELDS:
            assert output[key].shape == (2, 2) and np.isfinite(output[key]).all()
    assert measured.n_actions.value == baseline.n_actions.value == 40
    assert diagnostic.calls == 3
    assert calibration.parameter_digest(measured) == digest
