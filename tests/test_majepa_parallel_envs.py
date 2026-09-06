from functools import partial
import os
from pathlib import Path

import elements
import embodied
import numpy as np
import pytest

from majepa.main import make_stream
from majepa.replay import DualViewReplay
from majepa.train import train


class TeamEnv(embodied.Env):
    def __init__(self, worker, tickdir=None):
        self.worker = worker
        self.tick = -1
        self.timestep = 0
        self.tickdir = tickdir
        self.closed = False

    @property
    def obs_space(self):
        return {
            "vector": elements.Space(np.int32, (2, 3)),
            "tick": elements.Space(np.int32),
            "pid": elements.Space(np.int32),
            "reward": elements.Space(np.float32, (2,)),
            **{
                key: elements.Space(bool)
                for key in ("is_first", "is_last", "is_terminal")
            },
        }

    @property
    def act_space(self):
        return {
            "action": elements.Space(np.int32, (2,), 0, 3),
            "reset": elements.Space(bool),
        }

    def step(self, action):
        self.tick += 1
        if self.tickdir is not None:
            (Path(self.tickdir) / f"worker-{self.worker}").write_text(
                str(self.tick + 1)
            )
        self.timestep = 0 if action["reset"] else self.timestep + 1
        done = self.timestep == self.worker + 2
        return {
            "vector": np.array(
                [[self.worker, agent, self.timestep] for agent in range(2)], np.int32
            ),
            "tick": np.asarray(self.tick, np.int32),
            "pid": np.asarray(os.getpid(), np.int32),
            "reward": np.full(2, not action["reset"], np.float32),
            "is_first": np.asarray(action["reset"], bool),
            "is_last": np.asarray(done, bool),
            "is_terminal": np.asarray(done, bool),
        }

    def close(self):
        self.closed = True


class RecurrentRandomAgent(embodied.RandomAgent):
    spaces = {"_behavior_replay/vector": elements.Space(np.int32, (2, 3))}

    def __init__(self, step, prefill, fail_at=None):
        env = TeamEnv(0)
        super().__init__(env.obs_space, env.act_space)
        self.fail_at = fail_at
        self.step = step
        self.prefill = prefill
        self.policy_calls = 0
        self.train_calls = 0
        self.pids = set()
        self.policy_batch_sizes = []
        self.train_steps = []

    def init_policy(self, batch_size):
        return {
            "state": list(np.zeros((batch_size, 2), np.int32)),
            "tick": np.zeros(batch_size, np.int32),
        }

    def policy(self, carry, obs, mode="train"):
        self.policy_calls += 1
        self.policy_batch_sizes.append(len(obs["is_first"]))
        self.pids.update(obs["pid"].tolist())
        assert len(carry["state"]) == len(obs["is_first"])
        np.testing.assert_array_equal(carry["tick"], obs["tick"])
        worker, agent, timestep = np.moveaxis(obs["vector"], -1, 0)
        identity = 1000 * worker + 100 * agent
        state = np.where(obs["is_first"][:, None], identity, np.stack(carry["state"]))
        np.testing.assert_array_equal(state, identity + timestep)
        if self.policy_calls == self.fail_at:
            raise RuntimeError("injected policy failure")
        _, acts, outs = super().policy(carry, obs, mode)
        return {"state": list(state + 1), "tick": obs["tick"] + 1}, acts, outs

    def train(self, carry, data):
        assert int(self.step) >= self.prefill
        self.train_calls += 1
        self.train_steps.append(int(self.step))
        for prefix in ("", "_behavior_replay/"):
            vector = data[prefix + "vector"]
            worker = vector[:, :, 0, 0]
            np.testing.assert_array_equal(np.diff(worker, axis=1), 0)
            np.testing.assert_array_equal(np.diff(data[prefix + "tick"], axis=1), 1)
            np.testing.assert_array_equal(
                vector[:, :, 0, 2], data[prefix + "tick"] % (worker + 3)
            )
            np.testing.assert_array_equal(vector[:, :, 1, 1], 1)
        return super().train(carry, data)


@pytest.mark.parametrize(
    "envs,budget,debug,fail_at",
    [
        (1, 43, False, None),
        (3, 203, False, None),
        (3, 2, True, None),
        (3, 43, False, 15),
    ],
)
def test_training_uses_independent_processes_and_closes_them(
    tmp_path, monkeypatch, envs, budget, debug, fail_at
):
    drivers = []
    real_driver = embodied.Driver

    def create_driver(*args, **kwargs):
        driver = real_driver(*args, **kwargs)
        drivers.append(driver)
        return driver

    monkeypatch.setattr(embodied, "Driver", create_driver)
    replay = DualViewReplay(length=3, capacity=1000, chunksize=4)
    metrics = []
    logger = elements.Logger(elements.Counter(), [metrics.extend])
    agent = RecurrentRandomAgent(logger.step, (budget + 9) // 10, fail_at)
    make_env = partial(TeamEnv, tickdir=tmp_path)
    args = elements.Config(
        logdir=str(tmp_path),
        envs=envs,
        debug=debug,
        steps=budget,
        num_agents=2,
        batch_size=2,
        batch_length=3,
        train_ratio=3.0,
        actor_critic_start_step=0,
        usage={"psutil": False},
        log_every=-1,
        report_every=0,
        save_every=0,
        final_save=False,
        curve_eval_interval=0,
        checkpoint_at_curve_eval=False,
        consec_train=1,
        consec_report=1,
        report_length=3,
        replay_context=0,
    )
    run = partial(
        train,
        lambda: agent,
        lambda: replay,
        make_env,
        partial(make_stream, args),
        lambda: logger,
    )
    try:
        if fail_at:
            with pytest.raises(RuntimeError, match="injected policy failure"):
                run(args)
        else:
            run(args)
        assert len(drivers) == 1
        assert drivers[0].parallel == (not debug)
        if debug:
            assert agent.pids == {os.getpid()}
            assert all(env.closed for env in drivers[0].envs)
        else:
            assert len(drivers[0].procs) == envs
            assert agent.pids == {proc.pid for proc in drivers[0].procs}
            assert os.getpid() not in agent.pids
            assert all(not proc.running for proc in drivers[0].procs)
        physical_counts = [
            int(path.read_text()) if path.exists() else 0
            for path in (tmp_path / f"worker-{index}" for index in range(envs))
        ]
        assert sum(physical_counts) == budget
        assert max(physical_counts) - min(physical_counts) <= 1
        assert agent.policy_batch_sizes[-1] == (budget % envs or envs)
        assert sum(agent.policy_batch_sizes) == budget
        if not fail_at:
            assert int(logger.step) == args.steps
            assert len(replay) == sum(
                max(0, count - replay.length + 1) for count in physical_counts
            )
            assert set(replay.streams) == {
                index for index, count in enumerate(physical_counts) if count
            }
            if len(replay) >= args.batch_size * args.batch_length:
                prefill = (budget + 9) // 10
                eligibility = (
                    envs * (replay.length - 1) + args.batch_size * args.batch_length
                )
                assert agent.train_steps[0] == max(prefill, eligibility)
                assert agent.train_calls == 1 + (budget - agent.train_steps[0]) // 2
            else:
                assert not agent.train_steps
            latest = {key: value for _, key, value in metrics}
            assert latest["counters/environment_steps"] == int(logger.step)
            assert latest["counters/agent_steps"] == 2 * int(logger.step)
            assert latest["counters/train_envs"] == envs
            assert latest["counters/learner_update_calls"] == len(agent.train_steps)
            assert latest["schedule/replay_prefill_steps"] == (budget + 9) // 10
            assert latest["schedule/first_learner_environment_step"] == (
                agent.train_steps[0] if agent.train_steps else -1
            )
        if debug:
            run(args)
            with pytest.raises(ValueError, match="already exceeds"):
                run(args.update(steps=1))
            assert int(logger.step) == budget
            assert all(
                env.closed and env.tick == -1
                for driver in drivers[1:]
                for env in driver.envs
            )
    finally:
        for driver in drivers:
            for proc in getattr(driver, "procs", ()):
                if proc.running:
                    proc.kill()
