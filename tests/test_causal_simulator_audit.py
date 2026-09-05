"""Frozen paired simulator tests, independent of any running environment."""

import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from majepa.main import _load_configs, _resolve_config_profiles
from scripts.audit_causal_simulator import main, make_auditor, select_roots
from scripts.probe_control_sufficiency import Episode


def tiny_inputs():
    agents, length = 2, 6
    return (
        jax.random.normal(jax.random.key(10), (agents, length, 6)),
        jnp.asarray([[1, 2, 1, 3, 1, 0], [2, 1, 3, 1, 1, 0]], jnp.int32),
        jnp.asarray([True, False, False, False, False, False]),
        jnp.ones((agents, length), bool),
        jnp.ones((length, agents), bool),
        jnp.ones((length, agents), bool).at[4:, 1].set(False),
        jnp.repeat(jnp.arange(length, dtype=jnp.float32)[:, None], agents, axis=1),
        jnp.zeros((length, agents), bool).at[-1].set(True),
        jnp.asarray(1, jnp.int32),
    )


def test_paired_simulator_couples_factual_oracle_draws_and_keeps_alignment():
    config = _resolve_config_profiles(
        _load_configs(), ("smac_vector", "ma_jepa", "debug")
    ).update(
        {
            "agent.dyn.parallel_transformer.deter": 8,
            "agent.dyn.parallel_transformer.hidden": 8,
        }
    )
    forward, initialize = make_auditor(config, 6, 4, 4)
    inputs = tiny_inputs()
    params = initialize({}, *inputs, seed=20)
    evaluate = jax.jit(forward)
    _, result = evaluate(params, *inputs, seed=21)
    teacher = result["paths"]["teacher_factual"]
    oracle = result["paths"]["oracle_observation_alive"]
    np.testing.assert_allclose(result["actual_reward"][:, 0], [2, 3, 4, 5])
    np.testing.assert_allclose(oracle["posterior_kl"], 0.0, atol=1e-6)
    # With the exact observation/liveness path, recurrent predicted signals
    # match the parallel teacher path under the very same stochastic history.
    for key in ("reward", "continuation", "alive_probability", "embedding_cosine"):
        np.testing.assert_allclose(teacher[key], oracle[key], atol=0.04, rtol=0.04)
    # Return labels include rewards after focal death; cohort stays root-live.
    np.testing.assert_array_equal(result["cohort"], [True, True])
    assert float(result["actual_return"][-1, 1]) > 13.0

    changed = list(inputs)
    changed[0] = changed[0].at[:, 4:].add(100.0)
    _, intervention = evaluate(params, *changed, seed=21)
    # Predictions at t=2,3 cannot depend on factual observations at t>=4.
    for path in result["paths"]:
        for key in ("reward", "continuation", "mask", "alive_probability"):
            np.testing.assert_array_equal(
                result["paths"][path][key][:2], intervention["paths"][path][key][:2]
            )

    changed_action = list(inputs)
    changed_action[1] = changed_action[1].at[:, 1].set(0)
    _, action_intervention = evaluate(params, *changed_action, seed=21)
    assert not np.allclose(
        result["paths"]["teacher_factual"]["embedding_cosine"][0],
        action_intervention["paths"]["teacher_factual"]["embedding_cosine"][0],
    )


def test_root_selection_uses_common_full_future_support():
    data = {
        "observation": np.zeros((12, 2, 3)),
        "action_mask": np.ones((12, 2, 4), bool),
        "is_first": np.asarray([True] + [False] * 5 + [True] + [False] * 5),
    }
    roots = select_roots([Episode("e", "chunk", 0, data)], 3, 6, 4)
    assert {root for _, root in roots} == {0, 1, 2, 6, 7, 8}


def test_offline_audit_end_to_end_outputs_paired_records(tmp_path):
    config = _resolve_config_profiles(
        _load_configs(), ("smac_vector", "ma_jepa", "debug")
    ).update(
        {
            "agent.num_agents": 2,
            "agent.dyn.parallel_transformer.deter": 8,
            "agent.dyn.parallel_transformer.hidden": 8,
        }
    )
    config.save(str(tmp_path / "config.yaml"))
    _, initialize = make_auditor(config, 6, 4, 4)
    params = initialize({}, *tiny_inputs(), seed=30)
    checkpoint = tmp_path / "ckpt/example"
    checkpoint.mkdir(parents=True)
    (checkpoint / "done").touch()
    (tmp_path / "ckpt/latest").write_text("example")
    with (checkpoint / "agent.pkl").open("wb") as stream:
        pickle.dump({"params": jax.device_get(params), "counters": {}}, stream)
    replay = tmp_path / "replay"
    replay.mkdir()
    length = 6 * 10
    rng = np.random.default_rng(31)
    first = np.zeros(length, bool)
    first[::10] = True
    last = np.zeros(length, bool)
    last[9::10] = True
    np.savez_compressed(
        replay / "example.npz",
        observation=rng.normal(size=(length, 2, 6)).astype(np.float32),
        action=rng.integers(0, 4, size=(length, 2), dtype=np.int32),
        action_mask=np.ones((length, 2, 4), bool),
        reward=np.repeat(rng.uniform(size=(length, 1)).astype(np.float32), 2, axis=1),
        is_first=first,
        is_last=last,
        is_terminal=last,
        stepid=np.arange(length, dtype=np.int64),
    )
    output = tmp_path / "audit.json"
    assert (
        main(
            [
                "--run",
                str(tmp_path),
                "--output",
                str(output),
                "--platform",
                "cpu",
                "--roots",
                "8",
                "--max-episodes",
                "6",
                "--horizons",
                "1",
                "2",
                "4",
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert len(result["records"]) == 8
    for horizon in (1, 2, 4):
        actual = result["summary"]["oracle_observation_alive"][f"h{horizon}"]
        assert actual["agent_roots"] == 16
        assert actual["mixed_posterior_kl"] < 1e-6
    assert "paired_self_feed_excess_return_mse" in result["summary"]
