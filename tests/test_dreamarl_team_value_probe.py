import jax
import jax.numpy as jnp
import numpy as np

from world_marl.dreamarl.team_value_probe import (
    ProbeConfig,
    ProbeDataset,
    _complete_episodes,
    _future_return,
    init_predictor,
    load_replay_dataset,
    paired_episode_bootstrap,
    predictor,
    train_probe,
)


def test_complete_episodes_ignores_incomplete_edges():
    first = np.array([False, True, False, False, True, False])
    last = np.array([True, False, False, True, False, False])
    assert _complete_episodes(first, last) == [(1, 4)]


def test_future_return_excludes_current_reward():
    reward = np.array([0.0, 1.0, 2.0, 3.0])
    terminal = np.array([False, False, False, True])
    result = _future_return(reward, terminal, 0, 4, 0.5)
    np.testing.assert_allclose(result, [2.75, 3.5, 3.0, 0.0])


def test_joint_predictor_is_permutation_invariant():
    params = init_predictor(jax.random.key(0), 7, 5)
    state = jax.random.normal(jax.random.key(1), (3, 4, 7))
    permutation = jnp.array([2, 0, 3, 1])
    expected = predictor(params, state, "joint")
    actual = predictor(params, state[:, permutation], "joint")
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def test_local_predictor_does_not_depend_on_other_agents():
    params = init_predictor(jax.random.key(2), 7, 5)
    state = jax.random.normal(jax.random.key(3), (2, 4, 7))
    changed = state.at[:, 1:].set(100 * state[:, 1:])
    expected = predictor(params, state, "local")[:, 0]
    actual = predictor(params, changed, "local")[:, 0]
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def test_modes_have_exactly_the_same_parameters():
    params = init_predictor(jax.random.key(4), 11, 8)
    state = jnp.zeros((2, 3, 11))
    local = predictor(params, state, "local")
    joint = predictor(params, state, "joint")
    assert local.shape == joint.shape == (2, 3, 2)
    assert sum(value.size for value in jax.tree.leaves(params)) == 250


def test_replay_loader_stitches_complete_episodes_across_chunks(tmp_path):
    episodes = 10
    length = 4
    rows = episodes * length
    first = np.zeros((rows,), bool)
    last = np.zeros((rows,), bool)
    first[::length] = True
    last[length - 1 :: length] = True
    reward = np.tile(np.arange(length, dtype=np.float32), episodes)
    arrays = {
        "dyn/deter": np.arange(rows * 2 * 3, dtype=np.float32).reshape(
            rows, 2, 3
        ),
        "dyn/stoch": np.ones((rows, 2, 1, 2), np.float32),
        "dyn/memory": np.ones((rows, 2, 1, 2), np.float32),
        "reward": reward,
        "is_first": first,
        "is_last": last,
        "is_terminal": last,
    }
    for index, (start, stop) in enumerate(((0, 17), (17, rows))):
        np.savez(
            tmp_path / f"{index:02d}.npz",
            **{key: value[start:stop] for key, value in arrays.items()},
        )
    dataset = load_replay_dataset(
        tmp_path, states_per_episode=2, gamma=0.5, seed=0
    )
    assert dataset.state.shape == (20, 2, 7)
    assert dataset.manifest["complete_episodes"] == 10
    assert set(dataset.episode) == set(range(10))


def test_probe_training_and_paired_bootstrap_smoke():
    generator = np.random.default_rng(0)
    episode = np.repeat(np.arange(10), 3)
    state = generator.normal(size=(30, 2, 6)).astype(np.float16)
    dataset = ProbeDataset(
        state=state,
        reward=state[..., 0].mean(1).astype(np.float32),
        return_=state[..., 1].mean(1).astype(np.float32),
        episode=episode,
        train_episodes=np.arange(6),
        validation_episodes=np.arange(6, 8),
        test_episodes=np.arange(8, 10),
        manifest={},
    )
    config = ProbeConfig(hidden=4, steps=2, batch_size=4)
    local = train_probe(dataset, "local", config, seed=0)
    joint = train_probe(dataset, "joint", config, seed=0)
    comparison = paired_episode_bootstrap(
        dataset, local, joint, seed=0, samples=10
    )
    assert local["parameter_count"] == joint["parameter_count"]
    assert set(comparison) == {"reward", "return"}
