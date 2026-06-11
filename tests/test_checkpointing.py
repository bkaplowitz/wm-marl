from __future__ import annotations

import jax

from world_marl.algs.ippo import IPPOConfig, create_train_state, tree_l2_distance
from world_marl.checkpointing import load_metadata, load_params, save_checkpoint


def test_checkpoint_save_load_equality(tmp_path):
  config = IPPOConfig()
  state = create_train_state(jax.random.PRNGKey(0), (8, 8, 3), 3, config)
  save_checkpoint(tmp_path, state, metadata={"hello": "world"})

  loaded = load_params(tmp_path / "checkpoint.msgpack", state.params)
  assert tree_l2_distance(state.params, loaded) == 0.0
  assert load_metadata(tmp_path)["hello"] == "world"
