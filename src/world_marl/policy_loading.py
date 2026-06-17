"""Checkpoint policy loading helpers shared by validation CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax

from world_marl.algs.ippo import IPPOConfig
from world_marl.algs.mappo import MAPPOConfig
from world_marl.checkpointing import load_metadata, load_params
from world_marl.scripts.train_e2e import (
  create_algorithm_train_state,
  policy_from_train_state,
)


def load_checkpoint_policy(
  checkpoint_dir: str | Path,
  adapter: Any,
  *,
  deterministic: bool,
  seed: int,
):
  """Load an IPPO/MAPPO checkpoint as an environment policy function."""
  checkpoint_path = Path(checkpoint_dir)
  metadata = load_metadata(checkpoint_path)
  algorithm = metadata.get("algorithm", "ippo")
  config_payload = metadata.get("algorithm_config", metadata.get("ippo_config"))
  if config_payload is None:
    raise KeyError("checkpoint metadata missing algorithm_config")
  config = MAPPOConfig(**config_payload) if algorithm == "mappo" else IPPOConfig(**config_payload)

  expected_substrate = metadata.get("substrate")
  if expected_substrate is not None and expected_substrate != adapter.substrate:
    raise ValueError(
      f"checkpoint substrate {expected_substrate!r} does not match "
      f"adapter substrate {adapter.substrate!r}"
    )
  expected_action_dim = metadata.get("action_dim")
  if expected_action_dim is not None and int(expected_action_dim) != adapter.action_dim:
    raise ValueError(
      f"checkpoint action_dim {expected_action_dim} does not match "
      f"adapter action_dim {adapter.action_dim}"
    )
  expected_num_agents = metadata.get("num_agents")
  if expected_num_agents is not None and int(expected_num_agents) != adapter.num_agents:
    raise ValueError(
      f"checkpoint num_agents {expected_num_agents} does not match "
      f"adapter num_agents {adapter.num_agents}"
    )
  expected_observation_shape = metadata.get("observation_shape")
  if expected_observation_shape is not None:
    expected_observation_shape = tuple(int(dim) for dim in expected_observation_shape)
    if expected_observation_shape != adapter.observation_shape:
      raise ValueError(
        "checkpoint observation_shape "
        f"{expected_observation_shape} does not match adapter "
        f"observation_shape {adapter.observation_shape}."
      )
  observation_mode = metadata.get("observation_mode", "vector")
  if observation_mode != "vector":
    raise ValueError(
      "CoinGame checkpoint policy loading expects vector-mode checkpoints; "
      f"got observation_mode={observation_mode!r}"
    )

  train_state = create_algorithm_train_state(
    algorithm,
    jax.random.PRNGKey(0),
    adapter,
    config,
    observation_mode=observation_mode,
  )
  params = load_params(checkpoint_path / "checkpoint.msgpack", train_state.params)
  train_state = train_state.replace(params=params)
  return (
    policy_from_train_state(
      algorithm,
      train_state,
      adapter=adapter,
      deterministic=deterministic,
      seed=seed,
      observation_mode=observation_mode,
    ),
    metadata,
  )
