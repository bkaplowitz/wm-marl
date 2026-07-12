from __future__ import annotations

import os

import pytest

from world_marl.baselines.dreamerv3.config import (
  DreamerV3RunSpec,
  default_dreamerv3_python,
)
from world_marl.baselines.dreamerv3.launcher import run_training


@pytest.mark.dreamerv3_integration
@pytest.mark.skipif(
  os.environ.get("RUN_DREAMERV3_INTEGRATION") != "1",
  reason="set RUN_DREAMERV3_INTEGRATION=1 to run upstream DMC smoke",
)
def test_official_dreamerv3_dmc_debug_smoke(tmp_path):
  python = default_dreamerv3_python()
  if not python.exists():
    pytest.skip("isolated DreamerV3 environment is not installed")
  spec = DreamerV3RunSpec(
    experiment_dir=tmp_path / "smoke",
    python=python,
    platform="cpu",
    train_steps=32,
    configs=("dmc_proprio", "debug"),
    save_every_seconds=1,
    extra_args=("--run.envs", "1", "--run.train_ratio", "1"),
  )
  assert run_training(spec) == 0
  assert (spec.upstream_logdir / "config.yaml").is_file()
  assert (spec.experiment_dir / "normalized" / "official_reference.json").is_file()
