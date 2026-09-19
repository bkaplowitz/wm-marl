import os
import random

import numpy as np

from majepa.launcher import runtime_environment
from majepa.main import _load_configs, _resolve_config_profiles, _worker_seed
from majepa.reproducibility import seed_everything


def test_worker_seeds_match_dmawm_spacing():
    assert [_worker_seed(7, index) for index in range(4)] == [7, 17, 27, 37]


def test_global_host_rngs_repeat():
    seed_everything(23)
    first = (random.random(), np.random.random())
    seed_everything(23)
    assert first == (random.random(), np.random.random())


def test_launcher_sets_hash_seed_before_child_process(tmp_path, monkeypatch):
    monkeypatch.setenv("SC2PATH", str(tmp_path / "StarCraftII"))
    environment = runtime_environment(
        task="smac_3m",
        infrastructure_root=tmp_path,
        artifact_dir=tmp_path,
        seed=31,
    )
    assert environment["PYTHONHASHSEED"] == "31"
    assert os.environ.get("PYTHONHASHSEED") != "31" or environment is not os.environ


def test_maintained_profile_uses_synchronous_snapshot_sampling():
    config = _resolve_config_profiles(
        _load_configs(), ("smac_vector", "ma_jepa")
    )
    assert config.run.replay_stream_mode == "snapshot_staggered"
    assert config.run.replay_startup_behavior_min_starts == 4
    assert config.run.replay_trace_batches == 32
    assert config.run.isolate_report_rng is True
