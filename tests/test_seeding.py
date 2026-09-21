from majepa.main import _load_configs, _resolve_config_profiles, _worker_seed


def test_worker_seeds_match_pinned_mapping():
    assert [_worker_seed(7, index) for index in range(4)] == [
        hash((7, index)) % (2**32 - 1) for index in range(4)
    ]


def test_maintained_profile_uses_synchronous_snapshot_sampling():
    config = _resolve_config_profiles(_load_configs(), ("baseline",))
    assert config.run.replay_stream_mode == "snapshot_staggered"
    assert config.run.replay_startup_behavior_min_starts == 4
    assert config.run.replay_trace_batches == 16
    assert config.run.isolate_report_rng is True
