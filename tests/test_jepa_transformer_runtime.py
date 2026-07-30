from __future__ import annotations

from pathlib import Path

from world_marl.jepa_transformer.runtime import (
    M3_DYNAMICS_DEFAULT,
    M3_PROFILE,
    prepare_runtime,
    runtime_fingerprint,
)


def test_runtime_overlay_is_reproducible_and_keeps_official_source_clean(tmp_path):
    runtime = prepare_runtime(tmp_path / "runtime")
    assert prepare_runtime(runtime) == runtime
    marker = (runtime / ".jepa-transformer-runtime").read_text()
    assert marker.splitlines()[1] == runtime_fingerprint()

    agent = (runtime / "dreamerv3" / "agent.py").read_text()
    assert "from . import m3_rssm, rssm" in agent
    assert "'jepa_transformer': m3_rssm.TransformerRSSM" in agent
    assert "for key in self.dyn.entry_space" in agent
    assert (runtime / "dreamerv3" / "m3_rssm.py").is_file()
    assert (runtime / "dreamerv3" / "configs.yaml").read_text().endswith(
        M3_PROFILE
    )


def test_m3_overlay_encodes_the_registered_temporal_and_replay_contract():
    source = (
        Path(__file__).parents[1]
        / "src"
        / "world_marl"
        / "jepa_transformer"
        / "upstream"
        / "m3_rssm.py"
    ).read_text()
    assert "pair = jnp.concatenate([previous_stoch, action], -1)" in source
    assert "nn.where(reset, start, projected)" in source
    assert "cache['keys'][:, index, 1:]" in source
    assert "policy(sg(carry))" in source
    assert "sg(slow_tokens)" in source
    assert "replay_context: 64" in M3_PROFILE
    assert "context: 64" in M3_DYNAMICS_DEFAULT
