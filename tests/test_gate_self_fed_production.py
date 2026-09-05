"""Pure preparation/guard tests; no GPU, environment, or checkpoint learning."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from majepa.main import _load_configs, _resolve_config_profiles

SCRIPT = Path(__file__).parents[1] / "scripts" / "gate_self_fed_production.py"
spec = importlib.util.spec_from_file_location("production_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_gate_preserves_saved_production_shape_and_ppo_protocol(tmp_path):
    saved = _resolve_config_profiles(_load_configs(), ("smac_vector", "ma_jepa"))
    saved = saved.update({"task": "smac_2s3z", "agent.num_agents": 5})
    prepared, overrides = gate.gate_config(saved, tmp_path)
    for name, value in saved.flat.items():
        if name not in overrides:
            assert prepared.flat[name] == value
    assert tuple(prepared.agent.marl.ctde.self_fed.horizons) == (2, 4, 5)
    assert prepared.agent.marl.ctde.self_fed.anchors == 8
    assert prepared.agent.marl.ctde.self_fed.consumer_kl_scale == 0
    disabled, _ = gate.gate_config(saved, tmp_path, enabled=False)
    assert not disabled.agent.marl.ctde.self_fed.enabled


def make_inputs(tmp_path):
    run = tmp_path / "frozen"
    checkpoint = run / "ckpt" / "completed"
    checkpoint.mkdir(parents=True)
    (run / "config.yaml").write_text("task: smac_2s3z\n")
    (checkpoint / "agent.pkl").write_bytes(b"untouched checkpoint bytes")
    (checkpoint / "done").touch()
    (checkpoint / "step.pkl").write_bytes(b"step bytes")
    (run / "replay").mkdir()
    (run / "replay" / "chunk.npz").write_bytes(b"replay bytes")
    return run, checkpoint


def test_inputs_are_copied_and_source_remains_independent(tmp_path):
    run, checkpoint = make_inputs(tmp_path)
    output = tmp_path / "gate"
    copied, manifest = gate.copy_inputs(run, checkpoint, output)
    assert len(manifest) == 5
    (copied / "agent.pkl").write_bytes(b"changed disposable copy")
    assert (checkpoint / "agent.pkl").read_bytes() == b"untouched checkpoint bytes"
    with pytest.raises(FileExistsError):
        gate.copy_inputs(run, checkpoint, output)
    with pytest.raises(ValueError, match="outside"):
        gate.copy_inputs(run, checkpoint, run / "unsafe")


def test_gate_requires_finite_metrics_actual_updates_and_valid_h5():
    metrics = {
        "opt/local_world/skipped": 0,
        "opt/joint_world/skipped": 0,
        **{f"ctde/self_fed_train_h{h}/team_count": 2 for h in (2, 4, 5)},
    }
    gate.require_update_metrics(metrics, enabled=True)
    metrics["ctde/self_fed_train_h5/team_count"] = 0
    with pytest.raises(ValueError, match="H5"):
        gate.require_update_metrics(metrics, enabled=True)
    gate.require_update_metrics(metrics, enabled=False)
    with pytest.raises(ValueError, match="Nonfinite"):
        gate.scalar_metrics({"reward": np.nan})
    metrics["opt/joint_world/skipped"] = 1
    with pytest.raises(ValueError, match="skipped"):
        gate.require_update_metrics(metrics, enabled=False)


def test_memory_sampler_ignores_other_processes():
    text = "123, 100\n456, 10000\n123, 200\n"
    assert gate.parse_process_memory(text, 123) == 300
    assert gate.parse_process_memory(text, 789) == 0
