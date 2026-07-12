from __future__ import annotations

import os

from world_marl.determinism import (
    DETERMINISTIC_XLA_FLAGS,
    configure_deterministic_environment,
)


def test_deterministic_environment_is_idempotent(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--existing-flag=true")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("TF_CUDNN_DETERMINISTIC", raising=False)
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)

    configure_deterministic_environment()
    configure_deterministic_environment()

    flags = os.environ["XLA_FLAGS"].split()
    assert flags[0] == "--existing-flag=true"
    for expected in DETERMINISTIC_XLA_FLAGS:
        assert flags.count(expected) == 1
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["TF_CUDNN_DETERMINISTIC"] == "1"
    assert os.environ["NVIDIA_TF32_OVERRIDE"] == "0"
