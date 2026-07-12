"""Process-level determinism settings that must precede JAX initialization."""

from __future__ import annotations

import os


DETERMINISTIC_XLA_FLAGS = (
    "--xla_gpu_deterministic_ops=true",
    "--xla_gpu_autotune_level=0",
    "--xla_gpu_enable_triton_gemm=false",
)


def configure_deterministic_environment() -> None:
    """Set deterministic accelerator environment variables before JAX starts."""

    configured_flags = os.environ.get("XLA_FLAGS", "").strip().split()
    for deterministic_flag in DETERMINISTIC_XLA_FLAGS:
        if deterministic_flag not in configured_flags:
            configured_flags.append(deterministic_flag)
    os.environ["XLA_FLAGS"] = " ".join(configured_flags)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
