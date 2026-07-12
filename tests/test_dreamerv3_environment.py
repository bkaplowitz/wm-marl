from __future__ import annotations

from world_marl.baselines.dreamerv3.config import default_upstream_root
from world_marl.baselines.dreamerv3.environment import resolved_requirements


def test_cpu_requirements_only_adapt_platform_specific_jax_packages():
  requirements = resolved_requirements(default_upstream_root(), accelerator="cpu")
  assert "jax==0.4.33" in requirements
  assert not any("cuda12" in requirement for requirement in requirements)
  assert not any(
    requirement.startswith("nvidia-cuda-") for requirement in requirements
  )
  assert "dm_control" in requirements
  assert "wandb" in requirements


def test_cuda_requirements_preserve_official_jax_pin():
  requirements = resolved_requirements(
    default_upstream_root(), accelerator="cuda12"
  )
  assert "jax[cuda12]==0.4.33" in requirements
  assert "nvidia-cuda-nvcc-cu12<=12.2" in requirements
