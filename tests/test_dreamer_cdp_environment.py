from dreamarl.baselines.dreamer_cdp.config import default_upstream_root
from dreamarl.baselines.dreamer_cdp.environment import resolved_requirements


def test_cpu_environment_removes_only_cuda_requirements():
    requirements = resolved_requirements(default_upstream_root(), accelerator="cpu")
    assert "jax==0.4.33" in requirements
    assert not any("cuda12" in requirement for requirement in requirements)
    assert "dm_control" in requirements
    assert "wandb[media]" in requirements


def test_cuda_environment_preserves_official_jax_pin():
    requirements = resolved_requirements(default_upstream_root(), accelerator="cuda12")
    assert "jax[cuda12]==0.4.33" in requirements
    assert "nvidia-cuda-nvcc-cu12<=12.2" in requirements
