"""Parity checks for DreaMARL's mechanically imported M3 base."""

from __future__ import annotations

import hashlib

from world_marl.baselines.dreamer_cdp.config import default_upstream_root
from world_marl.baselines.dreamer_cdp.launcher import verify_upstream
from world_marl.dreamarl.config import DreaMARLRunSpec
from world_marl.dreamarl.runtime import (
    algorithm_root,
    runtime_fingerprint,
    verify_first_party_source,
)
from world_marl.jepa_transformer.config import JEPATransformerRunSpec


ORACLE_COMMIT = "a851fa3e3d70b624b094ee1810ad4bb602346092"
ORACLE_HASHES = {
    "__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "agent.py": "a0ae3fbd50e76dc4c649e3d092711229b2205f4d0fa9986e77e26b3c1ce92ce2",
    "rssm.py": "76e4c87005fc997299470723adb392e08a59c8c6f08ceaae3db913f045001946",
    "m3_rssm.py": "daf3fbe973fb277d11527ca1f2090ff8e5368f7e604a3fed3ef8c3126fab9df7",
    "configs.yaml": "9a2a10726e604c314baffcd497e2d1eced72c328339d8e0348f78cebdf32a126",
    "main.py": "37870982ac9f02ffbffa4ebb142452418dffd01a1ec4d6bb4766783df2e4fd90",
}


def verify_m3_reduction_contract(spec: DreaMARLRunSpec) -> dict[str, object]:
    """Prove that agent count changes geometry and nothing algorithmic.

    M3 is the one-agent reduction oracle. The DreaMARL command is compared
    after removing only the explicit agent-axis extent, so this check applies
    to every agent count rather than selecting a special one-agent path.
    """

    source = verify_first_party_source()
    revision = verify_upstream(default_upstream_root())
    if revision != ORACLE_COMMIT:
        raise RuntimeError(
            f"Dreamer-CDP revision changed: expected {ORACLE_COMMIT}, got {revision}"
        )
    actual = {
        name: hashlib.sha256((algorithm_root() / "m3" / name).read_bytes()).hexdigest()
        for name in ORACLE_HASHES
    }
    if actual != ORACLE_HASHES:
        raise RuntimeError("first-party M3 source changed during the parity gate")

    oracle = JEPATransformerRunSpec(
        experiment_dir=spec.experiment_dir,
        task=(spec.task if spec.task.startswith("dmc_") else "dmc_reacher_easy"),
        seed=spec.seed,
        train_steps=spec.train_steps,
        platform=spec.platform,
        runtime_root=spec.infrastructure_root,
        python=spec.python,
        save_every_seconds=spec.save_every_seconds,
        wandb_project=spec.wandb_project,
        wandb_entity=spec.wandb_entity,
        extra_args=(),
    )
    normalize_environment = not spec.task.startswith("dmc_")
    if _semantic_arguments(
        spec.command, normalize_environment=normalize_environment
    ) != _semantic_arguments(
        oracle.command, normalize_environment=normalize_environment
    ):
        raise RuntimeError("first-party DreaMARL training regime diverged from M3")
    overrides = spec.configs[2:]
    return {
        "verified_official_commit": revision,
        "verified_algorithm_hashes": actual,
        "verified_source_fingerprint": runtime_fingerprint(),
        "verified_infrastructure_fingerprint": source,
        "num_agents": spec.num_agents,
        "agent_axis_reduction": "identity reshape when num_agents=1",
        "agent_count_semantics": "tensor geometry only",
        "algorithm_overrides": overrides,
    }


def _semantic_arguments(
    command: list[str], *, normalize_environment: bool = False
) -> list[str]:
    """Normalize source/log paths and the shape-only axis declaration."""

    arguments = list(command[3:] if command[1] == "-m" else command[2:])
    if "--agent.num_agents" in arguments:
        index = arguments.index("--agent.num_agents")
        del arguments[index : index + 2]
    for algorithm_config in ("local_memory_sidecar", "local_memory_unified"):
        if algorithm_config in arguments:
            arguments.remove(algorithm_config)
    logdir = arguments.index("--logdir") + 1
    arguments[logdir] = "<logdir>"
    if normalize_environment:
        configs = arguments.index("--configs") + 1
        arguments[configs] = "<environment-config>"
        task = arguments.index("--task") + 1
        arguments[task] = "<task>"
    return arguments
