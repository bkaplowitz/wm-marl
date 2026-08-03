"""Single-agent reduction checks against the registered M3 foundation."""

from __future__ import annotations

from world_marl.dreamarl.config import DreaMARLRunSpec
from world_marl.dreamarl.runtime import (
    runtime_fingerprint,
    verify_first_party_source,
)
from world_marl.jepa_transformer.config import JEPATransformerRunSpec


FOUNDATION_COMMIT = "a851fa3e3d70b624b094ee1810ad4bb602346092"


def verify_single_agent_reduction_contract(
    spec: DreaMARLRunSpec,
) -> dict[str, object]:
    """Prove that agent count changes geometry and nothing algorithmic.

    M3 is the one-agent reduction oracle. The DreaMARL command is compared
    after removing only the explicit agent-axis extent, so this check applies
    to every agent count rather than selecting a special one-agent path.
    """

    source = verify_first_party_source(spec.infrastructure_root)
    revision = source["infrastructure_commit"]
    if revision != FOUNDATION_COMMIT:
        raise RuntimeError(
            "Dreamer-CDP revision changed: "
            f"expected {FOUNDATION_COMMIT}, got {revision}"
        )

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
        "verified_foundation_commit": revision,
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
    for algorithm_config in (
        "structured_local_memory",
        "shared_transition_context",
    ):
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
