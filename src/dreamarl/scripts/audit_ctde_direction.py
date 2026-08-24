"""Create equal-KL actor perturbations along and against the imagined gradient."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from .audit_ctde_signal import (
    _agent,
    _discover_checkpoints,
    _load_checkpoint,
    _load_config,
    _load_replay_windows,
    _real_evaluations,
)


def _prepare_replay(agent, batch_size, batch):
    local_carry = agent._local_initial(batch_size * agent.team.size)
    local_data = agent.team.local_sequence_data(batch)
    return agent._apply_replay_context(local_carry, local_data)[:3]


def _gradient_function(agent, batch_size):
    import ninjax as nj

    def gradient(batch):
        local_carry, obs, prevact = _prepare_replay(agent, batch_size, batch)

        def objective():
            loss, (_, _, _, metrics) = agent.loss(
                local_carry,
                obs,
                prevact,
                training=False,
                reference_data=None,
            )
            return loss, {
                key: metrics[key]
                for key in (
                    "reward_return_raw",
                    "ret_root_raw",
                    "critic_rollout_raw",
                    "imagined_action/attack_fraction",
                    "imagined_action/move_fraction",
                )
                if key in metrics
            }

        loss, _, gradients, metrics = nj.grad(objective, agent.pol, has_aux=True)()
        return loss, gradients, metrics

    return gradient


def _logit_function(agent, batch_size):
    def logits(batch):
        local_carry, obs, prevact = _prepare_replay(agent, batch_size, batch)
        _, _, _, repfeat, _, _, _ = agent._world_model_terms(
            local_carry,
            obs,
            prevact,
            training=False,
        )
        distribution = agent.policy_distribution(
            agent.feat2tensor(repfeat),
            2,
            action_mask=obs["action_mask"],
        )
        decision = obs["action_mask"].astype(np.int32).sum(-1) > 1
        return distribution[agent.action_mask_key].logits, decision

    return logits


def _forward_kl(reference, current, decision) -> float:
    reference = np.asarray(reference, np.float64)
    current = np.asarray(current, np.float64)
    reference -= np.max(reference, axis=-1, keepdims=True)
    current -= np.max(current, axis=-1, keepdims=True)
    reference -= np.log(np.exp(reference).sum(axis=-1, keepdims=True))
    current -= np.log(np.exp(current).sum(axis=-1, keepdims=True))
    probability = np.exp(reference)
    divergence = (probability * (reference - current)).sum(axis=-1)
    decision = np.asarray(decision, bool)
    return float(divergence[decision].mean()) if decision.any() else 0.0


def _gradient_rms(gradient: dict[str, np.ndarray]) -> float:
    squares = sum(
        float(np.square(value.astype(np.float64)).sum()) for value in gradient.values()
    )
    count = sum(value.size for value in gradient.values())
    return float(np.sqrt(squares / max(count, 1)))


def _candidate(
    model: dict[str, np.ndarray],
    gradient: dict[str, np.ndarray],
    *,
    scale: float,
    sign: float,
    rms: float,
) -> dict[str, np.ndarray]:
    result = dict(model)
    for key, value in gradient.items():
        update = sign * scale * value.astype(np.float32) / max(rms, 1e-20)
        result[key] = (np.asarray(model[key], np.float32) + update).astype(
            np.asarray(model[key]).dtype
        )
    return result


def _choose_scale(
    model,
    gradient,
    *,
    sign,
    rms,
    target,
    replay,
    compiled_logits,
    base_logits,
    decision,
    jax,
    model_device,
):
    trials = []
    for scale in np.logspace(-6, -1, 16):
        candidate = _candidate(model, gradient, scale=float(scale), sign=sign, rms=rms)
        candidate_state = dict(model_device)
        candidate_state.update(
            jax.device_put(
                {
                    key: value
                    for key, value in candidate.items()
                    if key.startswith("pol/")
                }
            )
        )
        _, (candidate_logits, _) = compiled_logits(candidate_state, replay)
        candidate_logits = jax.device_get(candidate_logits)
        divergence = _forward_kl(base_logits, candidate_logits, decision)
        trials.append((abs(divergence - target), float(scale), divergence))
    _, scale, divergence = min(trials)
    return scale, divergence


def _selected_checkpoint(run: Path) -> tuple[int, Path]:
    checkpoints = dict(_discover_checkpoints(run))
    evaluations = _real_evaluations(run)
    available = [step for step in evaluations if step in checkpoints]
    if not available:
        raise ValueError("no checkpoint has a matching deterministic evaluation")
    step = max(
        available,
        key=lambda item: (
            evaluations[item].get("eval/win_rate", float("-inf")),
            evaluations[item].get("eval/legacy_return_mean", float("-inf")),
            -item,
        ),
    )
    return step, checkpoints[step]


def _save_variant(source: Path, destination: Path, actor) -> None:
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    payload["params"].update(actor)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "agent.pkl").open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (destination / "done").write_text("", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-kl", type=float, default=0.005)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=70_000)
    args = parser.parse_args(argv)
    if args.target_kl <= 0 or min(args.batch, args.starts, args.seeds) < 1:
        raise ValueError("target KL, batch, starts, and seeds must be positive")

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import ninjax as nj

    run = args.run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    step, checkpoint = _selected_checkpoint(run)
    config = _load_config(run)
    length = int(config.replay_context) + int(config.batch_length)
    replay = _load_replay_windows(run, batch=args.batch, length=length)
    agent = _agent(config, replay, args.starts)
    replay = jax.device_put(replay)
    model = _load_checkpoint(checkpoint)
    model_device = jax.device_put(model)

    pure_gradient = nj.pure(_gradient_function(agent, args.batch))
    compiled_gradient = jax.jit(
        lambda state, data, seed: pure_gradient(state, data, seed=seed)
    )
    gradients = []
    metrics = []
    for offset in range(args.seeds):
        seed = jax.device_put(np.asarray(args.seed + offset, np.int32))
        _, (_, gradient, sample_metrics) = compiled_gradient(model_device, replay, seed)
        gradients.append(jax.device_get(gradient))
        metrics.append(jax.device_get(sample_metrics))
    gradient = {
        key: np.mean([np.asarray(sample[key], np.float32) for sample in gradients], 0)
        for key in gradients[0]
    }
    rms = _gradient_rms(gradient)

    pure_logits = nj.pure(_logit_function(agent, args.batch))
    compiled_logits = jax.jit(lambda state, data: pure_logits(state, data, seed=0))
    _, (base_logits, decision) = compiled_logits(model_device, replay)
    base_logits, decision = jax.device_get((base_logits, decision))

    descent_scale, descent_kl = _choose_scale(
        model,
        gradient,
        sign=-1.0,
        rms=rms,
        target=args.target_kl,
        replay=replay,
        compiled_logits=compiled_logits,
        base_logits=base_logits,
        decision=decision,
        jax=jax,
        model_device=model_device,
    )
    ascent_scale, ascent_kl = _choose_scale(
        model,
        gradient,
        sign=1.0,
        rms=rms,
        target=args.target_kl,
        replay=replay,
        compiled_logits=compiled_logits,
        base_logits=base_logits,
        decision=decision,
        jax=jax,
        model_device=model_device,
    )
    descent = _candidate(model, gradient, scale=descent_scale, sign=-1.0, rms=rms)
    ascent = _candidate(model, gradient, scale=ascent_scale, sign=1.0, rms=rms)
    output.mkdir(parents=True, exist_ok=True)
    _save_variant(
        checkpoint,
        output / "imagined_improvement",
        {key: value for key, value in descent.items() if key.startswith("pol/")},
    )
    _save_variant(
        checkpoint,
        output / "imagined_degradation",
        {key: value for key, value in ascent.items() if key.startswith("pol/")},
    )
    result = {
        "run": str(run),
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "target_kl": args.target_kl,
        "gradient_rms": rms,
        "gradient_samples": args.seeds,
        "imagined_improvement": {"scale": descent_scale, "replay_kl": descent_kl},
        "imagined_degradation": {"scale": ascent_scale, "replay_kl": ascent_kl},
        "base_imagination": {
            key: float(np.mean([np.asarray(item[key]) for item in metrics]))
            for key in metrics[0]
        },
    }
    (output / "directional_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
