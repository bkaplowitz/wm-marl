"""Audit whether CTDE imagination ranks real policies correctly.

The audit freezes each saved world model and critic in turn, swaps in actors
from every saved checkpoint, and evaluates all combinations on the same replay
roots and random seeds. It then compares imagined reward, lambda-return, and
critic rankings with the deterministic environment evaluations recorded during
training. This separates a predictive-signal failure from a critic or actor
optimization failure without training another agent.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


GLOBAL_REPLAY_KEYS = frozenset(
    {"is_first", "is_last", "is_terminal", "consec", "stepid"}
)
REAL_METRICS = (
    "eval/win_rate",
    "eval/legacy_return_mean",
    "eval/corrected_return_mean",
    "eval/enemy_kills_mean",
    "eval/ally_survivors_mean",
    "eval/timeout_rate",
    "eval/action_attack_fraction",
    "eval/action_move_fraction",
)
SIGNAL_METRICS = (
    "reward_return_raw",
    "ret_root_raw",
    "critic_root_raw",
    "critic_rollout_raw",
    "ret_raw",
    "val_raw",
    "adv",
    "critic/value_explained_variance",
    "critic/value_rmse",
    "imagined_action/noop_fraction",
    "imagined_action/stop_fraction",
    "imagined_action/move_fraction",
    "imagined_action/attack_fraction",
    "ctde/embedding_cosine",
    "ctde/interface_smooth_l1",
    "ctde/posterior_kl",
    "ctde/reward_loss",
    "ctde/action_mask_loss",
    "ctde/alive_loss",
)


def _checkpoint_step(path: Path) -> int:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return int(payload["counters"]["actions"])


def _discover_checkpoints(run: Path) -> list[tuple[int, Path]]:
    found = []
    for path in sorted((run / "run" / "ckpt").glob("*/agent.pkl")):
        found.append((_checkpoint_step(path), path))
    unique = {}
    for step, path in found:
        unique[step] = path
    return sorted(unique.items())


def _load_checkpoint(path: Path, *, actor_only: bool = False) -> dict[str, Any]:
    with path.open("rb") as handle:
        params = pickle.load(handle)["params"]
    if actor_only:
        return {key: value for key, value in params.items() if key.startswith("pol/")}
    ignored = ("opt/", "actor_trust/", "trust_pol/", "trust_pol_count/")
    return {key: value for key, value in params.items() if not key.startswith(ignored)}


def _load_config(run: Path):
    import elements
    import ruamel.yaml as yaml

    raw = yaml.YAML(typ="safe").load((run / "run" / "config.yaml").read_text())
    config = elements.Config(raw)
    if str(config.agent.marl.stage) != "ctde":
        raise ValueError("the signal audit requires a CTDE checkpoint")
    return config


def _load_replay_windows(
    run: Path, *, batch: int, length: int
) -> dict[str, np.ndarray]:
    chunks = sorted((run / "run" / "replay").glob("*.npz"))
    if not chunks:
        raise FileNotFoundError(f"no replay chunks under {run / 'run' / 'replay'}")
    loaded = []
    for path in chunks:
        with np.load(path) as chunk:
            loaded.append({key: np.asarray(value) for key, value in chunk.items()})
    common = set.intersection(*(set(chunk) for chunk in loaded))
    sequence = {
        key: np.concatenate([chunk[key] for chunk in loaded], axis=0)
        for key in sorted(common)
    }
    total = min(value.shape[0] for value in sequence.values())
    if total < length:
        raise ValueError(f"replay has {total} rows but audit requires {length}")
    starts = np.linspace(0, total - length, batch, dtype=np.int64)
    result = {
        key: np.stack([value[start : start + length] for start in starts], axis=0)
        for key, value in sequence.items()
    }
    result["consec"] = np.zeros((batch, length), np.int32)
    return result


def _spaces(batch: dict[str, np.ndarray]):
    import elements

    action = batch["action"]
    action_count = int(batch["action_mask"].shape[-1])
    observations = {}
    for key in (
        "observation",
        "reward",
        "agent_present",
        "agent_alive",
        "controllable_alive",
        "action_mask",
        "is_first",
        "is_last",
        "is_terminal",
    ):
        value = batch[key]
        observations[key] = elements.Space(value.dtype, value.shape[2:])
    actions = {
        "action": elements.Space(action.dtype, action.shape[2:], 0, action_count)
    }
    return observations, actions


def _agent(run_config, replay_batch, starts: int):
    import elements

    from dreamarl.marl.core import MARLCore

    observations, actions = _spaces(replay_batch)
    agent_config = elements.Config(
        **run_config.agent,
        logdir=str(run_config.logdir),
        seed=int(run_config.seed),
        jax=run_config.jax.update(precompile=False),
        batch_size=int(replay_batch["is_first"].shape[0]),
        batch_length=int(run_config.batch_length),
        replay_context=int(run_config.replay_context),
        report_length=int(run_config.report_length),
        replica=0,
        replicas=1,
    ).update(imag_last=int(starts))
    return MARLCore(observations, actions, agent_config).model


def _audit_function(agent, batch_size: int):
    def audit(batch):
        local_carry = agent._local_initial(batch_size * agent.team.size)
        local_data = agent.team.local_sequence_data(batch)
        local_carry, obs, prevact, _ = agent._apply_replay_context(
            local_carry, local_data
        )
        _, (_, _, _, metrics) = agent.loss(
            local_carry,
            obs,
            prevact,
            training=False,
            reference_data=None,
        )
        return {key: metrics[key] for key in SIGNAL_METRICS if key in metrics}

    return audit


def _real_evaluations(run: Path) -> dict[int, dict[str, float]]:
    path = run / "run" / "metrics.jsonl"
    result = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "eval/episodes" not in row:
            continue
        step = int(row["step"])
        result[step] = {key: float(row[key]) for key in REAL_METRICS if key in row}
    return result


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1)
        index = end
    return ranks


def _spearman(left, right) -> float | None:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return None
    left = _rank(left[finite])
    right = _rank(right[finite])
    if left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for model_step in sorted({row["model_step"] for row in rows}):
        selected = [row for row in rows if row["model_step"] == model_step]
        for predicted in (
            "reward_return_raw",
            "ret_root_raw",
            "critic_rollout_raw",
        ):
            for actual in (
                "eval/win_rate",
                "eval/legacy_return_mean",
                "eval/enemy_kills_mean",
            ):
                correlation = _spearman(
                    [row.get(predicted, np.nan) for row in selected],
                    [row.get(actual, np.nan) for row in selected],
                )
                result.append(
                    {
                        "model_step": model_step,
                        "predicted": predicted,
                        "actual": actual,
                        "spearman": correlation,
                        "pairs": int(
                            sum(
                                bool(np.isfinite(row.get(predicted, np.nan)))
                                and bool(np.isfinite(row.get(actual, np.nan)))
                                for row in selected
                            )
                        ),
                    }
                )
    return result


def _diagnosis(correlations: list[dict[str, Any]]) -> str:
    if not correlations:
        return "insufficient checkpoint/evaluation pairs"
    latest = max(row["model_step"] for row in correlations)

    def best(signal):
        values = [
            row["spearman"]
            for row in correlations
            if row["model_step"] == latest
            and row["predicted"] == signal
            and row["spearman"] is not None
        ]
        return max(values) if values else None

    reward = best("reward_return_raw")
    returns = best("ret_root_raw")
    critic = best("critic_rollout_raw")
    if reward is not None and reward <= 0:
        return "world-model/reward signal misranks real policies"
    if reward is not None and reward > 0 and critic is not None and critic <= 0:
        return "reward signal ranks policies, but the central critic does not"
    if reward is not None and reward > 0 and returns is not None and returns > 0:
        return (
            "imagined objective ranks policies; actor optimization/support is primary"
        )
    return "ranking evidence is ambiguous; run the registered directional test"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=60_000)
    args = parser.parse_args(argv)
    if min(args.batch, args.starts, args.seeds) < 1:
        raise ValueError("batch, starts, and seeds must be positive")

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import ninjax as nj

    run = args.run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    checkpoints = _discover_checkpoints(run)
    if len(checkpoints) < 2:
        raise ValueError("the cross-check requires at least two saved checkpoints")
    config = _load_config(run)
    replay_length = int(config.replay_context) + int(config.batch_length)
    replay = _load_replay_windows(run, batch=args.batch, length=replay_length)
    agent = _agent(config, replay, args.starts)
    pure = nj.pure(_audit_function(agent, args.batch))
    compiled = jax.jit(lambda state, data, seed: pure(state, data, seed=seed))
    replay = jax.device_put(replay)
    real = _real_evaluations(run)

    actors = {
        step: _load_checkpoint(path, actor_only=True) for step, path in checkpoints
    }
    rows = []
    for model_step, path in checkpoints:
        model = _load_checkpoint(path)
        for actor_step, actor in actors.items():
            state = dict(model)
            state.update(actor)
            state = jax.device_put(state)
            samples = []
            for offset in range(args.seeds):
                seed = jax.device_put(np.asarray(args.seed + offset, np.int32))
                _, metrics = compiled(state, replay, seed)
                samples.append(
                    {
                        key: float(np.asarray(jax.device_get(value)))
                        for key, value in metrics.items()
                    }
                )
            row = {
                "model_step": model_step,
                "actor_step": actor_step,
                **{
                    key: float(np.mean([sample[key] for sample in samples]))
                    for key in samples[0]
                },
                **real.get(actor_step, {}),
            }
            rows.append(row)

    correlations = _correlations(rows)
    result = {
        "run": str(run),
        "checkpoint_steps": [step for step, _ in checkpoints],
        "replay_batch": args.batch,
        "imagination_starts": args.starts,
        "imagination_horizon": int(config.agent.imag_length),
        "random_seeds": args.seeds,
        "rows": rows,
        "correlations": correlations,
        "diagnosis": _diagnosis(correlations),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "signal_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output / "cross_checkpoint_matrix.csv", rows)
    _write_csv(output / "rank_correlations.csv", correlations)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
