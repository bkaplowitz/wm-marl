"""Run the seven frozen DreaMARL multi-agent representation interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.dreamarl.representation_diagnostics import (
    AdapterConfig,
    Control,
    INTERVENTIONS,
    Intervention,
    loss_and_metrics,
    train_adapter,
)


REQUIRED_KEYS = frozenset(
    {
        "pair",
        "stoch",
        "post_logit",
        "prior_logit",
        "pred_token",
        "target_token",
        "reset",
    }
)
OPENLOOP_KEYS = frozenset(
    {
        "openloop_pair",
        "openloop_stoch",
        "openloop_prior_logit",
        "openloop_pred_token",
        "valid",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    return parser


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_KEYS - set(archive.files)
        if missing:
            raise ValueError(f"dataset is missing keys: {sorted(missing)}")
        tensors = {key: archive[key] for key in REQUIRED_KEYS}
        present_openloop = OPENLOOP_KEYS & set(archive.files)
        if present_openloop and present_openloop != OPENLOOP_KEYS:
            missing_openloop = OPENLOOP_KEYS - present_openloop
            raise ValueError(
                f"dataset has incomplete open-loop tensors: {sorted(missing_openloop)}"
            )
        tensors.update({key: archive[key] for key in present_openloop})
    leading = {key: value.shape[:3] for key, value in tensors.items()}
    if len(set(leading.values())) != 1:
        raise ValueError(f"unaligned [batch,time,agent] axes: {leading}")
    if tensors["pair"].shape[-1] <= np.prod(tensors["stoch"].shape[-2:]):
        raise ValueError("pair tensor does not contain an encoded action suffix")
    return tensors


def _objective_tensors(
    tensors: dict[str, jax.Array], intervention: Intervention
) -> dict[str, jax.Array]:
    if not OPENLOOP_KEYS <= tensors.keys() or intervention is Intervention.PAIRED_PRIOR:
        return {key: tensors[key] for key in REQUIRED_KEYS if key in tensors}
    result = {key: tensors[key] for key in REQUIRED_KEYS if key in tensors}
    result.update(
        pair=tensors["openloop_pair"],
        stoch=tensors["openloop_stoch"],
        prior_logit=tensors["openloop_prior_logit"],
        pred_token=tensors["openloop_pred_token"],
        valid=tensors["valid"],
    )
    return result


def _horizon_metrics(
    params,
    tensors: dict[str, jax.Array],
    intervention: Intervention,
    control: Control,
    seed: int,
) -> dict[str, dict[str, float]]:
    if intervention is Intervention.PAIRED_PRIOR or "valid" not in tensors:
        return {}
    length = tensors["reset"].shape[1]
    horizons = [value for value in (1, 2, 4, 8, 16, 32) if value <= length]
    result = {}
    for index, horizon in enumerate(horizons):
        step_tensors = jax.tree.map(
            lambda value: value[:, horizon - 1 : horizon], tensors
        )
        _, metrics = loss_and_metrics(
            params,
            step_tensors,
            intervention,
            control=control,
            key=jax.random.key(seed + index),
        )
        result[f"h{horizon}"] = _json_metrics(metrics)
    return result


def _split(
    tensors: dict[str, np.ndarray], fraction: float, seed: int
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    if not 0 < fraction < 1:
        raise ValueError("validation-fraction must lie strictly between 0 and 1")
    trajectories = next(iter(tensors.values())).shape[0]
    if trajectories < 4:
        raise ValueError("at least four complete trajectories are required")
    order = np.random.default_rng(seed).permutation(trajectories)
    validation_size = max(1, round(trajectories * fraction))
    validation_ids = order[:validation_size]
    training_ids = order[validation_size:]
    def convert(ids):
        return {key: jnp.asarray(value[ids]) for key, value in tensors.items()}

    return convert(training_ids), convert(validation_ids)


def _json_metrics(metrics: dict[str, jax.Array]) -> dict[str, float]:
    return {key: float(np.asarray(value)) for key, value in metrics.items()}


def run(args: argparse.Namespace) -> dict[str, object]:
    tensors = _load(args.dataset)
    train, validation = _split(tensors, args.validation_fraction, args.seed)
    config = AdapterConfig(
        hidden=args.hidden,
        learning_rate=args.learning_rate,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    results = {}
    for intervention in INTERVENTIONS:
        train_objective = _objective_tensors(train, intervention)
        validation_objective = _objective_tensors(validation, intervention)
        params, history = train_adapter(train_objective, intervention, config)
        controls = [Control.CORRECT]
        if intervention not in {
            Intervention.BASELINE,
            Intervention.PAIRED_SHUFFLED_PREDICTOR,
        }:
            controls.extend(
                [
                    Control.SHUFFLE_AGENTS,
                    Control.SHUFFLE_ENVIRONMENTS,
                    Control.NULL,
                ]
            )
        evaluations = {}
        for index, control in enumerate(controls):
            _, metrics = loss_and_metrics(
                params,
                validation_objective,
                intervention,
                control=control,
                key=jax.random.key(args.seed + index + 1),
            )
            evaluations[control.value] = {
                **_json_metrics(metrics),
                "horizons": _horizon_metrics(
                    params,
                    validation_objective,
                    intervention,
                    control,
                    args.seed + 100 * (index + 1),
                ),
            }
        parameter_count = 0 if params is None else sum(
            int(value.size) for value in jax.tree.leaves(params)
        )
        results[intervention.value] = {
            "source": intervention.value.split("_")[0],
            "adapter_parameters": parameter_count,
            "training_history": history,
            "validation": evaluations,
        }
    output = {
        "contract": "frozen_checkpoint_frozen_replay_v1",
        "dataset": str(args.dataset.resolve()),
        "trajectory_split": {
            "train": next(iter(train.values())).shape[0],
            "validation": next(iter(validation.values())).shape[0],
        },
        "adapter": {
            "hidden": config.hidden,
            "learning_rate": config.learning_rate,
            "steps": config.steps,
            "batch_size": config.batch_size,
            "seed": config.seed,
        },
        "interventions": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run(args)
    for name, result in output["interventions"].items():
        correct = result["validation"]["correct"]
        print(
            f"{name:28s} cosine={correct['cosine']:.5f} "
            f"raw_kl={correct['raw_kl']:.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
