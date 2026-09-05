#!/usr/bin/env python3
"""Read-only paired audit of a frozen JEPA simulator on real replay roots.

Reconstructs raw complete episodes with current parameters. Factual, self-fed,
and oracle-observation paths share categorical randomness at each target time.
One root-live factual cohort is used at every horizon, including unit deaths.
No environment, optimizer, actor update, or checkpoint write is performed.
Outcome semantics must match the evaluated policy's simulator: corrected team
pooling is the default; original_slots preserves the older raw per-slot heads.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_control_sufficiency import load_episodes, load_frozen_params


def select_roots(episodes, horizon, count, seed):
    """Uniformly sample complete factual futures without crossing an episode."""
    candidates = []
    for index, episode in enumerate(episodes):
        data = episode.data
        alive = data.get("controllable_alive", data["action_mask"][..., 1:].any(-1))
        for root in range(len(data["observation"]) - horizon):
            if (
                alive[root].any()
                and not data["is_first"][root + 1 : root + horizon + 1].any()
            ):
                candidates.append((index, root))
    if len(candidates) < count:
        raise ValueError(f"Need {count} common-cohort roots, found {len(candidates)}")
    selected = np.random.default_rng(seed).choice(len(candidates), count, replace=False)
    return sorted(candidates[int(index)] for index in selected)


def outcome_predictions(reward, continuation, present, source_alive, next_alive, mode):
    """Apply only the selected deployed simulator's reward/continuation rules."""
    if mode == "original_slots":
        return reward, continuation
    if mode != "corrected_team":
        raise ValueError(f"Unknown outcome semantics: {mode}")
    from majepa.training.ctde import shared_team_outcomes

    return shared_team_outcomes(reward, continuation, present, source_alive, next_alive)


def make_auditor(
    config, observation_dim, action_count, horizon, outcome_semantics="corrected_team"
):
    """Build checkpoint-compatible local and joint modules, without a learner."""
    import elements
    import embodied.jax
    import embodied.jax.nets as nn
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from majepa.models.visual import Encoder
    from majepa.models.ctde import JointObservationJEPA
    from majepa.world_model.transformer import (
        ParallelTransformerDynamics,
        _where_active,
    )

    config = elements.Config(config)
    cfg = config.agent.marl.ctde
    encoder = Encoder(
        {"observation": elements.Space(np.float32, (observation_dim,))},
        **config.agent.enc.simple,
        name="enc",
    )
    dynamics = ParallelTransformerDynamics(
        {"action": elements.Space(np.int32, (), 0, action_count)},
        encoder.calculate_encoder_output_dim(),
        **config.agent.dyn.parallel_transformer,
        name="dyn",
    )
    joint = JointObservationJEPA(
        action_count,
        0,
        encoder.calculate_encoder_output_dim(),
        **cfg.joint,
        name="ctde_joint",
    )
    head = {key: cfg.head[key] for key in ("layers", "units", "act", "norm", "winit")}
    rew = embodied.jax.MLPHead(
        elements.Space(np.float32),
        output="symexp_twohot",
        bins=cfg.head.bins,
        outscale=cfg.head.outscale,
        **head,
        name="ctde_rew",
    )
    con = embodied.jax.MLPHead(
        elements.Space(bool, (), 0, 2),
        output="binary",
        outscale=1.0,
        **head,
        name="ctde_con",
    )
    mask = embodied.jax.MLPHead(
        elements.Space(bool, (action_count,), 0, 2),
        output="binary",
        outscale=0.0,
        **head,
        name="ctde_mask",
    )
    alive_head = embodied.jax.MLPHead(
        elements.Space(bool, (), 0, 2),
        output="binary",
        outscale=1.0,
        **head,
        name="ctde_alive",
    )
    discount = 1.0 - 1.0 / float(config.agent.horizon) if config.agent.contdisc else 1.0

    def feature(state):
        return jnp.concatenate(
            [
                nn.cast(state["deter"]),
                nn.cast(state["stoch"].reshape((*state["stoch"].shape[:-2], -1))),
            ],
            axis=-1,
        )

    def complete(cache, deter, tokens, key):
        logits = dynamics.posterior(nn.cast(tokens), nn.cast(deter))
        stoch = nn.cast(dynamics.distribution(logits).sample(seed=key))
        return nn.cast({"deter": deter, "stoch": stoch, **cache}), logits

    def signals(prediction, source_alive, present):
        hidden = prediction["hidden"]
        probability = alive_head(hidden, 2).prob(1)
        next_alive = source_alive & present & (probability >= 0.5)
        reward, continuation = outcome_predictions(
            rew(hidden, 2).pred(),
            con(hidden, 2).prob(1),
            present,
            source_alive,
            next_alive,
            outcome_semantics,
        )
        output = mask(hidden, 2)
        binary = output.output if hasattr(output, "output") else output
        legal = binary.logit >= 0.0
        noop = jnp.zeros_like(legal).at[..., 0].set(True)
        legal = jnp.where(legal.any(-1, keepdims=True), legal, noop)
        legal = jnp.where(next_alive[..., None], legal, noop)
        return {
            "reward": reward,
            "continuation": continuation,
            "mask": legal,
            "alive": next_alive,
            "alive_probability": source_alive.astype(jnp.float32) * probability,
        }

    def mixed_kl(target, predicted):
        target = jax.nn.softmax(target.astype(jnp.float32), axis=-1)
        predicted = jax.nn.softmax(predicted.astype(jnp.float32), axis=-1)
        uniform = dynamics.unimix / target.shape[-1]
        target = (1.0 - dynamics.unimix) * target + uniform
        predicted = (1.0 - dynamics.unimix) * predicted + uniform
        return (target * (jnp.log(target) - jnp.log(predicted))).sum(axis=(-1, -2))

    def audit(
        observation, actions, first, active, present, alive, reward, terminal, root
    ):
        # Local arrays are [A,T,...]; team arrays are [T,A,...].
        agents, length = observation.shape[:2]
        resets = jnp.broadcast_to(first[None], (agents, length))
        _, _, tokens = encoder({}, {"observation": observation}, resets, training=False)
        previous = jnp.concatenate(
            [jnp.zeros_like(actions[:, :1]), actions[:, :-1]], axis=1
        )
        keys = jax.random.split(nj.seed(), length)

        def observe_step(current, inputs):
            token, action, reset, valid, key = inputs
            pair = dynamics._temporal_pair(
                current["stoch"], {"action": action}, training=False, reset=reset
            )
            cache, deter = dynamics._temporal().step(
                dynamics._cache(current), pair, reset
            )
            updated, logits = complete(cache, deter, token, key)
            updated = _where_active(valid, updated, current)
            return updated, (updated, logits)

        _, (local_history, factual_logits) = nj.scan(
            observe_step,
            dynamics.initial(agents),
            (tokens.swapaxes(0, 1), previous.T, resets.T, active.T, keys),
            axis=0,
        )
        initial_joint = joint.initial(1, agents)
        _, teacher_prediction, joint_history = joint.sequence(
            initial_joint,
            feature(local_history)[None, :-1],
            actions.T[None, :-1],
            present[None, :-1],
            alive[None, :-1],
            first[None, :-1],
            training=False,
        )

        def window(value, offset=0):
            return jax.lax.dynamic_slice_in_dim(value, root + offset, horizon, axis=0)

        local_root = jax.tree.map(lambda value: value[root], local_history)
        joint_root = {
            key: jnp.where(
                root == 0, value, joint_history[key][:, jnp.maximum(root - 1, 0)]
            )
            for key, value in initial_joint.items()
        }
        source_present = present[root][None]
        source_alive = alive[root][None]
        true_tokens = window(tokens.swapaxes(0, 1), 1)
        true_logits = window(factual_logits, 1)
        teacher_prediction = jax.tree.map(
            lambda value: window(value[0]), teacher_prediction
        )
        teacher = signals(teacher_prediction, window(alive), window(present))
        teacher["embedding"] = teacher_prediction["embedding"]
        teacher["posterior"] = dynamics.posterior(
            nn.cast(teacher["embedding"]), nn.cast(window(local_history["deter"], 1))
        )

        def rollout(oracle):
            def step(current, inputs):
                local, central, current_alive, reset = current
                action, actual_token, actual_alive, key = inputs
                central, prediction = joint.step(
                    central,
                    feature(local)[None],
                    action[None],
                    source_present,
                    current_alive,
                    reset,
                    training=False,
                )
                output = signals(prediction, current_alive, source_present)
                cache, deter = dynamics.advance(
                    local,
                    {"action": action},
                    training=False,
                    active=source_present[0],
                )
                token = actual_token if oracle else prediction["embedding"][0]
                local, logits = complete(cache, deter, token, key)
                next_alive = actual_alive[None] if oracle else output["alive"]
                output = {name: value[0] for name, value in output.items()}
                output.update(embedding=prediction["embedding"][0], posterior=logits)
                return (local, central, next_alive, jnp.zeros_like(reset)), output

            initial = (
                local_root,
                joint_root,
                source_alive,
                jnp.reshape(first[root], (1,)),
            )
            _, output = nj.scan(
                step,
                initial,
                (window(actions.T), true_tokens, window(alive, 1), window(keys, 1)),
                axis=0,
            )
            return output

        paths = {
            "teacher_factual": teacher,
            "self_fed": rollout(False),
            "oracle_observation_alive": rollout(True),
        }
        actual_reward = window(reward, 1).astype(jnp.float32)
        actual_continuation = (~window(terminal, 1)).astype(jnp.float32)
        if config.agent.contdisc:
            actual_continuation *= discount
        actual_alive = window(alive, 1)
        actual_return = jnp.cumsum(
            discount ** jnp.arange(horizon)[:, None] * actual_reward, axis=0
        )
        result = {
            "cohort": alive[root] & present[root],
            "actual_reward": actual_reward,
            "actual_return": actual_return,
            "actual_alive": actual_alive,
            "actual_continuation": actual_continuation,
            "paths": {},
        }
        for name, output in paths.items():
            embedding = output.pop("embedding").astype(jnp.float32)
            target = true_tokens.astype(jnp.float32)
            cosine = (embedding * target).sum(-1) / jnp.maximum(
                jnp.linalg.norm(embedding, axis=-1) * jnp.linalg.norm(target, axis=-1),
                1e-8,
            )
            weights = jnp.concatenate(
                [
                    jnp.ones_like(output["continuation"][:1]),
                    jnp.cumprod(output["continuation"][:-1], axis=0),
                ],
                axis=0,
            )
            cumulative_return = jnp.cumsum(weights * output["reward"], axis=0)
            output.update(
                embedding_cosine=cosine,
                posterior_kl=mixed_kl(true_logits, output.pop("posterior")),
                cumulative_return=cumulative_return,
                reward_error=output["reward"] - actual_reward,
                cumulative_return_error=cumulative_return - actual_return,
                continuation_squared_error=jnp.square(
                    output["continuation"] - actual_continuation
                ),
                alive_squared_error=jnp.square(
                    output["alive_probability"] - actual_alive
                ),
            )
            result["paths"][name] = output
        return jax.lax.stop_gradient(result)

    return nj.pure(audit), nj.init(audit)


def episode_inputs(episode, padded_length):
    data = episode.data
    length, agents = data["observation"].shape[:2]

    def pad(value, fill=0):
        return np.pad(
            value,
            ((0, padded_length - length), *((0, 0) for _ in value.shape[1:])),
            constant_values=fill,
        )

    present = data.get("agent_present", np.ones((length, agents), bool))
    active = data.get("agent_alive", present) & present
    alive = (
        data.get("controllable_alive", data["action_mask"][..., 1:].any(-1)) & present
    )
    reward = data["reward"].astype(np.float32)
    if not np.allclose(reward, reward[:, :1]):
        raise ValueError("The simulator audit requires shared SMAC team rewards")
    return (
        pad(data["observation"]).transpose(1, 0, 2),
        pad(data["action"]).T,
        pad(data["is_first"], True),
        pad(active).T,
        pad(present),
        pad(alive),
        pad(reward),
        pad(data["is_terminal"], True)[:, None].repeat(agents, axis=1),
    )


def aggregate_records(records, horizons):
    cohort = np.stack([record["metrics"]["cohort"] for record in records]).astype(bool)
    weight = cohort.astype(np.float64)
    count = max(float(weight.sum()), 1.0)
    result = {}
    for name in records[0]["metrics"]["paths"]:
        arrays = {
            key: np.stack([record["metrics"]["paths"][name][key] for record in records])
            for key in records[0]["metrics"]["paths"][name]
        }
        result[name] = {}
        for horizon in horizons:
            index = horizon - 1

            def mean(value):
                return float((value * weight).sum() / count)

            mask_actual = np.stack([record["actual_mask"] for record in records])[
                :, index
            ]
            mask_predicted = arrays["mask"][:, index].astype(bool)
            positive = mask_actual * weight[..., None]
            negative = ~mask_actual * weight[..., None]
            result[name][f"h{horizon}"] = {
                "agent_roots": int(weight.sum()),
                "team_roots": len(records),
                "reward_rmse": mean(arrays["reward_error"][:, index] ** 2) ** 0.5,
                "reward_bias": mean(arrays["reward_error"][:, index]),
                "cumulative_return_rmse": mean(
                    arrays["cumulative_return_error"][:, index] ** 2
                )
                ** 0.5,
                "cumulative_return_bias": mean(
                    arrays["cumulative_return_error"][:, index]
                ),
                "continuation_brier": mean(
                    arrays["continuation_squared_error"][:, index]
                ),
                "deployed_alive_brier": mean(arrays["alive_squared_error"][:, index]),
                "embedding_cosine": mean(arrays["embedding_cosine"][:, index]),
                "mixed_posterior_kl": mean(arrays["posterior_kl"][:, index]),
                "action_mask_false_positive": float(
                    (mask_predicted * negative).sum() / max(negative.sum(), 1.0)
                ),
                "action_mask_false_negative": float(
                    (~mask_predicted * positive).sum() / max(positive.sum(), 1.0)
                ),
            }
    teacher = np.stack(
        [
            record["metrics"]["paths"]["teacher_factual"]["cumulative_return_error"]
            for record in records
        ]
    )
    model = np.stack(
        [
            record["metrics"]["paths"]["self_fed"]["cumulative_return_error"]
            for record in records
        ]
    )
    result["paired_self_feed_excess_return_mse"] = {
        f"h{h}": float(
            ((model[:, h - 1] ** 2 - teacher[:, h - 1] ** 2) * weight).sum() / count
        )
        for h in horizons
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--external", type=Path)
    parser.add_argument("--platform", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--outcome-semantics",
        choices=("corrected_team", "original_slots"),
        default="corrected_team",
        help="Match the checkpoint's deployed simulator: team pooling/absorbing guards "
        "or original unpooled per-slot reward and continuation heads.",
    )
    parser.add_argument("--roots", type=int, default=64)
    parser.add_argument("--max-episodes", type=int, default=128)
    parser.add_argument("--max-chunks", type=int, default=64)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 5, 8])
    parser.add_argument("--seed", type=int, default=2718)
    args = parser.parse_args(argv)
    if min(args.roots, args.max_episodes, args.max_chunks, *args.horizons) < 1:
        parser.error("Root limits and horizons must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.run.expanduser().resolve()
    checkpoint = (
        args.checkpoint or run / "ckpt" / (run / "ckpt/latest").read_text().strip()
    )
    episodes, replay_provenance = load_episodes(
        args.replay or run / "replay", args.max_chunks, args.max_episodes, args.seed
    )
    horizon = max(args.horizons)
    roots = select_roots(episodes, horizon, args.roots, args.seed)
    from ruamel.yaml import YAML

    config_text = (run / "config.yaml").read_text()
    config = YAML(typ="safe").load(config_text)
    if config["agent"]["marl"]["ctde"]["mask_calibration"]["enabled"]:
        raise ValueError("Only the maintained hard-mask PPO simulator is supported")
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [
        str(repository / "src"),
        str(args.external or repository / "external/dreamerv3"),
    ]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import embodied.jax.nets as nets

    jax.config.update("jax_platforms", args.platform)
    nets.COMPUTE_DTYPE = getattr(jnp, config["jax"].get("compute_dtype", "bfloat16"))
    observation_dim = episodes[0].data["observation"].shape[-1]
    action_count = episodes[0].data["action_mask"].shape[-1]
    forward, initialize = make_auditor(
        config, observation_dim, action_count, horizon, args.outcome_semantics
    )
    padded_length = 64 * (
        (max(len(episode.data["observation"]) for episode in episodes) + 63) // 64
    )
    example = (
        *jax.device_put(episode_inputs(episodes[0], padded_length)),
        jnp.asarray(0, jnp.int32),
    )
    params, checkpoint_provenance = load_frozen_params(checkpoint, initialize, example)
    compiled = jax.jit(forward)
    records = []
    for ordinal, (episode_index, root) in enumerate(roots):
        episode = episodes[episode_index]
        inputs = (
            *jax.device_put(episode_inputs(episode, padded_length)),
            jnp.asarray(root, jnp.int32),
        )
        # Fixed episode seed makes repeated roots share the same reconstructed
        # factual trajectory and target-step draws, not different encodings.
        _, output = compiled(params, *inputs, seed=args.seed + episode_index)
        output = jax.device_get(output)
        records.append(
            {
                "episode": episode.identity,
                "source": episode.source,
                "episode_start": episode.start,
                "root": root,
                "actions": episode.data["action"][root : root + horizon],
                "actual_mask": episode.data["action_mask"][
                    root + 1 : root + horizon + 1
                ],
                "metrics": output,
            }
        )
        print(f"Audited root {ordinal + 1}/{len(roots)}", flush=True)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": checkpoint_provenance,
        "replay": replay_provenance,
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "horizons": args.horizons,
        "seed": args.seed,
        "outcome_semantics": args.outcome_semantics,
        "outcome_semantics_description": (
            "Present-slot team pooling; zero reward for all-dead sources and zero "
            "continuation when no next controllable agent remains."
            if args.outcome_semantics == "corrected_team"
            else "Raw per-slot reward and continuation heads, matching the original "
            "92014f1 imagination path; no team pooling or all-dead absorbing guards."
        ),
        "cumulative_return_definition": "Finite model rollout sum of predicted rewards "
        "weighted by preceding predicted continuation products; factual sum uses "
        "configured discount and includes terminal reward. No critic bootstrap, "
        "lambda mixing, GAE, or focal-death actor mask is applied. In particular, "
        "original_slots does not reproduce the old GAE's separate focal-death "
        "credit truncation; this is simulator-return accuracy, not PPO advantage accuracy.",
        "team_roots": len(roots),
        "unique_episodes": len({record["episode"] for record in records}),
        "protocol": "Raw complete-episode reconstruction; identical target-step categorical keys across factual/self-fed/oracle paths; fixed root-live cohort and complete factual future; KL includes configured unimix and sums categorical variables. Oracle feeds actual online embeddings and factual liveness. Teacher-factual and oracle predicted outcomes should agree up to numerical sequence/step parity; oracle injected posterior KL should be zero.",
        "limits": "Factual replay action sequences, not current-policy or counterfactual action-value validation. Correlated roots may share episodes. Deployed liveness Brier includes hard-threshold absorbing decisions. No gradients, actor training, optimizer, or environment used.",
        "summary": aggregate_records(records, args.horizons),
        "records": records,
    }
    source_digest = hashlib.sha256()
    for path in sorted((repository / "src/majepa").rglob("*.py")):
        source_digest.update(str(path.relative_to(repository)).encode() + b"\0")
        source_digest.update(path.read_bytes() + b"\0")
    result["model_source_sha256"] = source_digest.hexdigest()
    snapshot_manifest = run / "snapshot_manifest.json"
    if snapshot_manifest.is_file():
        result["snapshot_manifest"] = json.loads(snapshot_manifest.read_text())

    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(
            result, stream, default=convert, indent=2, sort_keys=True, allow_nan=False
        )
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
