#!/usr/bin/env python3
"""Fixed-data, joint-model-only interface intervention; never a benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
FIELDS = ("observation", "reward", "agent_present", "agent_alive",
          "controllable_alive", "action_mask", "is_first", "is_last",
          "is_terminal", "action")
TRAIN_PREFIXES = ("ctde_joint/", "ctde_rew/", "ctde_con/", "ctde_mask/", "ctde_alive/")


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(args):
    from probe_control_sufficiency import complete_episode_slices

    partitions, manifest = {}, []
    for role, path in (("own", args.own), ("strong", args.strong)):
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in FIELDS}
        slices = list(complete_episode_slices(data["is_first"], data["is_last"]))
        if len(slices) != 32:
            raise ValueError(f"Expected the preserved 32-episode bank: {path}")
        order = np.random.default_rng(314159).permutation(len(slices))
        partitions[role] = {}
        for split, selected in (("train", order[:16]), ("heldout", order[16:])):
            episodes = [{k: v[slices[i][0]:slices[i][1]] for k, v in data.items()}
                        for i in selected]
            partitions[role][split] = episodes
            directory = args.output / "data" / split
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / f"{role}.npz",
                                **{k: np.concatenate([e[k] for e in episodes]) for k in FIELDS})
            manifest.append(dict(role=role, split=split, source=str(path),
                                 source_sha256=file_hash(path),
                                 episode_indices=selected.tolist(),
                                 transitions=sum(len(e["is_first"]) - 1 for e in episodes)))
    write(args.output / "data_manifest.json", manifest)
    return partitions


def batch_episode(episode, length):
    result = {}
    for key in FIELDS:
        value = episode[key]
        pad = [(0, length - len(value))] + [(0, 0)] * (value.ndim - 1)
        result[key] = np.pad(value, pad, constant_values=key in {
            "is_first", "is_last", "is_terminal"})[None]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--own", type=Path, required=True)
    parser.add_argument("--strong", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("base", "align"), required=True)
    parser.add_argument("--coverage", choices=("own", "mixed"), required=True)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--wandb-id", required=True)
    args = parser.parse_args()
    if not 1 <= args.updates <= 500:
        parser.error("The diagnostic is bounded to 500 updates")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(SOURCE / "src"), "/workspace/external/dreamerv3"]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from ruamel.yaml import YAML
    import elements
    import embodied.jax.nets as nn
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import optax
    import wandb
    from majepa.marl.core import MARLCore
    from majepa.training.self_fed import self_fed_losses
    from probe_control_sufficiency import load_frozen_params

    jax.config.update("jax_platforms", "cuda")
    saved = YAML(typ="safe").load((args.run / "config.yaml").read_text())
    nn.COMPUTE_DTYPE = getattr(jnp, saved["jax"].get("compute_dtype", "bfloat16"))
    partitions = prepare_data(args)
    all_episodes = [episode for split in partitions.values() for rows in split.values()
                    for episode in rows]
    length = 64 * ((max(len(e["is_first"]) for e in all_episodes) + 63) // 64)
    example = batch_episode(all_episodes[0], length)
    agents = example["action"].shape[-1]
    action_count = example["action_mask"].shape[-1]
    config = elements.Config(**saved["agent"], logdir=str(args.output),
        seed=saved["seed"], jax=saved["jax"], batch_size=1, batch_length=length,
        replay_context=saved["replay_context"], replay_sampling=saved["replay"]["sampling"],
        ppo_start_step=5000, report_length=length, replica=0, replicas=1)
    config = elements.Config({**config.flat,
        "marl.ctde.self_fed.trajectory_kl_scale": 0.1 if args.arm == "align" else 0.0,
        "marl.ctde.self_fed.consumer_kl_scale": 0.0,
        "marl.ctde.self_fed.fresh_history": False,
        "marl.ctde.self_fed.bptt_steps": 2,
        "marl.ctde.self_fed.anchors": 8})
    obs_space = {key: elements.Space(value.dtype, value.shape[2:])
                 for key, value in example.items() if key != "action"}
    act_space = {"action": elements.Space(np.int32, (agents,), 0, action_count)}
    model = object.__new__(MARLCore)
    MARLCore.__init__(model, obs_space, act_space, config)

    def objective(data):
        local = model.team.local_sequence_data(data)
        previous = {"action": jnp.concatenate(
            [jnp.zeros_like(local["action"][:, :1]), local["action"][:, :-1]], 1)}
        carry = model._local_initial(agents)[:3]
        _, entries, tokens, features, _, _, ema = model._world_model_terms(
            carry, local, previous, training=False)
        losses, metrics = self_fed_losses(
            model, tokens, features, entries[1], ema, local, previous)
        loss = sum(value.mean() * model.scales[name] for name, value in losses.items())
        return loss, metrics

    checkpoint = args.run / "ckpt" / (args.run / "ckpt/latest").read_text().strip()
    checkpoint_hash = file_hash(checkpoint / "agent.pkl")
    _, provenance = load_frozen_params(checkpoint, nj.init(objective), (example,))
    # Keep the complete frozen state available to nested inference paths.
    with (checkpoint / "agent.pkl").open("rb") as stream:
        original = pickle.load(stream)
    params = jax.device_put({k: np.asarray(v) for k, v in original["params"].items()})
    del original
    variable = {k: v for k, v in params.items() if k.startswith(TRAIN_PREFIXES)}
    frozen = {k: v for k, v in params.items() if k not in variable}
    pure = nj.pure(objective)
    # A fresh, identical optimizer in all cells isolates the fixed-data repair.
    # No actor, critic, local history, encoder, EMA or collection updates occur.
    optimizer = optax.chain(optax.clip_by_global_norm(1000.0),
                            optax.adam(4e-5, b1=0.9, b2=0.999, eps=1e-8))
    state = optimizer.init(variable)

    @jax.jit
    def update(variable, state, data, key):
        def loss_fn(variables):
            # Ninjax scan discovers accessed keys only in a modifiable context.
            # The returned context is discarded; only explicit Optax updates to
            # the five selected joint-model prefixes are retained.
            return pure({**frozen, **variables}, data, seed=key,
                        create=False, modify=True)[1]
        (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(variable)
        updates, state = optimizer.update(gradients, state, variable)
        return optax.apply_updates(variable, updates), state, loss, metrics, optax.global_norm(gradients)

    status = dict(phase="initializing", started_at=time.time(), arm=args.arm,
        coverage=args.coverage, updates=args.updates, checkpoint=provenance,
        checkpoint_sha256=checkpoint_hash, environment_steps_added=0,
        trainable_prefixes=TRAIN_PREFIXES, trajectory_kl_scale=float(
            config.marl.ctde.self_fed.trajectory_kl_scale),
        objective="Existing BPTT2 auxiliary, optionally factual-history posterior KL",
        input_protocol="Whole-episode 16/16 candidate-train/heldout split per source. Own cell uses 16 own episodes; mixed cell uses first 8 own and first 8 strong episodes. Identical draw keys across objectives, 500 updates and eight anchors per update.",
        benchmark_eligible=False)
    write(args.output / "status.json", status)
    run = wandb.init(project="majepa-ppo-treatments", entity="osaze-obahor",
        group="ma-jepa-interface-verify-20260906", id=args.wandb_id,
        name=args.wandb_id, job_type="offline_diagnostic",
        config=status, tags=["diagnostic", "fixed-data", args.arm, args.coverage])
    rng = np.random.default_rng(1441)
    history = (args.output / "metrics.jsonl").open("x")
    try:
        for index in range(args.updates):
            slot = int(rng.integers(0, 8))
            role = "strong" if args.coverage == "mixed" and index % 2 else "own"
            episode_index = slot + (8 if args.coverage == "own" and index % 2 else 0)
            data = batch_episode(partitions[role]["train"][episode_index], length)
            variable, state, loss, metrics, norm = update(
                variable, state, data, jax.random.PRNGKey(9001 + index))
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == args.updates:
                loss, metrics, norm = jax.device_get((loss, metrics, norm))
                row = dict(update=index + 1, loss=float(loss), gradient_norm=float(norm),
                           elapsed_seconds=time.time() - status["started_at"])
                row.update({k: float(v) for k, v in metrics.items()})
                if not all(np.isfinite(value) for value in row.values()):
                    raise ValueError("Nonfinite diagnostic update")
                history.write(json.dumps(row) + "\n")
                history.flush()
                run.log(row, step=index + 1)
                status.update(phase="training", completed_updates=index + 1,
                              updated_at=time.time(), loss=row["loss"])
                write(args.output / "status.json", status)
                print(json.dumps({k: row[k] for k in ("update", "loss", "gradient_norm", "elapsed_seconds")}), flush=True)
        replacement = jax.device_get(variable)
        if not all(np.isfinite(v).all() for v in replacement.values()):
            raise ValueError("Nonfinite joint-model weights")
        with (checkpoint / "agent.pkl").open("rb") as stream:
            payload = pickle.load(stream)
        payload["params"].update({k: np.asarray(v) for k, v in replacement.items()})
        target = args.output / "run/ckpt/diagnostic-final"
        target.mkdir(parents=True)
        with (target / "agent.pkl").open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        (target / "step.pkl").write_bytes((checkpoint / "step.pkl").read_bytes())
        (target / "done").touch()
        (target.parent / "latest").write_text(target.name + "\n")
        (args.output / "run/config.yaml").write_text((args.run / "config.yaml").read_text())
        if file_hash(checkpoint / "agent.pkl") != checkpoint_hash:
            raise RuntimeError("Original checkpoint changed during diagnostic")
        status.update(phase="complete", finished_at=time.time(),
                      original_checkpoint_unchanged=True,
                      frozen_parameter_keys_unchanged=True,
                      output_checkpoint=str(target))
        write(args.output / "status.json", status)
        run.summary.update(status)
    except BaseException as error:
        status.update(phase="failed", error=repr(error), finished_at=time.time())
        write(args.output / "status.json", status)
        raise
    finally:
        history.close()
        run.finish()


if __name__ == "__main__":
    main()
