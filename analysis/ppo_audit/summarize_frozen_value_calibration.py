"""Reproduce episode-clustered calibration estimates from frozen raw episodes.

The two policies share a training seed and evaluation protocol, not trajectories.
Cross-policy intervals resample each policy's complete episodes independently.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_calibration_module():
    path = Path(__file__).resolve().parents[2] / "scripts/evaluate_value_calibration.py"
    spec = importlib.util.spec_from_file_location("value_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cluster_estimates(rows, draws):
    """Ratios of per-episode sufficient statistics retain state/agent weighting."""
    rows = np.asarray(rows, np.float64)

    def calculate(totals):
        count = totals[..., 0]
        denominator = np.maximum(count, 1)
        return {
            "prediction_mean": totals[..., 1] / denominator,
            "target_mean": totals[..., 2] / denominator,
            "bias": totals[..., 3] / denominator,
            "rmse": np.sqrt(totals[..., 4] / denominator),
            "mae": totals[..., 5] / denominator,
        }

    total = rows.sum(0)
    if not total[0]:
        return {"count": 0}, {}
    sampled = rows[draws].sum(1)
    valid = sampled[:, 0] > 0
    point = calculate(total)
    samples = calculate(sampled)
    result = {
        "count": int(total[0]),
        "contributing_episodes": int(np.count_nonzero(rows[:, 0])),
        "valid_bootstrap_draws": int(valid.sum()),
    }
    for name in point:
        result[name] = {
            "point": float(point[name]),
            "episode_cluster_percentile_95": np.quantile(
                samples[name][valid], [0.025, 0.975]
            ).tolist(),
        }
        samples[name] = np.where(valid, samples[name], np.nan)
    return result, samples


def sufficient_statistics(prediction, target, mask):
    mask = mask & np.isfinite(prediction) & np.isfinite(target)
    prediction, target = prediction[mask], target[mask]
    error = prediction - target
    return np.asarray(
        [
            len(error),
            prediction.sum(),
            target.sum(),
            error.sum(),
            np.square(error).sum(),
            np.abs(error).sum(),
        ],
        np.float64,
    )


def summarize_run(directory, module, *, seed, bootstrap_draws):
    summary = json.loads((directory / "summary.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    raw_path = directory / "episodes.npz"
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if digest != summary["raw_archive"]["sha256"]:
        raise ValueError(f"Raw episode hash mismatch: {raw_path}")
    if summary["frozen_state_before"] != summary["frozen_state_after"]:
        raise ValueError(f"Frozen state changed: {directory}")
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    offsets = raw["episode_offsets"]
    episodes = len(offsets) - 1
    if episodes != summary["episodes"]:
        raise ValueError("Episode count differs from completed summary")
    draws = np.random.default_rng(seed).integers(
        0, episodes, (bootstrap_draws, episodes)
    )
    rows = {}
    for start, stop in zip(offsets[:-1], offsets[1:]):
        data = {
            key.replace("__", "/"): value[start:stop]
            for key, value in raw.items()
            if key not in {"episode_offsets"}
        }
        targets = module.episode_targets(data, summary["protocol"]["discount"])
        present = data["agent_present"].astype(bool)
        alive = data["controllable_alive"].astype(bool)
        sampled = data["sampled_state"].astype(bool)[:, None]
        initial = np.zeros_like(present)
        initial[0] = present[0]
        death_arrival = np.zeros_like(present)
        death_arrival[1:] = alive[:-1] & ~alive[1:] & present[1:]
        death_arrival[-1] = False
        masks = {
            "all": sampled & present,
            "alive": sampled & present & alive,
            "dead": sampled & present & ~alive,
            "episode_initial": initial,
            "first_post_death": death_arrival,
            "terminal": data["is_terminal"][:, None] & present,
        }
        for head in ("live", "slow"):
            prediction = data[f"calibration/{head}"].astype(np.float64)
            comparisons = {
                "monte_carlo": targets["monte_carlo"],
                "h5_bootstrap": targets[f"h5/{head}_bootstrap"],
                "h15_bootstrap": targets[f"h15/{head}_bootstrap"],
            }
            for comparison, target in comparisons.items():
                for slice_name, mask in masks.items():
                    key = f"{head}/{comparison}/{slice_name}"
                    rows.setdefault(key, []).append(
                        sufficient_statistics(prediction, target, mask)
                    )
    estimates, samples = {}, {}
    for key, value in rows.items():
        estimates[key], samples[key] = cluster_estimates(value, draws)
        head, comparison, slice_name = key.split("/")
        if slice_name != "first_post_death":
            reference = summary["calibration"][head][comparison][slice_name]
            if estimates[key]["count"] != reference["count"]:
                raise ValueError(f"Reference count mismatch: {key}")
            if reference["count"]:
                for metric in ("bias", "rmse", "prediction_mean", "target_mean"):
                    if not np.isclose(
                        estimates[key][metric]["point"], reference[metric], atol=1e-6
                    ):
                        raise ValueError(
                            f"Reference statistic mismatch: {key}/{metric}"
                        )
    records = summary["episode_records"]
    episodic = {}
    for metric in ("return", "discounted_return", "win", "steps"):
        values = np.asarray([record[metric] for record in records], np.float64)
        estimate = values[draws].mean(1)
        episodic[metric] = {
            "point": float(values.mean()),
            "episode_percentile_95": np.quantile(estimate, [0.025, 0.975]).tolist(),
        }
        if metric == "win":
            # A bootstrap of zero observed wins has a degenerate [0, 0] interval.
            # Use the score interval when reporting uncertain win probabilities.
            z = 1.959963984540054
            proportion = values.mean()
            denominator = 1 + z**2 / episodes
            center = (proportion + z**2 / (2 * episodes)) / denominator
            half = (
                z
                * np.sqrt(
                    proportion * (1 - proportion) / episodes + z**2 / (4 * episodes**2)
                )
                / denominator
            )
            episodic[metric]["wilson_binomial_95"] = [
                float(max(0, center - half)),
                float(min(1, center + half)),
            ]
        samples[f"episode/{metric}"] = {"mean": estimate}
    result = {
        "directory": str(directory),
        "episodes": episodes,
        "raw_sha256": digest,
        "source": provenance["source"],
        "training_manifest": provenance["training_manifest"],
        "checkpoint_files": provenance["checkpoint_files"],
        "frozen_state_unchanged": True,
        "protocol": summary["protocol"],
        "bootstrap_seed": seed,
        "episode_outcomes": episodic,
        "calibration": estimates,
    }
    return result, samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    args = parser.parse_args(argv)
    module = load_calibration_module()
    original, left = summarize_run(
        args.original, module, seed=73101, bootstrap_draws=args.bootstrap_draws
    )
    corrected, right = summarize_run(
        args.corrected, module, seed=73102, bootstrap_draws=args.bootstrap_draws
    )
    differences = {}
    for key in left.keys() & right.keys():
        differences[key] = {}
        for metric in left[key].keys() & right[key].keys():
            values = right[key][metric] - left[key][metric]
            values = values[np.isfinite(values)]
            if len(values):
                differences[key][metric] = {
                    "independent_episode_percentile_95": np.quantile(
                        values, [0.025, 0.975]
                    ).tolist()
                }
    result = {
        "bootstrap_draws": args.bootstrap_draws,
        "uncertainty": "Conditional on these two frozen seed1 checkpoints and the evaluation stream. Whole episodes retain within-episode temporal and agent correlation. The policy arms have different trajectories; cross-arm intervals independently resample episodes, not aligned state-agent pairs. These intervals omit training-seed uncertainty and do not establish an algorithm-level causal gain.",
        "original": original,
        "corrected": corrected,
        "corrected_minus_original": differences,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
