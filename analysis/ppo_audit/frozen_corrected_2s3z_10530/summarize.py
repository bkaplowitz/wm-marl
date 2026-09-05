"""Conditional episode-cluster bootstrap of paired frozen-simulator errors."""

from pathlib import Path
import json
import gzip
import tarfile
import numpy as np


def summarize(path):
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            audit = json.load(archive.extractfile("paired_simulator.json"))
    elif path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            audit = json.load(stream)
    else:
        audit = json.loads(path.read_text())
    records = audit["records"]
    episodes = sorted({record["episode"] for record in records})
    assignment = np.asarray([episodes.index(record["episode"]) for record in records])
    cohort = np.asarray([record["metrics"]["cohort"] for record in records], bool)
    draws = np.random.default_rng(86420).integers(
        0, len(episodes), (4000, len(episodes))
    )

    def estimate(numerator, denominator):
        # Preserve all correlated roots and roster slots within each episode.
        num = np.asarray(
            [numerator[assignment == i].sum() for i in range(len(episodes))]
        )
        den = np.asarray(
            [denominator[assignment == i].sum() for i in range(len(episodes))]
        )
        valid = den[draws].sum(-1) > 0
        samples = num[draws].sum(-1)[valid] / den[draws].sum(-1)[valid]
        return {
            "point": float(num.sum() / den.sum()),
            "episode_cluster_percentile_95": np.quantile(
                samples, [0.025, 0.975]
            ).tolist(),
        }

    result = {
        "file": str(path),
        "team_roots": len(records),
        "episodes": len(episodes),
        "agent_roots": int(cohort.sum()),
        "outcome_semantics": audit["outcome_semantics"],
        "bootstrap_draws": 4000,
        "bootstrap_seed": 86420,
        "limits": "Conditional uncertainty for this frozen checkpoint and replay cohort only; episode-cluster resampling preserves correlated roots/agents. Not training-seed uncertainty or evidence of a causal difference between historical 50k and corrected 10.5k checkpoints.",
        "horizons": {},
    }
    teacher = [record["metrics"]["paths"]["teacher_factual"] for record in records]
    model = [record["metrics"]["paths"]["self_fed"] for record in records]
    oracle = [
        record["metrics"]["paths"]["oracle_observation_alive"] for record in records
    ]
    result["max_oracle_posterior_kl"] = float(
        max(np.max(row["posterior_kl"]) for row in oracle)
    )
    result["max_teacher_oracle_outcome_difference"] = {
        key: float(
            max(
                np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key])))
                for a, b in zip(teacher, oracle)
            )
        )
        for key in ("reward", "continuation", "alive_probability")
    }
    for horizon in audit["horizons"]:
        index = horizon - 1

        def get(rows, key):
            return np.asarray([row[key] for row in rows])[:, index]

        model_error = get(model, "cumulative_return_error")
        teacher_error = get(teacher, "cumulative_return_error")
        oracle_error = get(oracle, "cumulative_return_error")
        true_mask = np.asarray([record["actual_mask"] for record in records], bool)[
            :, index
        ]
        model_mask = get(model, "mask").astype(bool)
        teacher_mask = get(teacher, "mask").astype(bool)
        positive = true_mask * cohort[..., None]
        negative = (~true_mask) * cohort[..., None]
        result["horizons"][f"h{horizon}"] = {
            "oracle_numerical_excess_return_mse": estimate(
                (oracle_error**2 - teacher_error**2) * cohort, cohort
            ),
            "self_feed_vs_oracle_excess_return_mse": estimate(
                (model_error**2 - oracle_error**2) * cohort, cohort
            ),
            "self_feed_excess_return_mse": estimate(
                (model_error**2 - teacher_error**2) * cohort, cohort
            ),
            "model_return_bias": estimate(model_error * cohort, cohort),
            "self_feed_excess_return_bias": estimate(
                (model_error - teacher_error) * cohort, cohort
            ),
            "teacher_return_bias": estimate(teacher_error * cohort, cohort),
            "self_feed_excess_alive_brier": estimate(
                (
                    get(model, "alive_squared_error")
                    - get(teacher, "alive_squared_error")
                )
                * cohort,
                cohort,
            ),
            "self_feed_excess_mask_false_positive": estimate(
                (model_mask.astype(float) - teacher_mask) * negative, negative
            ),
            "self_feed_excess_mask_false_negative": estimate(
                (teacher_mask.astype(float) - model_mask) * positive, positive
            ),
        }
    return result


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    result = {
        "historical_2s3z_seed234_simulator": summarize(
            directory.parent / "historical_2s3z_seed234_simulator.json.gz"
        ),
        "paired_simulator": summarize(directory / "diagnostic_outputs.tar.gz"),
    }
    (directory / "paired_confidence.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    for name, values in result.items():
        print(name, values["team_roots"], values["episodes"], values["agent_roots"])
        for horizon in ("h1", "h5", "h8"):
            print(horizon, json.dumps(values["horizons"][horizon]))
