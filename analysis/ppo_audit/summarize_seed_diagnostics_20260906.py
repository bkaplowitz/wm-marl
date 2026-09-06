"""Summarize frozen diagnostics with paired episode-cluster uncertainty."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent / "majepa_seed_diagnostics_20260906"


def bootstrap_pair(records, first, second, denominator, transform=lambda x: x):
    episodes = sorted({r["episode"] for r in records})
    ids = np.array([episodes.index(r["episode"]) for r in records])
    def cluster(values):
        return np.bincount(ids, weights=values, minlength=len(episodes))
    a, b, den = map(cluster, (first, second, denominator))
    draws = np.random.default_rng(20260906).integers(
        0, len(episodes), (4000, len(episodes))
    )
    div = den[draws].sum(1)
    valid = div > 0
    delta = transform(a[draws].sum(1)[valid] / div[valid])
    delta -= transform(b[draws].sum(1)[valid] / div[valid])
    return {"difference": float(transform(a.sum() / den.sum()) - transform(b.sum() / den.sum())),
            "episode_bootstrap_95_interval": np.quantile(delta, [.025, .975]).tolist(),
            "episodes": len(episodes), "roots": len(records)}


def summarize_map(map_name, seeds, reference):
    models = {}
    for seed in seeds:
        p = ROOT / f"factor-oracles-{map_name}-s{seed}"
        if not p.exists():
            p = ROOT / f"common-world-{map_name}-s{seed}"
        if not p.exists():
            return None
        models[seed] = json.loads(p.read_text())
    records = models[reference]["records"]
    def identity(record):
        return record["episode"], record["source"], record["root"]
    for model in models.values():
        assert list(map(identity, model["records"])) == list(map(identity, records))
    cohort = np.array([r["metrics"]["cohort"] for r in records], bool)
    truth = np.array([r["metrics"]["actual_alive"] for r in records], bool)
    arrays, scores, source_scores = {}, [], []
    for seed, model in models.items():
        for path in model["records"][0]["metrics"]["paths"]:
            values = [r["metrics"]["paths"][path] for r in model["records"]]
            alive = np.array([v["alive"] for v in values], bool)
            probability = np.array([v["alive_probability"] for v in values])
            reward_error = np.array([v["cumulative_return_error"] for v in values])
            for horizon in (1, 2, 4, 5, 8):
                target = truth[:, horizon - 1]
                factual_live = target & cohort
                factual_dead = ~target & cohort
                false = ((~alive[:, horizon - 1]) & factual_live).sum(-1)
                missed = (alive[:, horizon - 1] & factual_dead).sum(-1)
                brier = (((probability[:, horizon - 1] - target) ** 2) * cohort).sum(-1)
                mse = ((reward_error[:, horizon - 1] ** 2) * cohort).sum(-1)
                arrays[seed, path, horizon] = (false, brier, mse)
                scores.append({"seed": seed, "path": path, "horizon": horizon,
                    "false_deaths": int(false.sum()), "actually_alive": int(factual_live.sum()),
                    "false_death_rate": float(false.sum() / max(factual_live.sum(), 1)),
                    "missed_deaths": int(missed.sum()), "actually_dead": int(factual_dead.sum()),
                    "liveness_brier": float(brier.sum() / cohort.sum()),
                    "finite_reward_sum_rmse": float(np.sqrt(mse.sum() / cohort.sum()))})
                if horizon == 5:
                    for source in sorted({r["source"] for r in records}):
                        selected = np.array([r["source"] == source for r in records])
                        source_scores.append({
                            "seed": seed, "path": path, "source_policy": source,
                            "roots": int(selected.sum()),
                            "false_deaths": int(false[selected].sum()),
                            "actually_alive": int(factual_live[selected].sum()),
                            "liveness_brier": float(
                                brier[selected].sum() / cohort[selected].sum()
                            ),
                        })
    differences = []
    for seed in seeds:
        if seed == reference:
            continue
        a, b = arrays[seed, "self_fed", 5], arrays[reference, "self_fed", 5]
        differences.append({"seed": seed, "reference_seed": reference,
            "h5_false_death_rate": bootstrap_pair(records, a[0], b[0], (truth[:, 4] & cohort).sum(-1)),
            "h5_liveness_brier": bootstrap_pair(records, a[1], b[1], cohort.sum(-1)),
            "h5_finite_reward_sum_rmse": bootstrap_pair(records, a[2], b[2], cohort.sum(-1), np.sqrt)})
    return {"map": map_name, "reference_seed": reference, "scores": scores,
            "h5_by_source_policy": source_scores,
            "paired_differences": differences, "model_source_sha256": models[reference]["model_source_sha256"],
            "cohort": {"roots": len(records), "episodes": len({r['episode'] for r in records}),
                       "root_live_agents": int(cohort.sum()),
                       "roots_per_source_policy": {s: sum(r['source'] == s for r in records)
                                                   for s in sorted({r['source'] for r in records})}}}


def main():
    output = {"calibration": [], "common_world": [],
        "limits": "Diagnostics of fixed checkpoints and recorded actions. Oracle interventions isolate simulator feedback errors, not a measured change in policy win rate. Model comparisons use matched raw roots; different maps remain separate. Bootstrap resamples whole contributing episodes, not agents or individual states. All five calibrations use 32 fresh sampled-policy episodes, not the greedy final100 protocol."}
    for p in sorted(ROOT.glob("calibration-*/summary.json")):
        d = json.loads(p.read_text())
        assert d["frozen_state_before"] == d["frozen_state_after"]
        output["calibration"].append({"name": p.parent.name, "episodes": d["episodes"],
            "win_rate": d["win_rate"], "return_mean": d["return_mean"],
            "discount": d["protocol"]["discount"], "calibration": d["calibration"],
            "frozen_verified": True})
    for map_name, seeds, reference in [("3s_vs_3z", (0, 1, 2), 1), ("3s_vs_5z", (0, 2), 2)]:
        result = summarize_map(map_name, seeds, reference)
        if result:
            output["common_world"].append(result)
    dest = ROOT.parent / "seed_diagnostic_results_20260906.json"
    dest.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for case in output["common_world"]:
        print(case["map"], case["cohort"])
        for row in case["scores"]:
            if row["horizon"] == 5:
                print(row)
        for row in case["paired_differences"]:
            print(row)


if __name__ == "__main__":
    main()
