# Parallel-environment stability comparison

User authorized revisiting parallel collection on the successful alignment + mixed-replay configuration, to measure cross-seed stability.

- Maps: `2s3z`, `3s_vs_4z`.
- Paired training seeds: **0, 1000, 2000**. SMAC uses `training_seed + worker_index`; these blocks avoid overlapping environment seeds between replicate runs.
- Arms: one versus eight parallel environments; no factual-value or representation-loss intervention.
- Each arm/map/seed: **50,000 total environment transitions**, including 5,000 replay prefill; expected **5,626 learner updates**; training ratio 128 and existing batch size/sequence length unchanged.
- Same BPTT2, trajectory alignment, half uniform/half recency world replay, uniform behavior roots, architecture, learning rates, EMA settings and exploration mixture.
- Same four evaluation workers, 32-episode training-curve evaluations, and 100-episode final evaluation. Eight-environment driver batches can cross intermediate curve thresholds by eight transitions; the final 50k budget is exact.
- Three seed pairs per map: 12 training results, comprising 10 new runs and two reused, matching L40 seed-0 controls from the factual-value queue.
- All final models get an encoder/history-versus-actor policy-drift comparison from their 45k curve checkpoint to their 50k final checkpoint. The two reused controls are reanalyzed over that same window. Eight-environment snapshots may be at 45,008; exact endpoints are recorded.

The pinned `elements.when.Ratio` scheduler was exercised once per individual transition for both collection widths: both produced exactly 5,626 calls. The full resolved training configurations differ only in `run.envs`; final evaluation configurations are identical within each seed/map pair. Actual completed checkpoints must have exactly 50k transitions and 5,626 learner updates before being accepted.

Queue: `/workspace/majepa_parallel_env_stability_20260907/queue.json` on L40 pod `gw2ctmfarlgbsb`. Three persistent workers wait for MMM seeds 0–2 to finish training and final100 evaluation, then use their respective GPUs. They do not treat a temporary idle interval during MMM as completion. GPU 3 retains the value-comparison queue. No additional pod is created.

The original claim cutoff (2026-09-08 07:42:23 UTC) and decision deadline (2026-09-08 10:42:23 UTC) are preserved. Ten new runs add at most 500k training transitions, plus separately labeled evaluations/diagnostics. Completion before the cutoff is not guaranteed; workers stop claiming work at the cutoff and preserve failures instead of silently retrying.

W&B group: https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-parallel-env-stability-20260907

Primary readout is each seed's final win rate and the mean, range, and variability across the three paired seeds. Three replicates provide an initial stability screen, not a definitive variance estimate. Intermediate curves distinguish temporary success from sustained performance. Policy-drift measurements test whether greater collection diversity also reduces executable-policy movement; parallel collection does not directly constrain that movement.
