# MA-JEPA PPO reference results — 8 September 2026

`ma-jepa-ref` preserves the maintained `ma_jepa` profile from commit `b4b5f04`.
The branch adds this results inventory and its evidence files. It selects BPTT2
recurrence, factual-history posterior alignment, 50/50 uniform/recency world
replay, Bernoulli imagined action availability, H5 imagination, collection mixing
0.05, actor LR 3e-5, full-copy critic targets and encoder EMA 0.01. Factual
V-trace and representation-value objectives are disabled. This is a development
reference with demonstrated strengths and remaining failures.

## Completed results for this exact configuration

Every number below is the win percentage from a **100-episode greedy final
evaluation at 50k joint real environment transitions**, including the 5k prefill.
All ten training runs completed exactly 5,626 learner updates. Each seed value
links to its final evaluation. These are final checkpoint results, not the best
training-curve evaluations. The mean weights each completed seed equally.

| Map | Seed 0 | Seed 1 | Seed 2 | Mean | Completed seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3s_vs_3z | [90%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/am1-bernoulli-3s_vs_3z-s0-final100) | [85%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/am1-bernoulli-3s_vs_3z-s1-final100) | [95%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/am1-bernoulli-3s_vs_3z-s2-final100) | 90.0% | 3 |
| 2s3z | [42%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/am1-bernoulli-2s3z-s0-final100) | [49%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/am1-bernoulli-2s3z-s1-final100) | [40%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/am1-bernoulli-2s3z-s2-final100) | 43.7% | 3 |
| 3s_vs_4z | [0%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/am1-bernoulli-3s_vs_4z-s0-final100) | [0%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/am1-bernoulli-3s_vs_4z-s1-final100) | [0%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/am1-bernoulli-3s_vs_4z-s2-final100) | 0.0% | 3 |
| MMM | [35%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/am1-bernoulli-MMM-s0-final100) | — | — | 35.0% | 1 |

MMM has only one completed seed. The broader earlier BPTT2 benchmark matrix
predates alignment, mixed replay and sampled availability; its other maps are
not measurements of this exact reference. No completed reference results are
available here for the other maps in that matrix.

## Previous alignment + mixed replay configuration

These runs used thresholded imagined availability. They are historical
comparisons, not additional replicates of the Bernoulli reference. Runtime and
source provenance should be retained when comparing batches.

| Map | Seed 0 | Seed 1 | Seed 2 | Mean |
| --- | ---: | ---: | ---: | ---: |
| 3s_vs_3z | [93%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/cv2-mix-align-3s_vs_3z-s0-final100) | [70%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/cv2-mix-align-3s_vs_3z-s1-final100) | [80%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/cv2-mix-align-3s_vs_3z-s2-final100) | 81.0% |
| 2s3z | [22%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/cv2-mix-align-2s3z-s0-final100) | [29%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/cv2-mix-align-2s3z-s1-final100) | [46%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/cv2-mix-align-2s3z-s2-final100) | 32.3% |
| 3s_vs_4z | [0%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/cv2-mix-align-3s_vs_4z-s0-final100) | [0%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/cv2-mix-align-3s_vs_4z-s1-final100) | [30%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/cv2-mix-align-3s_vs_4z-s2-final100) | 10.0% |
| MMM | [64%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/l40d-mem95-MMM-s0-final100) | [45%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/l40d-mem95-MMM-s1-final100) | [58%](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/l40d-mem95-MMM-s2-final100) | 55.7% |

The later paired threshold controls on 2s3z scored 23% / 11% on seeds 0/1;
their Bernoulli partners scored 42% / 49%. These controls are included in the
evidence JSON and kept separate from the older 22% / 29% / 46% batch.

## Interpretation and provenance

- 3s_vs_4z fails across all three Bernoulli seeds. Strong 3s_vs_3z performance
  shows that the failure is not shared by every Stalker-versus-Zealot map.
- Bernoulli availability has not established a universal improvement: the
  older threshold MMM results are stronger, while Bernoulli has only one MMM seed.
- The newly launched collection/horizon/LR interventions are separate
  experiments and do not change this branch's reference settings.
- The historical executable package SHA256 is
  `c67602693ea1a1f2e4656fe5babe0adccf64d330cb1f6c9ab8847f68832fd966`.
  The public-default promotion is documented in
  [the reference pin audit](../reinforce_regression_review_20260908.md).
- Fresh W&B final100 summaries were matched against retained pod summaries;
  counts, completion state, episode quotas and learner budgets agree.
- Retrieval time: `2026-09-08T11:26:48.251209+00:00`.

Raw reference summaries, executed configs, training/final URLs and the W&B
verification are in [results_and_configs.json](results_and_configs.json).
Historical threshold results are in
[historical_threshold_results.json](historical_threshold_results.json).
