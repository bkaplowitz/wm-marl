# Target benchmark and interpretation

The target is a reproducible advantage over DMAWM and MATWM across most of
their overlapping tasks, using one selected algorithm configuration. The
current two-map screen tests mechanisms; it cannot establish that claim.

## MATWM reference

Published SMAC median win rates and training interaction budgets from
[MATWM Table 2](https://arxiv.org/html/2506.18537v1#S4.T2):

| Task | Environment steps | MATWM median win rate |
| --- | ---: | ---: |
| 2m_vs_1z | 50,000 | 98% |
| 2s_vs_1sc | 50,000 | 96% |
| 2s3z | 50,000 | 80% |
| 3m | 50,000 | 83% |
| 3s_vs_3z | 50,000 | 87% |
| 3s_vs_4z | 50,000 | 12% |
| 8m | 50,000 | 67% |
| MMM | 50,000 | 7% |
| so_many_baneling | 50,000 | 86% |
| 3s_vs_5z | 200,000 | 64% |
| 2c_vs_64zg | 200,000 | 7% |
| 5m_vs_6m | 200,000 | 46% |

At 50k, recovering 40–50% on 2s3z would be useful progress but would still
fall substantially short of the target. The previous local seven-map
comparison omitted five of these twelve tasks, including two 50k tasks;
its selected-task mean is not a full-suite result.

## DMAWM reference

The current official release is pinned to commit
`2acaaeb82805b55d275e3ce08ac8f713ec9afbb2`. Its scripts request 1,005,000
steps on several SMAC tasks and 405,000 on corridor, with seeds 0–4. The
default evaluation is 32 greedy episodes, four environments, every 10k
steps. Exact links and implementation details are in
[baseline_design.md](baseline_design.md).

Compare curves at the same environment-interaction count, and separately
compare final performance at the same total budget. Do not compare our 50k
result against a DMAWM final score after a million interactions. The
published DMAWM final score table has not been independently extracted in
this audit; do not invent thresholds from its training defaults. The official
paper link encountered a browser verification challenge.

## Evaluation contract

- Count joint environment transitions, not per-agent transitions. Include
  prefill in the reported training budget; keep evaluation samples separate.
- Record SMAC/StarCraft versions, reward and timeout semantics, observation
  settings, action availability, and seeds for all methods.
- Use fixed final checkpoints rather than selecting each seed's best curve
  point. Log fixed-seed curve evaluations and a separate final evaluation.
- Treat training seeds as the replication unit. More evaluation episodes
  reduce evaluation noise but do not replace independently trained seeds.
- Select a configuration on development tasks, freeze it, then evaluate the
  complete predeclared task set. Report each seed and each task, including
  failures; no per-task architecture selection or omitted difficult maps.
- Report final win rate, learning-curve area, dispersion across training
  seeds, and compute cost. Check both the count of task-level wins and the
  aggregate improvement, with uncertainty from training seeds.

The maintained branch currently exposes SMAC only. Claims about MATWM's
visual PettingZoo/MeltingPot tasks require separate environment integration
and experiments, not extrapolation from vector SMAC results.
