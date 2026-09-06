# Value-learning and PPO regression sweep

**Superseded on 6 September 2026 by explicit user authorization to use all six
GPUs for interface/coverage verification.** Read
`interface_verification_protocol_20260906.md` and monitor
`/workspace/majepa_interface_verify_20260906/queue.json` instead. Preserve this
sweep's completed and interrupted results. Do not resume its pending jobs or
its old checkpoint keeper. The original 48-hour deadline still applies.

The user authorized a larger sweep on the existing six-A100 pod, with a decision
within 48 hours. Existing matrix jobs retain their GPUs through final evaluation.
The first available GPU starts the factual-target arm. Source snapshots and all
completed and failed runs remain separate from the frozen workshop matrix.

## First stage: 48 runs

Twelve configurations, maps `2s3z` and `3s_vs_3z`, seeds 0 and 1. Every run starts
fresh, collects 50,000 joint environment transitions including 5,000 prefill,
and evaluates the fixed final checkpoint for 100 greedy episodes. Curve evaluations
remain 32 episodes every 5,000 transitions. No partial-curve performance pruning.

| Arm | Change from BPTT2 reference |
| --- | --- |
| base | Fresh reference with diagnostic RNG isolation |
| fact | Factual successor critic targets, clipped joint lambda V-trace |
| factrep | fact + local factual-value representation loss, scale .1 |
| actor1 | One actor epoch; five critic epochs |
| fact-a1 | fact + one actor epoch |
| frep-a1 | factrep + one actor epoch |
| entlow | Entropy coefficient .0003 instead of .01 |
| retperc | Running percentile return scale instead of batch-standardized advantages |
| ret-ent | retperc + entropy .0003 |
| vreg | Slow-critic scalar-prediction regularizer, scale 1 |
| fact-vreg | fact + slow-critic regularizer |
| frep-ent | factrep + entropy .0003 |

The first six arms form a 3 x 2 comparison of value supervision and actor update
intensity. base/entlow/retperc/ret-ent separate advantage scaling, entropy, and
their interaction. vreg keeps critic target copying at 1.0 and therefore tests
within-update anchoring, not the exact historical slow-EMA REINFORCE setup.
ret-ent still uses PPO and is not an exact REINFORCE replica.

All arms keep BPTT2 self-fed horizons 2/4/5, eight anchors, recurrent loss .1,
imagination H5, one collection environment, actor 3 x 1024, world LR 4e-5,
actor/critic LR 3e-5, five critic epochs, replay-value scale .3, legal collection
mix .05, critic target full copy, encoder EMA .01, train ratio 128, and optimizer
warmup 0. Factual rho and c are clipped at 1; replay lambda remains .95.

Diagnostics receive an independent replay selector and independent JAX batch
seed counter in every arm. This removes two confirmed report/training RNG
couplings. It does not claim full determinism of asynchronous collection,
prefetch, or simulator execution. Old matrix runs are historical context;
the fresh base arm is the comparison reference.

## Conditional validation: at most 24 additional runs

After the full first stage, consider only candidates with all four final results.
Promote at most two candidates whose mean win rate across equally weighted maps
and seeds exceeds the fresh reference by at least five percentage points and
whose mean on neither map falls by more than five points. Rank eligible candidates
by macro win rate, then worst-map mean, then stable arm name. These are screening
rules, not significance tests or a claim of an established improvement.

For the selected candidates plus a fresh reference, run:

- `3s_vs_4z`, `MMM`, and `8m`, seeds 0 and 1, 50k/final100;
- `2s3z` and `3s_vs_3z`, seed 2, 50k/final100.

That is at most 72 training jobs and 3.6M training transitions for this sweep.
If no candidate qualifies, or reference data are incomplete, the scheduler stops
promotion for review. Technical failures remain failures and are not converted
to zero win rates or silently substituted. Further hypotheses require review
within the user's 48-hour decision window.

Workers stop claiming new jobs after 45 hours to leave time for evaluation and
the 48-hour decision. Already running jobs finish normally. The scheduler does
not claim the globally best configuration or guarantee DMAWM-level performance.
The decision should report every seed, uncertainty, compute, and retained-map
performance; broader DMAWM claims require the remaining matched-budget maps.

## Execution and evidence

- Pod: `root@154.54.102.56:14808`, six A100s.
- Queue: `/workspace/majepa_value_sweep_20260906/queue.json`.
- Launcher: `scripts/run_ppo_value_sweep.py` in the frozen deployed source.
- W&B group: https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-value-sweep-20260906
- Run IDs: `vs1-ARM-MAP-sSEED-train` and `vs1-ARM-MAP-sSEED-final100`.
- Package hash is checked before initialization and before each job; all possible
  train and final-evaluation commands are resolved before queue initialization.
- Matrix queue ownership and actual GPU compute processes are checked before
  claims, so matrix train-to-final handoffs cannot be mistaken for free GPUs.

Focused checks cover factual return/probability contracts, PPO invariants,
diagnostic RNG isolation, masked return normalization, and promotion rules.
A complete BPTT2 learner update passed with the combined new controls, including
separate actor/critic update counters, representation gradients, finite losses,
and both imagined and factual critic regularizers. Real-map performance remains
unmeasured at preparation time; deployment status is recorded separately.

## Diagnostic amendment, 6 September 2026

Read `seed_instability_diagnosis_20260906.md` and
`seed_diagnostic_results_20260906.json` when interpreting this sweep. Matched-root
oracle audits identify compounding predicted-observation feedback errors and
premature imagined deaths in weak `3s_vs_3z` checkpoints, with a same-direction
but less conclusive replication on `3s_vs_5z`. These errors concentrate on roots
collected by stronger policies: distinguish replay coverage/extrapolation from
the weak model's own on-policy error, and do not infer temporal causality from
the final-checkpoint comparison. Frozen factual value calibration
does not show a blanket value-overoptimism explanation. These findings change
the priority for the next controlled comparison, but do not establish a training
fix or modify the existing sweep arms or promotion rule.

A passive checkpoint keeper at
`/workspace/majepa_seed_checkpoint_archive_20260906` retains existing completed
checkpoints for `3s_vs_3z` seeds 0/1, arms base/actor1/ret-ent/frep-a1. Windows
are 5–15k, 15–25k, 25–35k, and 35–45k; at most one checkpoint per run/window,
32 total and 64 GiB. It uses hardlinks without changing training or diagnostic
RNG, and expires at the existing decision deadline. Its launcher is
`/workspace/retain_diagnostic_checkpoints_20260906.py` (initial PID 629317).
Monitor manifest freshness/errors and completion alongside queue health. Missing
checkpoints before these selected jobs start are expected. Do not replace final
evaluations with a retained checkpoint selected by curve performance.
