# Recurrent world learning rate and actor capacity screen

The user requested independent world-model learning-rate and actor-size changes,
then explicitly requested their combination. The user subsequently authorized
stopping the EMA 0.5 and 16-environment + EMA 0.5 extensions to prioritize this
screen. Other recurrent extensions continue.

All arms use the f2273e9 training snapshot, 2s3z seed 0, 50,000 total environment
transitions including 5,000 replay prefill, zero optimizer warmup, one collection
environment, self-fed recurrence at H2/H4/H5 with scale 0.1, default stopped
one-step gradients, default replay history, and full-copy critic targets.
Actor and critic learning rates stay at 3e-5. Each arm has 32-episode greedy
evaluations every 5k and a separate fixed 128-episode final evaluation.

| Cell | Local and joint world LR | Actor hidden layers | Run ID |
|---|---|---|---|
| Existing reference | 4e-5 | 3 x 1024 | pre-rec-ema-20260905-s1-train |
| Learning rate | 1e-4 | 3 x 1024 | rec-cap-20260905-s0-train |
| Actor capacity | 4e-5 | 2 x 256 | rec-cap-20260905-s1-train |
| Combination | 1e-4 | 2 x 256 | rec-cap-20260905-s2-train |

W&B entity/project: osaze-obahor/majepa-ppo-treatments.
Group: ma-jepa-recurrent-capacity-20260905.
This screens main effects and interaction at one seed; cross-seed and cross-map
confirmation remains necessary before selecting general defaults. These changes
are not a full DMAWM reproduction: representation, optimizer, collection, and
replay still differ.

The launcher resolves the complete training and evaluation configurations before
waiting for GPU availability, including architecture, both world learning rates,
actor/critic learning rates, recurrence, and evaluation settings. No additional
training test or production gate was run. Queue supervisors wait through their
predecessor's final evaluation. Source hashes, commands, and watcher PIDs are in
recurrent_capacity_deployment_20260905.json.

## Correction to benchmark context

The user supplied an image of the DMAWM author rebuttal on 2026-09-05. Its table
reports DMAWM 50k win rates of 76.9 (13.7) on 2s3z, 93.8 (2.4) on 8m, and
85.0 (2.7) on MMM. The screenshot does not define the parenthesized statistic.
It says these experiments use the same hyperparameters as their main experiments.
Thus the released million-step launch scripts do not imply an absence of DMAWM
50k results. Those reported 50k results are relevant benchmark targets; exact seed
aggregation and evaluation protocol still need alignment for numerical claims.
