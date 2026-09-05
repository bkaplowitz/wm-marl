# Paired recurrent simulator audit — historical 2s3z checkpoint

The 50k checkpoint from commit 976ce62 (seed 234, action scale 0.25, final 33/128
wins) was evaluated at 64 replay roots from 29 complete episodes. The local
dynamics module and the JointObservationJEPA class are identical to current
code; the encoder differs only in its module documentation. Outcomes use the
explicit original_slots mode, without the new shared-team aggregation.

All horizons use the same root-live agent cohort and recorded joint actions.
Categorical posterior draws are coupled by target time. The teacher path
receives factual local states at every step; the self-fed path receives its
own predicted embeddings. An oracle path injects factual embeddings and
liveness while retaining recurrent execution. Oracle posterior KL is zero and
its head outputs match the teacher path within numerical sequence/step error.

| Quantity | Teacher H1 | Self-fed H1 | Teacher H5 | Self-fed H5 | Teacher H8 | Self-fed H8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Online embedding cosine |0.864|0.864|0.847|0.724|0.857|0.635|
| Mixed posterior KL, sum over 32 variables |7.64|7.64|8.36|12.16|7.29|12.61|
| Action-mask false-positive rate |1.46%|1.46%|3.94%|9.20%|5.15%|15.01%|
| Action-mask false-negative rate |1.82%|1.76%|2.18%|7.39%|3.13%|9.90%|
| Deployed liveness Brier |0.0003|0.0003|0.0141|0.0511|0.0102|0.0872|
| Cumulative discounted reward RMSE |0.0921|0.0921|0.1947|0.2668|0.2253|0.3003|

The H5 paired excess cumulative-return MSE is 0.03325; an episode-cluster
percentile bootstrap gives a 95% interval of [0.00868, 0.06709]. The H8 excess is
0.03938, interval [0.00791, 0.07289]. This quantifies uncertainty over these
sampled episodes, not over training seeds or new environments. Rewards are
not uniformly optimistic: H5 self-fed cumulative bias is −0.0513 and H8 bias
is +0.0087. Actor exploitation of particular optimistic actions is not tested
by a factual-action audit.

These are training-replay roots, not a held-out world-model test set. Complete
H8 roots exclude the final seven source states of episodes. Correlated roots
share some episodes. The cumulative model reward sum is not the old PPO GAE
operator, which additionally cut focal-agent returns at individual death.

The observed recurrent degradation supports a controlled test of training the
deployed joint predictor on self-fed trajectories. The existing direct-horizon
JEPA head does not train that recurrence. First confirm this pattern on the
fresh corrected checkpoint before promoting a recurrent-training change.

Raw records: historical_2s3z_seed234_simulator.json (local, also compressed for
versioned provenance). Bootstrap details: historical_simulator_bootstrap.json.
