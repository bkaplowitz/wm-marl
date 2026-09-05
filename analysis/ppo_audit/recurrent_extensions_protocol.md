# Four recurrent extensions, 2026-09-05

User authorized stopping all EMA .02 jobs and using the four released GPUs
for the four proposed extensions. EMA 3m and 2s3z had already finished;
the owned EMA 8m supervisor received SIGTERM and cleaned up its process group.
The ongoing recurrent 2s3z/8m jobs were not stopped.

All new arms are fresh 2s3z seed0 runs. The reference is the existing
`pre-rec-ema-20260905-s1-train`, with the same 5k prefill, zero world optimizer
warmups, H5 imagination, recurrent H2/H4/H5, eight anchors, auxiliary .1,
replay critic .3, train ratio128, 50k total transitions and fixed final128.
Each arm changes one setting:

| Slot / GPU | Change |
|---|---|
| 0 | Reconstruct auxiliary history and local states from current raw replay observations |
| 1 | Two-transition truncated backpropagation, frozen local parameters |
| 2 | Sixteen collection environments, unchanged total transition/update budgets |
| 3 | Critic target rate .5 |

Fresh-history reconstruction changes only the auxiliary root states. Ordinary
world losses retain their existing replay path, and factual targets are stopped.
The new branch shares the behavior/PPO raw-history reconstruction code. Tests
check independence from stored latent codes and sensitivity to raw observations.

BPTT uses chunks (1,2), (3,4), (5). Thus H2/H4 losses credit two transitions;
H5 is a one-transition tail. Local parameters are frozen while their input
Jacobian is retained. Hard liveness/action support remains discrete; no gradient
crosses a hard death decision. Consumer KL stays zero. Tests verify exact
derivatives, chunk boundaries, real-model parameter ownership and active PPO.

EMA .5 uses a local subclass of the external SlowModel, preserving its state
paths/counter/update formula while allowing the full convex rate interval.
The external dependency and prior source snapshots are not modified.

This is a one-map, one-seed screening experiment. Improvements need replication
and validation on other maps. Fixed final128 is separate from 32-episode curves.
There are no queued additional seeds or second waves.

## Additional allocation requested by the user

The user subsequently authorized two more experiments on GPUs4/5. Slot4
combines fresh history with BPTT2; slot5 combines sixteen environments with
critic EMA .5. All remaining settings and the 2s3z seed0 budget are unchanged.
These combinations can be compared with their individual components in slots0-3.
They use the same immutable f2273e9 training package and a separate 8052058
launcher. Each waits for the preceding recurrent supervisor and an idle GPU;
ongoing final evaluations are not interrupted. Results live in a separate
`majepa_recurrent_combinations_20260905` directory with independent two-run
mirror/backup watchers, and share the six-run W&B extensions group.

At the user's request, the additional production gates and broad regression
run were stopped and the four original experiments were launched immediately.
Completed checks included full tiny PPO execution for each code variant,
local-parameter gradient ownership, raw-history invariance, EMA averaging,
resolved production configurations, and exact active-recurrence/PPO parity
between the previous and current default implementation. Production-size
executability of the new structural variants was not established by the
cancelled gates; their actual training runs provide that check.
