# Launch handoff, 2026-09-05

All six live children have saved configs and initialized GPU contexts; logs are in agent/model construction. The runner constructs the agent before make_logger(), so W&B run registration has not occurred in the bounded checks. URLs are deterministic intended run addresses, not verified online at handoff. No learning outcome is claimed.

| Candidate | Map | GPU | Train PID | Intended W&B run |
|---|---|---|---|---|
| recurrent | 3m | 1 | 54631 | [Run](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/pre-rec-ema-20260905-s0-train) |
| recurrent | 2s3z | 4 | 54762 | [Run](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/pre-rec-ema-20260905-s1-train) |
| recurrent | 8m | 5 | 54759 | [Run](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/pre-rec-ema-20260905-s2-train) |
| ema | 3m | 0 | 54760 | [Run](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/pre-rec-ema-20260905-s3-train) |
| ema | 2s3z | 2 | 54695 | [Run](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/pre-rec-ema-20260905-s4-train) |
| ema | 8m | 3 | 54761 | [Run](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/pre-rec-ema-20260905-s5-train) |

All six fixed configurations passed resolution against their pinned source. Twenty focused launcher tests passed. New mirror PID34107 and archive watcher PID34108 are detached, active, and preserve source data; prior-screen archive watcher PID21048 remains active. The new mirror has captured all six saved configurations. Immutable baseline, recurrent source and launcher archives are already stored under `/Users/osaze/MARL/remote_archives/majepa_prefill_recurrent_ema_20260905/source_snapshots/`. Each successful final128 will trigger independent verified checkpoint/config/raw-replay backup.

No further interactive monitoring, analysis, experiments, or second wave is scheduled by this agent. Existing detached workers continue as authorized.
