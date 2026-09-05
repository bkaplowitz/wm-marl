# Fixed prefill recurrent-versus-EMA screen, 2026-09-05

The user's final allocation is six runs only: three maps (3m, 2s3z, 8m), seed 0, with recurrent H5 fitting at slow critic rate 1.0 versus the prior corrected package without recurrent fitting at slow critic rate 0.02. Both use replay prefill until step 5,000, zero local/joint optimizer warmup, PPO beginning at step 5,000, one training environment, and the selected factual replay-value scale 0.3. This startup matches the prior best verified prefill treatment (96.875% 3m, 27.34375% 2s3z, one seed).

All arms train to 50,000 joint environment transitions, train ratio 128, imagination horizon 5, actor 3×1024, actor/critic learning rate 3e-5, collection unimix .05, fixed entropy .01. Each curve point uses 32 deterministic-policy episodes every 5,000 transitions with worker offset 50,000; the final checkpoint receives a separate fixed 128-episode evaluation with four workers and offset 100,000. No best-checkpoint selection, second wave, or map-specific tuning is authorized.

| Slot | Map | Candidate | Slow critic rate | GPU |
|---|---|---|---|---|
| 0 | 3m | Recurrent | 1.0 | 1 |
| 1 | 2s3z | Recurrent | 1.0 | 4 |
| 2 | 8m | Recurrent | 1.0 | 5 |
| 3 | 3m | EMA | .02 | 0 |
| 4 | 2s3z | EMA | .02 | 2 |
| 5 | 8m | EMA | .02 | 3 |

Recurrent fitting uses horizons 2/4/5, eight anchors, common scale .1, consumer KL 0. The immutable optional package is `/workspace/ma_jepa_self_fed_da98e2d` (package SHA256 a06ed84235620d433a2030189e7017d05393d32a5b7247325cb37a046d64e7bc); EMA controls use `/workspace/ma_jepa_ppo_2663ae5` (9db9598e0da7cced1eab43036845b9ff593bb9f61713720f4654460a5c49b4e7). Source-bound, executed CPU parity covers initialized state and a complete PPO update with recurrence disabled. Recurrent arms require the separate finite full-shape production gate before dispatch. The current hardware gate uses 2s3z; 8m has greater agent count and is a new scaling check.

This compares two candidate packages; their contrast alone cannot separately identify the recurrence effect and the EMA effect. One seed per map gives a bounded generalization screen, not a variance estimate or robust algorithm ranking. The prior self-fed/H2 and subsequently proposed 12-run EMA allocations were superseded before launch; they produced no results.

W&B group: `ma-jepa-prefill-recurrent-vs-ema-20260905`, existing project `osaze-obahor/majepa-ppo-treatments`. Deterministic IDs `pre-rec-ema-20260905-s{slot}-train` and `...-final128`. Compact metrics/config/manifests mirror locally every two minutes. A separate persistent watcher archives each successful final checkpoint, raw replay and configurations independently, verifies SHA256 and preserves all remote originals. Existing correction-screen backups continue independently. The user requested launch verification and links, then stop interactive work rather than waiting for 50k outcomes.
