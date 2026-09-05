# Recovered consumer-posterior experiments, 25 August 2026

The bounded read-only check found the explicitly named W&B project
`osaze-obahor/dreamarl-ctde-consumer-loss-8m` (six runs). Run summaries and small
metadata files recover the four substantial comparisons below. The adjacent
`dreamarl-ctde-original-loss-depth-8m` project was checked at the metadata/summary
level only (21 runs); its changing depth, actor settings, and KL variants are not
a matched control for this table. No old pod was contacted, raw output log was
downloaded, or unrelated run history was searched.

| Run | Embedding / interface / posterior weights | Last logged step | W&B state | Latest curve32 wins | Latest curve return | Latest report posterior KL |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| [Current-loss control](https://wandb.ai/osaze-obahor/dreamarl-ctde-consumer-loss-8m/runs/99m67v2b) | 2 / 1 / 0 | 100,000 | finished | 0/32 | 10.771 | 0.296 |
| [Consumer KL only](https://wandb.ai/osaze-obahor/dreamarl-ctde-consumer-loss-8m/runs/faqcgi3y) | 0 / 0 / 4 | 72,880 | crashed | 0/32 | 0.053 | 0.105 |
| [Consumer KL manifold](https://wandb.ai/osaze-obahor/dreamarl-ctde-consumer-loss-8m/runs/lot0671m) | 0 / 0.1 / 4 | 71,252 | crashed | 0/32 | 1.160 | 0.112 |
| [Consumer KL hybrid](https://wandb.ai/osaze-obahor/dreamarl-ctde-consumer-loss-8m/runs/9ynpqwe0) | 0.25 / 0.1 / 4 | 70,430 | crashed | 0/32 | 2.256 | 0.094 |

“Crashed” is W&B's recorded run state. It does not establish a numerical failure,
an intentional stop, a pod interruption, or any other exit cause. The last logged
training step and latest evaluation do not necessarily coincide: evaluations
were every 2,500 transitions. These are latest curve summaries at differing
budgets, not matched-budget final128 results. No exact evaluation timestamp was
recovered from a full history scan in this bounded check.

Metadata records the same CLI profile set for all four runs:
`smac_vector ctde ctde_deep ctde_mask_full`, task `smac_8m`, seed 0, eight agents,
imagination horizon 15, train ratio 256, and intended budget 100k. Evaluation used
32 greedy episodes every 2,500 transitions, one evaluation environment, and seed
offset 50,000. Final saving was enabled, periodic saving effectively disabled
(`save_every=2147483647`), and checkpoint-at-curve-eval disabled. Loss overrides
in the table are exact `--agent.loss_scales.ctde_*` CLI values. Other profile
defaults cannot be reconstructed from these metadata files alone.

The executable was `/workspace/dreamarl/.venv/bin/python`, program
`-m dreamarl.main`, with outputs beneath
`/workspace/dreamarl_ctde_consumer_loss_clean_8m_20260825/`.
All four W&B metadata records have `git=null`, `codePath=null`, and
`codePathLocal=null`; the source commit/hash is **not established**. The control's
uploaded `config.yaml` contains W&B internals only, and no full runtime
configuration was available in the consumer variants' file inventories.

Local commit `f194171f2621f455f39a2755bc90d34af9619719`, dated 25 August 00:41 UTC,
introduced an optional `ctde_posterior` weight, frozen local-posterior consumer,
and mean-factor forward KL. Its existence is verified from git; attribution of
that exact source revision to these runs would require a separate source record.

The evidence argues against treating lower local-posterior KL as a sufficient
objective for control. However, these runs replaced or sharply weakened the
existing embedding/interface objectives while setting posterior weight to 4.
They do not isolate adding a small consumer KL to unchanged factual/recurrent
losses, and they did not test the new self-fed-input placement. The differing
training budgets and unknown exit reasons further limit causal attribution.
Recurrent fitting and a short-horizon PPO control remain separate, more direct
tests of the currently measured multi-step simulator mismatch; consumer KL can
remain an optional implementation without immediately consuming two screen slots.

Exact run names, metadata arguments, file inventories, summaries, and URLs are
retained in `historical_consumer_recovery.json`. No source fingerprint or matched
final evaluation is inferred where evidence is absent.
