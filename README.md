# MA-JEPA

Decoder-free multi-agent world modelling with a shared decentralized actor and
PPO on imagined team trajectories. `clean_jepa` retains the imported pod source
and adds configurable two-action imagination. `SOURCE_SHA256` and
`DEPLOYED_COMMIT` identify that imported baseline; they do not describe the
modified source tree. See [provenance](docs/provenance.md).

The maintained model has a local encoder and history for each agent, a shared
joint JEPA predictor during training, and a centralized critic. It learns future
embeddings and aligns the resulting local posterior distributions. The imported
pod configuration also enables a local teammate-belief actor adapter. There is no
observation decoder.

See [architecture](docs/architecture.md), [September 12 research notes](docs/research-notes.md),
and [source provenance and validation](docs/provenance.md).

## Install

Use Python 3.11. Run these commands from the repository root:

```bash
git submodule update --init --recursive
uv sync --locked --extra dev --extra smac --extra cuda12
export PYTHONPATH="$PWD/external/dreamerv3${PYTHONPATH:+:$PYTHONPATH}"
export SC2PATH=/path/to/StarCraftII
```

Install StarCraft II and the SMAC maps separately, following the
[SMAC instructions](https://github.com/oxwhirl/smac#installation).
`external/dreamerv3` supplies the pinned Embodied/JAX runtime; its algorithms are
not alternative first-party learners. Keep this checkout and `PYTHONPATH` when
using either command-line entry point. CPU development needs only `--extra dev`;
CUDA training also requires a compatible NVIDIA driver.

## Train

The current launcher uses the `smac_vector` and `ma_jepa` pod profiles, with local
deterministic width 8192, actor width 1024, and world-model learning rate `4e-5`.
The reference settings, sweep aliases, and results described below are historical;
the old sweep aliases are not available in the current configuration.

The earlier reference profile was WM4096, 32 categorical variables with 64 classes each,
actor 3 × 512, and both local and joint world-model LRs `1e-4`. It uses BPTT2,
imagination horizon 5, actor/critic LRs `3e-5`, PPO clipping 0.2, and fixed entropy
coefficient 0.003. Dense posterior alignment is weighted 0.05 and action margin
0.1. WM replay mixes 50% uniform and 50% recent samples; imagination roots are
independently uniform. The original background replay sampling is retained.

```bash
uv run --no-sync majepa-train \
  --task smac_2s3z \
  --num-agents 5 \
  --seed 0 \
  --experiment-dir ./runs/majepa-2s3z-seed0
```

For `3s_vs_4z`, use `--task smac_3s_vs_4z --num-agents 3`.
Set `--imag-action-samples 1` (the default) for single-action imagination, or `2`
to train on two actions per agent sampled without replacement. Both samples
receive the same source-state weight. Only the first successor continues the
imagined trajectory; the second uses its own reward and successor-value bootstrap
for training. The low-level `python -m majepa.main` entry point exposes the same
setting as `--agent.imag_action_samples`.
Use a fresh experiment directory for each independent run. Pass `--resume`
explicitly to resume an existing run.

| Profile | Local deterministic width | Local and joint WM LR |
| --- | ---: | ---: |
| `reference` | 4096 | `1e-4` |
| `wm_lr15e5` | 4096 | `1.5e-4` |
| `wm2048` | 2048 | `1e-4` |
| `wm2048_lr15e5` | 2048 | `1.5e-4` |

These profiles retain the same 64 latent classes, actor, critic, loss weights,
and data budget. They are separate experiments; smaller width or higher LR is
not automatically promoted into the reference profile.

The default budget is 50,000 driver records including a 5,000-record replay
prefill, with one collection environment. Curve evaluations use 32 greedy
held-out episodes every 5,000 records. The driver clock includes reset records;
log `counters/environment_steps` as well when comparing sample budgets.
A 200k experiment uses `--total-env-steps 200000`; prefill remains explicitly 5k.
The current launcher defaults to saving every 900 wall-clock seconds, with
additional checkpoints at curve evaluations and training completion.

On the original deployed source, this configuration produced final 2s3z win rates
of **63%, 69%, and 79%** for seeds 0, 1, and 2: a **70.3% mean** at 50k records,
using 100 evaluation episodes per seed. [Seeds 0/1](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-actor512-wm4096-wmlr1e4-20260912)
and [seed 2](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/st14-wm1e4-2s3z-s2-final100).
This is the selected reference, not a claim of the best setting on every map.
The new entropy-annealing experiment is not enabled in this branch's defaults.

The seed controls model, action, environment and replay RNGs. Asynchronous replay
can still read different buffer contents depending on scheduling, so the same
seed does not guarantee an identical training trajectory or win rate. Neither
the reverted synchronous-sampling intervention nor the draft seed changes are
included. The historical WM4096/actor512 results above are separate from the
larger imported pod baseline recorded by `SOURCE_SHA256`; see
[provenance](docs/provenance.md).

The low-level `python -m majepa.main` entry point accepts dotted configuration flags from
[src/majepa/configs.yaml](src/majepa/configs.yaml). For example:

```bash
uv run --no-sync python -m majepa.main --configs smac_vector ma_jepa \
  --agent.opt.lr 0.0001 \
  --agent.marl.ctde.opt.lr 0.00015 \
  --agent.ppo.actor_lr 0.00003 \
  --agent.ppo.critic_lr 0.00003 \
  --logdir ./runs/separate-world-rates-seed0
```

The paper-style truncated-geometric world sampler is available as an explicit
dual-view replay mode. It leaves behavior replay uniform and uses the finite
buffer capacity when converting the paper's slope parameter to the stable age
sampler:

```bash
uv run --no-sync python -m majepa.main \
  --configs smac_vector ma_jepa \
  --task smac_2s3z \
  --agent.num_agents 5 \
  --seed 0 \
  --replay.sampling truncated_geometric_world_uniform_behavior \
  --replay.truncated_geometric_alpha 10 \
  --replay.world_uniform_mix 0 \
  --logdir ./runs/truncated-geometric-seed0
```

`TruncatedGeometric` is implemented in
[src/majepa/replay.py](src/majepa/replay.py) and is selected by
[src/majepa/main.py](src/majepa/main.py)'s replay factory. The existing
`recent_world_uniform_behavior` mode remains the default reference.

Local JSON logs are enabled by default. For W&B, authenticate using `wandb login`
and pass `--wandb-project` and `--wandb-entity` to `majepa-train`. The low-level
entry point instead accepts `--logger.outputs jsonl wandb` and the usual W&B
environment variables. The resolved configuration is saved locally and
sent to W&B. Use distinct `WANDB_RUN_ID` / `WANDB_NAME` values for each training
and evaluation process, or leave them unset.

## Evaluate the final checkpoint

```bash
uv run --no-sync majepa-evaluate ./runs/majepa-2s3z-seed0
```

This reads `launch.json`, selects the current configuration profiles for its
algorithm and environment, and evaluates the latest complete checkpoint. The
recorded evaluation protocol supplies the defaults: normally 32 greedy episodes,
four environments, and the training seed plus 50,000 for SMAC. Results go under
`evaluation/seed_<seed>_<timestamp>/`, with the evaluation command saved alongside.

The final checkpoint is evaluated rather than selecting the best training-curve
point. `--episodes`, `--envs`, and `--eval-seed` can override the protocol.

## Code layout

```text
src/majepa/main.py          configuration, environments, logging, entry points
src/majepa/agent.py         shared local modules and state
src/majepa/marl/            team axes, joint simulator, centralized critic
src/majepa/models/          encoder, JEPA predictors, heads, latent distributions
src/majepa/world_model/     causal local transformer and posterior
src/majepa/training/        representation, self-fed learning, PPO, optimizers
src/majepa/replay.py        independent world-model and behavior replay views
src/majepa/train.py         collection, learner scheduling, checkpoints
src/majepa/evaluation.py    training-curve and final evaluation protocols
tests/                     focused algorithm and runtime correctness checks
```

Standard model, PPO, availability, and outcome metrics remain available. One-off
experiment launchers, remote queues, archived treatments, and standalone diagnostic
programs are outside this branch. Existing experiment data remains in W&B and the
original checkouts.

## Check

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
```

MIT. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
