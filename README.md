# MA-JEPA

A predictive world model for multi-agent reinforcement learning in SMAC.
Agents share an observation encoder, a causal history model, and a decentralized
policy. During training, a joint JEPA predictor models interactions between agents
and supplies imagined trajectories for PPO. There is no observation decoder.

This branch contains the baseline and a history-gradient variant. The latter
changes only whether factual joint JEPA prediction also trains local history
features. Both profiles are in [configs.yaml](src/majepa/configs.yaml).

## Installation

Use Python 3.11 and an NVIDIA GPU with a compatible CUDA 12 driver:

```bash
git submodule update --init --recursive
uv sync --locked --extra smac --extra cuda12
export SC2PATH=/path/to/StarCraftII
```

Install StarCraft II and the SMAC v1 maps separately, following the
[SMAC installation instructions](https://github.com/oxwhirl/smac#installation).
The environment uses difficulty `7`. Use the same StarCraft II build when
comparing runs; the Python dependency lock does not install the game itself.

The pinned `external/dreamerv3` submodule supplies Embodied's environment driver,
replay storage, JAX execution wrapper, and neural-network primitives. The agent,
JEPA objectives, PPO training, and SMAC adapter live in `src/majepa`.

## Train and evaluate

From the repository root:

```bash
PYTHON=.venv/bin/python ./scripts/run_baseline.sh
```

This trains `2s3z`, seed `0`, for 50,000 environment steps, then evaluates the
final checkpoint for 100 episodes. Each invocation creates a fresh output
directory under `runs/`.

Change the map, agent count, seed, or budget explicitly:

```bash
TASK=smac_3s_vs_4z NUM_AGENTS=3 STEPS=100000 SEED=1 \
  PYTHON=.venv/bin/python ./scripts/run_baseline.sh

TASK=smac_8m NUM_AGENTS=8 STEPS=50000 SEED=2 \
  CONFIG=history_gradient PYTHON=.venv/bin/python ./scripts/run_baseline.sh
```

Additional dotted configuration overrides go after the script name:

```bash
PYTHON=.venv/bin/python ./scripts/run_baseline.sh \
  --agent.ppo.entropy_coefficient 0.005
```

The launcher saves the **resolved configuration** in `train/config.yaml` and
reloads it for evaluation. To evaluate an existing training checkpoint:

```bash
PYTHON=.venv/bin/python ./scripts/evaluate.sh runs/RUN_NAME/train
```

For direct CLI use, expose the pinned infrastructure package:

```bash
export PYTHONPATH="$PWD/src:$PWD/external/dreamerv3"
.venv/bin/python -m majepa.main --help
```

`--configs baseline` and `--configs history_gradient` select profiles.
`--config PATH` reloads a saved configuration; dotted CLI overrides are applied
last. Fresh training requires a new log directory: resuming the exact pending
replay batch and sampler state is not implemented.

## Outputs

JSONL logging is enabled by default. To also log to your W&B account:

```bash
export WANDB_PROJECT=my-project
export WANDB_ENTITY=my-team     # optional
export WANDB_RUN_GROUP=my-study # optional
PYTHON=.venv/bin/python ./scripts/run_baseline.sh
```

Training and evaluation have separate run IDs and output directories. Outputs
include resolved settings, metrics, win rates, elapsed wall-clock seconds, and
the final checkpoint. Final evaluation also writes `evaluation_summary.json`.

Curve evaluation uses 32 greedy episodes every 5,000 steps, across four workers
with a worker offset of `50000`. Final evaluation uses 100 greedy episodes,
25 per worker, with offset `100000`. Environment seeds are the training seed
plus the worker offset plus the worker index. Evaluation preserves the training
policy's random state.

## Code layout

| Location | Responsibility |
| --- | --- |
| `configuration.py`, `configs.yaml` | Profiles, settings, and saved-config loading |
| `main.py` | Construct environments, replay, agent, and logging |
| `train.py`, `evaluation.py` | Collection/update scheduling and evaluation |
| `agent.py` | Local model and optimizer construction |
| `marl/agent.py` | Team adapter, joint modules, centralized critic |
| `marl/replay.py`, `marl/imagination.py` | Reconstruct histories and roll out the joint simulator |
| `models/` | Encoder, joint attention, prediction heads, categorical distributions |
| `world_model/` | Local Transformer dynamics and attention/cache mechanics |
| `training/learner.py`, `training/behavior.py` | World-model update, frozen PPO batches, actor/critic updates |
| `training/joint.py`, `training/direct_jepa.py`, `training/self_fed.py` | Factual, direct multi-step, and recurrent JEPA supervision |
| `training/ppo.py`, `training/replay_value.py` | Policy and value objectives |
| `replay.py`, `streams.py` | Independent replay views and deterministic sampling order |
| `envs/smac.py` | SMAC observations, legal actions, rewards, and episode outcomes |
| `scripts/` | Train-and-evaluate and checkpoint-evaluation launchers |

See [the architecture guide](docs/architecture.md) for dimensions, gradient
boundaries, and the order of a learner update.

Replay reads use the fixed4 ordering, and parameter names and active model
operations were retained during this cleanup. This is not a guarantee of
identical results across GPU types, software stacks, or arbitrary refactors.
Record the Git revision, resolved settings, hardware, driver, and game version
when reporting results. Full-training equivalence of this cleaned branch has
not yet been established.
