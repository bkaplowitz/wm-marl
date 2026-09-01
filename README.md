# MA-JEPA

MA-JEPA is a decoder-free joint-embedding predictive world model for
cooperative multi-agent reinforcement learning. This repository contains the
single architecture used in the paper: stopped EMA cosine targets, a
joint-action-conditioned CTDE world model, all-legal action discrimination, a
centralized critic, and shared decentralized actors.

## Setup

```bash
git submodule update --init --recursive
uv sync --python 3.11 --extra dev --extra smac --extra cuda12
export SC2PATH=/path/to/StarCraftII
```

## Train

```bash
uv run majepa-train \
  --task smac_3m \
  --num-agents 3 \
  --seed 0 \
  --total-env-steps 50000 \
  --eval-interval 1000 \
  --eval-episodes 16 \
  --eval-envs 4
```

The command always resolves `smac_vector + ma_jepa`; there is no public
architecture or ablation selector.

## Evaluate

```bash
uv run majepa-evaluate runs/majepa/smac_3m/seed_0/<run> \
  --episodes 128 \
  --envs 4 \
  --eval-seed 100000
```

## Layout

```text
src/majepa/agent.py       local world model and decentralized actor
src/majepa/marl/          synchronized CTDE training
src/majepa/models/        predictive models and centralized critic
src/majepa/training/      objectives and optimization
src/majepa/envs/          environment adapters
src/majepa/configs.yaml   locked architecture and runtime defaults
tests/                    focused correctness tests
```

## Check

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## License

MIT. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
