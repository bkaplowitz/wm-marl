# DreaMARL Ablations

This package contains controls that are intentionally excluded from canonical
DreaMARL: the DreamerV3 RSSM and decoder, compact and 224px ViTs, V-JEPA
multi-block prediction, alternative mask topologies, online/MSE targets, and
partial JEPA objectives.

The production command has no architecture switches. Run a control explicitly:

```bash
python -m dreamarl.ablations.train \
  --task dmc_walker_walk \
  --num-agents 1 \
  --temporal-model rssm \
  --world-model-objective reconstruction
```

Ablation launches use their own manifest and evaluator. They share
environment, replay, actor-critic, and logging infrastructure with the canonical
algorithm so comparisons remain controlled.
