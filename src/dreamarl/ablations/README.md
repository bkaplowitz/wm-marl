# DreaMARL paper controls

This package contains explicit attribution controls for the local DreaMARL
world model. It is intentionally separate from the maintained training path.

The retained controls vary one scientific choice at a time: DreamerV3 RSSM
versus the causal Transformer, reconstruction versus embedding prediction,
EMA versus online targets, partial JEPA objectives, CNN versus a compact
64-pixel ViT, spatial-mask topology, and SIGReg configuration.

Run a control explicitly:

```bash
python -m dreamarl.ablations.train \
  --task dmc_walker_walk \
  --num-agents 1 \
  --temporal-model rssm \
  --world-model-objective reconstruction
```

Ablation launches use their own manifest and evaluator. They share environment,
replay, actor-critic, and logging infrastructure with DreaMARL so comparisons
remain matched. Large visual recipes rejected as impractical for control are
not part of the maintained package.
