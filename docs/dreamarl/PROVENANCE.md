# DreaMARL Provenance

DreaMARL is maintained as first-party source in `world_marl.dreamarl`. Its
single-agent foundation was registered from Dreamer-CDP commit
`a851fa3e3d70b624b094ee1810ad4bb602346092` plus the repository's causal JEPA
Transformer overlay.

The frozen foundation used during the completed numerical parity milestone had
the following SHA-256 digests:

| File | SHA-256 |
| --- | --- |
| `agent.py` | `a0ae3fbd50e76dc4c649e3d092711229b2205f4d0fa9986e77e26b3c1ce92ce2` |
| `rssm.py` | `76e4c87005fc997299470723adb392e08a59c8c6f08ceaae3db913f045001946` |
| `m3_rssm.py` | `daf3fbe973fb277d11527ca1f2090ff8e5368f7e604a3fed3ef8c3126fab9df7` |
| `configs.yaml` | `9a2a10726e604c314baffcd497e2d1eced72c328339d8e0348f78cebdf32a126` |
| `main.py` | `37870982ac9f02ffbffa4ebb142452418dffd01a1ec4d6bb4766783df2e4fd90` |

The imported snapshot is not part of the installed DreaMARL package. Reduction
is now protected by the pinned infrastructure revision, command-semantic
checks, explicit agent-axis tests, and numerical tests for the maintained
transition modules.

The Dreamer-derived foundation remains subject to the license included at
`src/world_marl/dreamarl/LICENSE`.
