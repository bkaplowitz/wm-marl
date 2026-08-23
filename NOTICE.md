# Notices and third-party components

DreaMARL first-party source is distributed under the repository's MIT
[License](LICENSE). Portions of the local runtime and algorithmic structure are
derived from or designed for numerical parity with DreamerV3; the root license
therefore retains the applicable Danijar Hafner copyright notice.

## Pinned source repositories

The following Git submodules are included for runtime support, numerical
reference, or isolated benchmark comparison. Each remains governed by the
license shipped in its own directory.

| Component | Revision | License and copyright | Use in this repository |
| --- | --- | --- | --- |
| [DreamerV3](https://github.com/danijar/dreamerv3) | `e3f02248693a79dc8b0ebd62c93683888ddaccfe` | MIT; copyright 2024 Danijar Hafner | Embodied runtime, numerical parity reference, official DMC score archive, isolated baseline |
| [MARIE](https://github.com/breez3young/MARIE) | `5dc114f78e9f35389b843e05f01c455988451d0e` | MIT; copyright 2025 Yang Zhang | Isolated multi-agent comparison implementation |
| [Dreamer-CDP](https://github.com/fmi-basel/Dreamer-CDP) | `a851fa3e3d70b624b094ee1810ad4bb602346092` | MIT; see `external/dreamer-cdp/LICENSE` | Isolated single-agent comparison implementation |
| [NE-Dreamer](https://github.com/corl-team/nedreamer) | `11cd3a978b83743f795cbfa81c2e095344912c17` | MIT; copyright 2026 Naoki Morihira and George Bredis | Isolated single-agent comparison implementation |

The table is an attribution summary, not a replacement for the complete
licenses in the corresponding submodules.

## Environment software and assets

DreaMARL can interface with DeepMind Control Suite, SMAC, and StarCraft II.
These environments and game assets are not relicensed by the
DreaMARL MIT License. Users are responsible for obtaining them from their
official distributions and complying with their respective licenses and terms.

See [provenance.md](docs/provenance.md) for exact algorithm,
environment, and experiment boundaries.
