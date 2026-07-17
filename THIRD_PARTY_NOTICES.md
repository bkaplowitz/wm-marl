# Third-Party Notices

This repository contains source-derived adaptations of the projects below.
The adaptations retain their upstream license and identify local changes in
the corresponding module headers and source-conformance documents.

## Jafar

- Project: Jafar
- Repository: `https://github.com/FLAIROx/jafar`
- Pinned commit: `5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665`
- Upstream authors named by the project: Timon Willi, Matthew Thomas Jackson,
  and Jakob Nicolaus Foerster.
- License: Apache License, Version 2.0.
- Local conformance record:
  `src/world_marl/jafar/SOURCE_CONFORMANCE.md`.

The Jafar-derived portions include adaptations of preprocessing, axial
spatiotemporal transformer, vector-quantization, VQ-VAE tokenizer, latent
action model, MaskGIT dynamics, training loss/schedule, code-reset, and
sampling logic.

## Jasmine

- Project: Jasmine
- Repository: `https://github.com/p-doom/jasmine`
- Pinned commit: `420859bc99eecf6b07a7e9edf65d5d145935f1e1`
- Upstream authors named by the project: Mihir Mahajan, Alfred Nguyen, Franz
  Srambical, and Stefan Bauer.
- License: Apache License, Version 2.0.
- Local conformance record:
  `src/world_marl/jasmine/SOURCE_CONFORMANCE.md`.

The Jasmine-derived portions include adaptations of preprocessing, axial
transformer and attention, vector quantization, MAE tokenizer, latent action
model, diffusion-forcing dynamics, WSD scheduling, training loss, and sampling
logic.

## MuJoCo Playground

- Project: MuJoCo Playground
- Repository: `https://github.com/google-deepmind/mujoco_playground`
- GPU validation package: `playground==0.2.0`
- License: Apache License, Version 2.0.

The `playground-vision:CartpoleBalance` adapter uses the upstream registry,
MJX/Warp implementation, MJWarp batch renderer, and `pixels/view_0`
observation contract. The local adapter only converts the upstream grayscale
frame stack from `[-0.5, 0.5]` to the repository pixel contract `[0, 1]` and
adds scan-compatible environment metadata.

## Apache License 2.0

These projects are distributed under the Apache License, Version 2.0. A copy of
that license is available at `https://www.apache.org/licenses/LICENSE-2.0`.
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations.

Repository-specific reward, continuation, replay conversion, expert bridge,
simulator, PPO, artifact, and evaluation extensions are original integration
work and are not represented as upstream Jafar or Jasmine behavior.
