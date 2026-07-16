# Jafar/Jasmine Change Log

## 2026-07-16

### Starting point

- Branch: `jafar-jasmine-world-models`.
- Starting commit: `44eb8e94d39dae4b129ba5fe89ddd78670fc4d64`.
- Baseline command: `uv run pytest -m "not integration"`.
- Baseline result: 427 passed, 3 skipped, 7 deselected; exit 0.
- JAX: 0.4.36.
- Flax: 0.10.4.

### Documentation-first change

- Added architecture and source-conformance contracts for the Jafar arm.
- Added architecture and source-conformance contracts for the Jasmine arm.
- Added the approved implementation plan and acceptance sequence.
- Added Apache-2.0 third-party notices for both pinned sources.

No model, test, dependency, CLI, or runtime behavior changed in this
documentation-first commit.

### Evidence policy

Future entries record the exact red/green commands, commits, local artifacts,
GPU artifacts, and quality outputs produced by each implementation phase.
Results are not marked complete until fresh command output and artifact
inspection satisfy the corresponding gate.
