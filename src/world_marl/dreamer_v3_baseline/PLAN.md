# DreamerV3 repair execution plan

**Goal:** replace the approximate baseline with a readable native JAX/Flax
translation of the 2025 Nature DreamerV3 algorithm and its official online
system, defaulting to the published `paper` profile and integrating with
wm-marl's real DMC and comparison interfaces.

`ARCHITECTURE.md` is the production contract. The two pinned official revisions
are read-only implementation oracles; their relevant Dreamer files are
identical. The `paper` profile is the Nature experiment profile, while
`upstream-current` explicitly selects the pinned source defaults.

## Controller and safety protocol

- Work only on `dreamer-v3-parity-port-backup-20260713` in
  `/Users/bkaplowitz/Developer/work/wm-marl-dreamer-v3-parity-port`. Do not
  create branches/worktrees, mutate Git internals, edit any other checkout,
  launch remote/paid compute, push, merge, or rewrite history.
- Preserve unrelated user changes. Before every dispatch, record branch, HEAD,
  status, symlink state, and `BASE_SHA`; stop if an owned path has an unrelated
  edit.
- For each unit, a controller-captured live reference snapshot records both
  read-only checkouts' HEAD, branch, exact NUL status, tracked diff, and
  untracked inventory before dispatch and compares them afterward. If external drift occurs,
  report it and establish a new accepted read-only baseline only
  after confirming no agent mutation; never reset, clean, or edit either
  reference checkout. Final verification uses the last accepted live process
  snapshot, not a tracked historical tuple.
- Execute tasks in the dependency order below. Only one editing agent may run
  at a time. Read-only spec and quality reviewers may run concurrently after
  editing stops. Never dispatch agents that edit overlapping files.
- Every behavior change follows red-green-refactor TDD. The first failing test,
  failure reason, focused green output, changed files, self-review, concerns,
  and `Commit: pending independent review` go in the implementer report.
- Generate the small non-staging review handoff below from the recorded base
  plus the uncommitted diff. Dispatch a fresh read-only spec reviewer and a
  separate fresh read-only quality reviewer. Both reviewers audit the handoff
  and live owned files before reviewing behavior. A
  fresh fixer adds covering tests for every Critical or Important finding;
  repeat both relevant reviews until Critical=0 and Important=0.
- Only after that gate may the controller rerun acceptance commands and create
  the task's Conventional Commit. Append the final SHA and command results to
  the report and mutable progress ledger. Commit messages and history are not
  scientific evidence.
- Every task report contains: task/base/final SHA, summary, authority/source
  mapping, files changed, RED command/output, GREEN and acceptance
  commands/output, review reports and verdicts, self-review, concerns, deferred
  external gates, and final commit subject.
- Each implementer or fixer receives one explicit owned-path allowlist. Its
  brief, report, and ledger are process state, not production provenance.
  After each unit, the controller compares the literal diff path/symbol set to
  that allowlist; any needed prior-owner change stops the unit and becomes a
  fresh smallest-owner fixer with literal files/symbols, a RED, focused pytest,
  Ruff check/format, independent reviews, and its own commit before the blocked
  unit resumes. A later task may never use a vague “boundary fix” exception to
  edit earlier Agent, optimizer, replay, DMC, driver, artifact, or checkpoint
  owners.
  The current handoff contains only current RED/GREEN/acceptance commands and
  their results, changed owned files, and concerns. Reviewers may write only
  their current verdict file.
- Local CPU/debug evidence, local real-dm_control evidence, and unauthorized
  Linux GPU/scientific-scale evidence are reported as three distinct levels.
  No mock or synthetic run is labeled a real DMC or scientific result.

### Small current review handoff

For each implementer or fixer cycle, the controller writes one short Markdown
handoff and one patch. The handoff records only `base/branch/HEAD`, the literal
owned-path allowlist, the exact before/current status, and the exact named RED,
GREEN, focused-acceptance, Ruff, `git diff --check`, and format-baseline
commands with their current results. Each evidence result appears immediately after its exact command,
immediately beside its unedited current result. The
patch is one owned diff from the recorded base,
including allowed new files. Ignored process files are not patch inputs.

Before dispatch, the controller captures the literal status and rejects an
owned path that already has an unrelated edit. After editing, it requires every
new status entry to be owned and requires every pre-existing unowned status
entry to remain byte-for-byte identical. This is a status/allowlist safety
check, not a provenance system. Reviewers reproduce the named commands and
inspect the live owned files and patch. No authentication layer, callback
identity protocol, or recursive package is created.

The handoff labels the command evidence literally: `RED command and unedited result`,
`semantic/source/AST GREEN command and unedited result`, `focused
acceptance command and unedited result`, `Ruff command and unedited result`,
`git diff --check command and unedited result`, and `format-baseline command and unedited result`.
It contains no cumulative report history.

### Stable fixture-generator and sequential `agent.py` ownership

Tasks 1-5 build one CLI without overlapping editors. Its stable invocation is

```text
python -m world_marl.dreamer_v3_baseline.fixture_generator SUBCOMMAND
  --profile {paper,upstream-current}
  --observation-mode {vision,proprio}
  --reference-checkout /private/tmp/danijar-dreamerv3-20260713
  --source-revision REVISION
  [--current-source-revision REVISION]
  --output-dir tests/fixtures/dreamer_v3
```

The profile and observation-mode flags are required for each case subcommand;
`generate-all` omits them and deterministically runs the complete registered
matrix for both profiles using its required `--source-revision` paper pin and
`--current-source-revision` current pin. `--compute-dtype
{bfloat16,float32}` is required only by `networks` and `rssm`.
`refresh-manifest` additionally requires `--fixture-stem STEM` and reads the
existing tracked NPZ without changing it. Task 1 owns the stable root `main`,
`_parse_args`, `_validate_reference`, `_canonical_request`, `_write_pair`,
`refresh_manifest`, `_PARSER_REGISTRY`, the `_register_parser` decorator, and
`_register_refresh_manifest_parser`. Later tasks do not edit `_parse_args` or
the registry mechanism; each adds its decorated exact helper, and `_parse_args`
iterates the resulting registration order. Task 2 owns
`generate_replay` and `_register_replay_parser`. Task 3 owns
`generate_distributions`/`_register_distributions_parser`,
`generate_networks`/`_register_networks_parser`,
`generate_rssm`/`_register_rssm_parser`, and
`generate_world_model`/`_register_world_model_parser`. Task 4b owns
`generate_world_loss`/`_register_world_loss_parser`; Task 4c owns
`generate_agent_loss`/`_register_agent_loss_parser`; Task 5 owns
`generate_optimizer_five_step`/`_register_optimizer_five_step_parser` and
`generate_all`/`_register_generate_all_parser`. Each task owns only its named
implementation/helper pair and the named parser test in its behavioral test
file. `generate-all` invokes every registered generation case in deterministic
stem order and is the sole complete-regeneration command.

Sequential `agent.py` ownership is equally explicit. Task 2 owns `AgentCarry`,
`validate_action_tree`, `validate_replay_row`, `DreamerAgent.initial`, and
`DreamerAgent.apply_replay_context`. Task 3 owns `DreamerAgent.preprocess` and
`DreamerAgent.world_model`. Task 4a creates the complete `AgentLoss` declaration
and owns `lambda_return`; Tasks 4b-4d populate or consume its fixed fields but
never edit the declaration. Task 4b owns `DreamerAgent.world_loss`; Task 4c owns `DreamerAgent.imagine`,
`DreamerAgent.imag_loss`, and `DreamerAgent.repl_loss`; Task 4d owns final
composition in `DreamerAgent.initial`, `policy`, `loss`, and `report`; Task 5
owns only the transactional return boundary of `DreamerAgent.loss`. These are
sequential overlaps, never permissions for simultaneous editors.

## Task 0: Correct architecture, plan, and ledger

**Owned files and boundary:** only
`src/world_marl/dreamer_v3_baseline/ARCHITECTURE.md`,
`src/world_marl/dreamer_v3_baseline/PLAN.md`, and
`.superpowers/sdd/progress.md`, plus the ignored Task 0 process report. No
production code, tests, fixtures, metadata, or reference files.

**Interfaces/dependencies:** establish the binding production design, corrected
dependency graph, historical verdicts, baseline failures, and later task gates.
Depends only on the read-only reassessment and published/official authorities.

**First RED:** run a read-only documentation-contract check against the starting
commit. It must expose the stale Task 3/Task 5 ledger, wrong paper Proprio model,
mischaracterized preprint authority, and replay provenance/security scope before
any edit.

**Red-green-refactor:** (1) capture RED; (2) rewrite architecture around actual
production classes, state, tensor/update contracts, and source correspondence;
(3) rewrite this plan into bounded tasks; (4) reconcile the ledger; (5) remove
vague/contradictory/provenance-framework wording; (6) rerun the contract check.

**Focused acceptance:** the blocking gates are this current semantic/source
contract, the live migration/caller audits, `uv run ruff check .`, and
`git diff --check`. No tracked gate reads the ignored ledger/report or asserts a
prior package hash, verdict, reviewer identity, or attempt count.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run python - <<'PY_CONTRACT'
import hashlib
import json
import re
import subprocess
from pathlib import Path

architecture = Path("src/world_marl/dreamer_v3_baseline/ARCHITECTURE.md").read_text()
plan = Path("src/world_marl/dreamer_v3_baseline/PLAN.md").read_text()
fixture_path = Path("tests/fixtures/dreamer_v3/dm_control_1_0_17_state_schema.json")
fixture_bytes = fixture_path.read_bytes()
fixture = json.loads(fixture_bytes)
paper_revision = "bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01"
reference = "/private/tmp/danijar-dreamerv3-20260713"
restore_order = [
    "validate_closed_candidate",
    "construct_locked_task",
    "copy_complete_model_arrays",
    "mj_setState(INTEGRATION)",
    "mj_step1(legacy_step=True)",
    "restore_task_rng_and_mutable_task_fields",
    "restore_environment_counters_and_adapter_current_time_step",
    "clear_only_enumerated_derived_caches",
]
rendered_order = " -> ".join(restore_order)

assert fixture["format"] == "world_marl.dreamer_v3.dmc_state_schema"
assert fixture["format_version"] == 3
assert fixture["backend"]["legacy_step"] is True
assert len(fixture["canonical_task_order"]) == 20
assert fixture["restore_order"] == restore_order
assert len(fixture_bytes) == 102198
assert hashlib.sha256(fixture_bytes).hexdigest() == (
    "55c1b76180e0a811c96efd0742ff972e61d5424f5de41d3ae54ea641b141dbd7"
)
assert rendered_order in architecture
assert rendered_order in plan

def section(start, end=None):
    start_match = re.search(rf"(?m)^{re.escape(start)}", plan)
    assert start_match is not None, start
    if end is None:
        return plan[start_match.end():]
    end_match = re.search(rf"(?m)^{re.escape(end)}", plan[start_match.end():])
    assert end_match is not None, end
    return plan[start_match.end():start_match.end() + end_match.start()]

units = (
    "1a", "1b", "1c", "2", "2b", "3a", "3b", "3c", "3d", "4a", "4b",
    "4c", "4d", "5", "6a", "6b", "6c", "7a", "7b", "7c", "8a1",
    "8a2", "8b", "8c", "8d", "9a", "9b", "9c", "9d", "10", "11",
)
task_heading = re.compile(r"(?m)^(#{2,3}) Task ([0-9]+[a-z0-9]*):[^\n]*$")
heading_matches = list(task_heading.finditer(plan))
heading_by_unit = {match.group(2): match for match in heading_matches}
assert len(heading_by_unit) == len(heading_matches)

synthetic = 'label = "### Task 1a:"\n### Task 1a: Real heading\nbody\n'
synthetic_matches = list(task_heading.finditer(synthetic))
assert len(synthetic_matches) == 1
assert synthetic_matches[0].start() == synthetic.index("### Task 1a: Real heading")

expected_fence_counts = {unit: 1 for unit in units}
expected_fence_counts["1c"] = 3
unit_sections = {}
for unit in units:
    start_match = heading_by_unit[unit]
    later = [match.start() for match in heading_matches if match.start() > start_match.start()]
    end = min(later) if later else len(plan)
    body = plan[start_match.start():end]
    unit_sections[unit] = body
    count = len(re.findall(r"```(?:bash|shell)\n.*?\n```", body, re.S))
    assert count == expected_fence_counts[unit], (unit, count)

unit_order = {unit: index for index, unit in enumerate(units)}
creation_owner = {}
for unit, body in unit_sections.items():
    preamble = body.split("**First RED:**", 1)[0]
    if "create" not in preamble:
        continue
    for path in re.findall(r"`((?:src|tests)/[^`]+\.py)`", preamble):
        creation_owner.setdefault(path, unit)
for unit, body in unit_sections.items():
    commands = "\n".join(re.findall(r"```(?:bash|shell)\n(.*?)\n```", body, re.S))
    for path in re.findall(r"(?:src|tests)/[A-Za-z0-9_./*-]+\.py", commands):
        creator = creation_owner.get(path)
        if creator is not None:
            assert unit_order[creator] <= unit_order[unit], (unit, path, creator)

cache_export = 'export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"'
for fence in re.findall(r"```(?:bash|shell)\n(.*?)\n```", plan, re.S):
    if re.search(r"(?m)^\s*uv(?:\s|$)", fence):
        assert cache_export in fence

stems = (
    "paper-proprio-agent-loss", "paper-proprio-distributions",
    "paper-proprio-optimizer-five-step", "paper-proprio-replay",
    "paper-proprio-rssm", "paper-proprio-rssm-float32",
    "paper-proprio-world-loss", "paper-proprio-world-model",
    "paper-vision-networks", "paper-vision-networks-float32",
    "paper-vision-world-loss", "paper-vision-world-model",
    "upstream-current-proprio-agent-loss",
    "upstream-current-proprio-distributions",
    "upstream-current-proprio-optimizer-five-step",
    "upstream-current-proprio-replay", "upstream-current-proprio-rssm",
    "upstream-current-proprio-rssm-float32",
    "upstream-current-proprio-world-loss",
    "upstream-current-proprio-world-model",
    "upstream-current-vision-networks",
    "upstream-current-vision-networks-float32",
    "upstream-current-vision-world-loss",
    "upstream-current-vision-world-model",
)
task1b = section("### Task 1b:", "### Task 1c:")
manifest_refresh_stems = (
    "paper-proprio-distributions", "upstream-current-proprio-distributions",
    "paper-proprio-replay", "upstream-current-proprio-replay",
    "paper-proprio-rssm", "paper-proprio-rssm-float32",
    "upstream-current-proprio-rssm", "upstream-current-proprio-rssm-float32",
    "paper-vision-networks", "paper-vision-networks-float32",
    "upstream-current-vision-networks",
    "upstream-current-vision-networks-float32",
)
assert all(stem in task1b for stem in manifest_refresh_stems)
assert task1b.count("refresh-manifest") == 1
task3a = section("### Task 3a:", "### Task 3b:")
assert "--profile paper" in task3a and "--profile upstream-current" in task3a
for label, end_marker in (("3b", "### Task 3c:"), ("3c", "### Task 3d:")):
    body = section(f"### Task {label}:", end_marker)
    for profile in ("paper", "upstream-current"):
        for dtype in ("bfloat16", "float32"):
            assert f"--profile {profile}" in body
            assert f"--compute-dtype {dtype}" in body
task7 = section("## Task 7:", "## Task 8:")
task8c = section("### Task 8c:", "### Task 8d:")
task8d = section("### Task 8d:", "## Task 9:")
assert "test_report_writer_failure_cold_resume_replays" not in task7
assert "post-crossing artifact failure" not in task7
assert "test_report_writer_failure_cold_resume_replays" in task8c
assert "post-crossing artifact failure" in task8c
assert "tests/test_dreamer_v3_resume_trace.py" not in task8c
assert "create `tests/test_dreamer_v3_resume_trace.py`" in task8d
task9 = section("## Task 9:", "## Task 10:")
assert 's["writer_offsets"]["metrics"]==0' in task9
for body in (section("## Task 10:", "## Task 11:"), section("## Task 11:")):
    quoted = set(re.findall(r'"([a-z0-9-]+)"', body))
    assert set(stems) <= quoted
    assert "len(expected)==48" in body
    assert "dm_control_1_0_17_state_schema.json" in body

source_paths = set(re.findall(
    r"git -C /private/tmp/danijar-dreamerv3-20260713 show [0-9a-f]{40}:([^ >\n]+)",
    plan,
))
assert source_paths
for path in sorted(source_paths):
    subprocess.run(
        ["git", "-C", reference, "cat-file", "-e", f"{paper_revision}:{path}"],
        check=True,
    )
task2 = section("## Task 2:", "## Task 3:")
assert "DreamerReplay.can_sample_batch(mode)" in task2
assert "Task 2 creates and owns" in task2
task4a = section("### Task 4a:", "### Task 4b:")
task4b = section("### Task 4b:", "### Task 4c:")
task4c = section("### Task 4c:", "### Task 4d:")
assert "create the complete `AgentLoss` declaration" in task4a
assert "must not edit the `AgentLoss` declaration" in task4b
assert "must not edit the `AgentLoss` declaration" in task4c
task6a = section("### Task 6a:", "### Task 6b:")
task6b = section("### Task 6b:", "### Task 6c:")
task6c = section("### Task 6c:", "## Task 7:")
dependencies = (
    "[project.optional-dependencies].dmc",
    "dm-control==1.0.17",
    "mujoco==3.1.3",
    "uv.lock",
)
assert all(value in task6a for value in dependencies)
assert "sole owner of all dependency" in task6a
assert "create `DMCState`" in task6a and "TypedDict" in task6a
assert "owns no dependency" in task6b and "owns no dependency" in task6c
for body in (task6b, task6c):
    assert "edit `pyproject.toml" not in body
    assert "update `uv.lock`" not in body
assert all(value in task6c for value in dependencies[1:])
assert "uv sync --frozen" in task6c
task7b = section("### Task 7b:", "### Task 7c:")
assert "create the complete frozen `RunnerOutput` declaration" in task7b
task8b = section("### Task 8b:", "### Task 8c:")
assert "flax.serialization.msgpack_serialize" in task8b
assert "flax.serialization.msgpack_restore" in task8b
assert "owns no dependency" in task8b
for token in ("ml-dtypes==", "msgpack==", "max_map_len", "native-array-of-Value"):
    assert token not in task8b and token not in architecture
for token in (
    "representative complete payload",
    "tuple",
    "dataclass",
    "FrozenDict",
    "unsupported object",
):
    assert token in task8b, token
task8c = section("### Task 8c:", "### Task 8d:")
assert "create `RunnerRestoreCandidate` and `DreamerRunCoordinator`" in task8c
task7a = section("### Task 7a:", "### Task 7b:")
assert "DreamerRunner.from_state" in task7a
assert "factories" not in task8c
task9a = section("### Task 9a:", "### Task 9b:")
task9b = section("### Task 9b:", "### Task 9c:")
task9c = section("### Task 9c:", "### Task 9d:")
assert "invocation-local" in task9a
assert "evaluation.py" not in task9a
for body in (task9a, task9b, task9c):
    assert "--dmc-camera-id" in body
    assert "--camera" not in body
    assert "default=None" in body
    assert "omit `--dmc-camera-id`" in body
for body in (task9b, task9c):
    assert "--dreamer-debug-local" in body
    assert "quadruped" in body and "effective camera 2" in body
assert "debug_snapshot" in task9b and "config_sha256" in task9b
assert "runtime_overrides" in task9b
assert "reject" in task9b and "debug" in task9b
assert "ten primary" in task9c and "--debug-local" not in task9c.split(
    "**Focused acceptance:**", 1
)[0]

task4d = section("### Task 4d:", "## Task 5:")
task5 = section("## Task 5:", "## Task 6:")
task7a = section("### Task 7a:", "### Task 7b:")
task10 = section("## Task 10:", "## Task 11:")
task11 = section("## Task 11:")
task1a = section("### Task 1a:", "### Task 1b:")
task1c = section("### Task 1c:", "#### Integrated Task 1 regression inventory")
def words(value):
    return " ".join(value.split())

for token in (
    "`src/world_marl/dreamer_v3_baseline/__init__.py` only for the exact "
    "public-config migration",
    "add imports and `__all__` entries for exactly `DebugSnapshot`, "
    "`ResolvedDreamerRun`, `RuntimeOverrides`, `SequenceShapeConfig`, and "
    "`resolve_dreamer_run`",
    "imports the package root and proves that `ActorCriticConfig` is absent "
    "and the five replacement public config interfaces are present",
    "No compatibility alias, forwarding import, or deprecated export is retained",
):
    assert token in words(task1a), ("Task 1a package ownership", token)
for token in (
    "report `report_length=32, report_consecutive=1`",
    "`RunConfig` separately owns the source-derived `eval_envs=4` and `report_batches=1`",
    "`valnorm.debias=True`, `advnorm.debias=True`, and explicit `retnorm.debias=False`",
    "During review this unit remains uncommitted",
    "its focused GREEN is not a claim that the whole package is runnable",
):
    assert token in words(task1a), ("Task 1a corrected config contract", token)
for token in (
    "Task 1c owns `__init__.py` only for oracle/tooling imports and exports",
    "must not restore, alias, or otherwise edit `ActorCriticConfig`",
):
    assert token in words(task1c), ("Task 1c package ownership", token)
for token in (
    "Task 1a removes the `ActorCriticConfig` import and `__all__` entry and "
    "adds imports and `__all__` entries for exactly",
    "Task 1c later removes only eager oracle/tooling imports and exports",
    "Task 9 later replaces the remaining production exports",
):
    assert token in words(architecture), ("architecture package ownership", token)

replay_config_row = next(
    line for line in architecture.splitlines() if line.startswith("| `ReplayConfig` |")
)
dreamer_replay_row = next(
    line for line in architecture.splitlines() if line.startswith("| `DreamerReplay` |")
)
sequence_shape_row = next(
    line for line in architecture.splitlines() if line.startswith("| `SequenceShapeConfig` |")
)
run_config_row = next(
    line for line in architecture.splitlines() if line.startswith("| `RunConfig` |")
)
for token in (
    "report_length=T_report",
    "report_consecutive=C_report",
    "report_raw_length=K+T_report*C_report",
):
    assert token in sequence_shape_row, token
for token in ("eval_envs", "report_batches"):
    assert token in run_config_row, token
for token in (
    "Bool-as-int, int-as-float, NumPy scalar, list-for-tuple",
    "reconstructs the expected final config",
    "missing, extra, or disagreeing override",
):
    assert token in words(architecture), token
assert "selector seed" not in replay_config_row
assert "DreamerV3Config.seed" not in replay_config_row
assert "seed)" not in dreamer_replay_row.split("|", 3)[2]
for token in (
    "Fresh `DreamerReplay` constructs `UniformSelector(seed=0)` internally",
    "public run seed never enters replay construction",
    "complete PCG64 state",
    "exact next sample",
):
    assert token in architecture, token
for stale in (
    "selector construction seed, train/evaluation",
    "replay seed identity",
):
    assert stale not in architecture, stale
for label, body in (
    ("Task 1a", task1a),
    ("Task 2", task2),
    ("Task 5", task5),
    ("Task 8b", task8b),
    ("Task 8c", task8c),
    ("Task 9a", task9a),
):
    assert "identical fresh replay selector seed-0 state and sample sequence" in words(body), label
    assert "advanced selector state resumes at the exact next sample" in words(body), label
for label, body in (("Task 1a", task1a), ("Task 5", task5), ("Task 9a", task9a)):
    assert "distinct official parameter/counter roots and DMC derived identities" in words(body), label

dmc_architecture = architecture.split("## 9. DMC environment contract", 1)[1].split(
    "## 10.", 1
)[0]
for token in (
    "wm_marl_seedsequence_v1",
    "one bounded native environment-integration/reproducibility exception shared by both profiles",
    "not paper/current algorithm behavior",
    "not a translation of official DMC",
    "both pins omit `use_seed` for DMC",
    "`make_env` therefore forwards no seed",
    "official `DMC` accepts no seed",
    "calls `suite.load(domain, task)`",
    "locked dm-control creates `RandomState(None)` automatically",
    "DMCSpec remains the only serialized representation",
    "same explicit native child seed on both sides",
):
    assert token in words(dmc_architecture), token
assert "source-derived" not in dmc_architecture
for label, body in (("Task 6a", task6a), ("Task 6c", task6c), ("Task 9a", task9a), ("Task 10", task10)):
    assert "wm_marl_seedsequence_v1" in body, label
for token in (
    "named native seed formula",
    "role/child distinctness",
    "vector-count stability",
    "same-seed reproducibility",
    "vector `state_dict`/`from_states` exact continuation and lifecycle",
    "`expected_specs` mismatch is rejected before any child construction",
):
    assert token in words(task6c), ("Task 6c", token)
for future_owner in (
    "manifest/checkpoint identity",
    "CheckpointManager",
    "RunnerRestoreCandidate",
    "DreamerRunCoordinator",
    "cold-resume",
):
    assert future_owner not in task6c, ("Task 6c future owner", future_owner)
for token in (
    "pure payload identity/schema rejection",
    "both complete DMC specs",
    "named native policy `wm_marl_seedsequence_v1`",
    "before any resource construction",
):
    assert token in words(task8b), ("Task 8b", token)
for token in (
    "cold-resume expected-identity/spec/seed-policy mismatch",
    "`DreamerRunCoordinator.resume`",
    "`RunnerRestoreCandidate`",
    "before any child construction",
    "no cleanup is needed",
    "reverse cleanup",
):
    assert token in words(task8c), ("Task 8c", token)
assert "CLI, dry-run, and run-manifest DMC policy identity" in words(task9a)
assert "real online artifact/checkpoint inspection of the DMC policy identity" in words(task10)

resolver_signature = (
    "resolve_dreamer_run(*, mode, task, profile=DreamerProfile.PAPER, seed=0, model=None, "
    "debug_local=False, overrides=RuntimeOverrides())"
)
wrapper_signature = (
    "resolve_dreamer_config(*, mode, task, profile=DreamerProfile.PAPER, seed=0, model=None, "
    "debug_local=False)"
)
for token in (resolver_signature, wrapper_signature):
    assert token in words(architecture) and token in words(task1a), token
for stale_signature in (
    "resolve_dreamer_run" + "(profile, mode, task",
    "resolve_dreamer_config" + "(profile, mode, task",
):
    assert stale_signature not in architecture and stale_signature not in plan
for token in (
    "`inspect.signature` proves `mode` and `task` are required keyword-only parameters",
    "`profile` is keyword-only with default `DreamerProfile.PAPER`",
    "omitting `profile` is byte/config/hash-exactly equal to explicit `DreamerProfile.PAPER`",
    "explicit `DreamerProfile.UPSTREAM_CURRENT` differs",
    "positional calls fail with `TypeError`",
    "the wrapper forwards the same omitted-paper and explicit-current behavior",
):
    assert token in words(task1a), token
wrapper_call = (
    "resolve_dreamer_run(mode=mode, task=task, profile=profile, seed=seed, "
    "model=model, debug_local=debug_local, overrides=RuntimeOverrides())"
)
task9_call = (
    "resolve_dreamer_run(mode=parsed.observation_mode, task=parsed.task, "
    "profile=parsed.profile, seed=parsed.seed, model=parsed.model, "
    "debug_local=parsed.debug_local, overrides=runtime_overrides)"
)
assert wrapper_call in words(architecture) and wrapper_call in words(task1a)
assert task9_call in words(task9a)
assert 'parser.add_argument("--profile", choices=("paper", "upstream-current"), default="paper")' in task9a
task9a_dry = next(
    line for line in task9a.splitlines()
    if "world-marl-train-dreamer-v3-baseline" in line and '"$TASK9A_ROOT/dry"' in line
)
assert "--profile" not in task9a_dry
for token in (
    "the omitted-profile dry run resolves profile `paper`",
    "authority revision `bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01`",
    "canonical argv may explicitly include normalized `--profile paper`",
):
    assert token in words(task9a), token
for token in (
    "DreamerV3Config.seed is the sole canonical public-seed owner",
    "Seed is a primary resolver input, not a `RuntimeOverrides` field",
    "non-bool Python integer",
    "0 <= seed <= 2**32 - 1 - 10_000",
    "before NumPy conversion",
    "canonical resolved config JSON/hash",
    "ResolvedDreamerRun.identity_state()",
):
    assert token in words(architecture) and token in words(task1a), token
assert "`seed`" not in re.search(
    r"`RuntimeOverrides` \|([^\n]+)", architecture
).group(1)
for token in (
    "two distinct nondefault seeds",
    "bool",
    "-1",
    "2**32 - 10_000",
    "wrapper forwards `seed`",
):
    assert token in words(task1a), token

driver_trust_sources = (
    "`run_config` supplies train/evaluation vector sizes and cadences; "
    "`sequence_shape` supplies `B/K/T/consecutive`; the trusted spaces supply "
    "row/action leaves; and the bound agent/train resolved identity supplies "
    "the canonical seed when later runner calls derive roots"
)
assert driver_trust_sources in words(architecture) and driver_trust_sources in words(task7a)
assert "`RunConfig` owns no seed or batch/sequence shape" in words(architecture)
assert "`RunConfig` owns no seed or batch/sequence shape" in words(task7a)
assert "duplicate DriverState seed leaf" in words(architecture)
assert "duplicate DriverState seed leaf" in words(task7a)
for stale in (
    "run config closes public seed, batch and",
    "`run_config` supplies public seed, batch size",
):
    assert stale not in architecture and stale not in task7a, stale

for body in (task5, task7a, task8b, task8c, task9a):
    assert "resolved.config.seed" in words(body)
assert "two nondefault seeds" in words(task5)
assert "distinct official parameter/counter roots" in words(task5)
assert "seed mismatch fails before construction" in words(task7a)
assert "seed mismatch fails before construction" in words(task8b)
assert "seed mismatch fails before construction" in words(task8c)
assert "passes parsed `--seed` through the resolver" in words(task9a)
for token in (
    "fresh bootstrap",
    "parameter initialization",
    "official counter roots",
    "native train/evaluation `DMCSpec`s",
    "canonical argv",
    "manifest",
    "checkpoint identity",
    "cold-resume expected identity",
):
    assert token in words(task9a), token

policy_signature = (
    "DreamerAgent.policy(params, carry, observation, mode, outer_seed)"
)
assert architecture.count(policy_signature) >= 2
assert policy_signature in words(task4d)
assert "policy(params, carry, obs, rng, mode)" not in architecture
initialize_signature = (
    "DreamerTrainState.initialize(agent, observation_spaces, action_spaces, "
    "resolved_config)"
)
schema_signature = (
    "DreamerTrainState.schema(agent, observation_spaces, action_spaces, "
    "resolved_config)"
)
train_step_signature = "train_step(agent, state, carry, batch, outer_seed)"
for token in (initialize_signature, schema_signature, train_step_signature):
    assert token in architecture and token in task5, token
assert "calls `DreamerTrainState.initialize` exactly once" in task9a
assert "calls `DreamerTrainState.initialize` zero times" in task8c
assert "raw parameter seed `uint32([resolved_config.seed, 0])`" in architecture
assert "zero complete observation/action/extra data `[B,T+K,...]`" in architecture
assert "DreamerTrainStateSchema" in architecture and "DreamerTrainStateSchema" in task5

inverse_signatures = {
    "ReplayBatch.from_state(state, transition_spaces, latent_spaces, expected_batch_size, expected_time_length)": task2,
    "PercentileNormalizerState.from_state(state, config)": task4a,
    "SlowValueState.from_state(state, online_critic_params, config)": task4a,
    "AgentCarry.from_state(state, agent, expected_leading_shape)": task2,
    "DreamerOptimizerState.from_state(state, params, config)": task5,
    "DriverState.from_state(state, agent, run_config, sequence_shape, observation_spaces, action_spaces)": task7a,
}
for signature, owner in inverse_signatures.items():
    assert signature in architecture, signature
    assert signature in owner, signature
assert "ReplayBatch.from_state" in task7a and "ActiveReport" in task7a
assert "ReplayBatch.from_state" in task2 and "ConsecutiveStream" in task2
for owner in inverse_signatures.values():
    assert "from_state(state, " + "dependencies)" not in owner
    assert "generic " + "dependencies bag" not in owner

official_seed_formula = (
    "default_rng([public_seed, int(counter)]).integers("
    "0, np.iinfo(np.uint32).max, (2,), np.uint32)"
)
assert official_seed_formula in architecture and official_seed_formula in task5
for token in ("policy_call_counter", "batch_seed_counter"):
    assert token in architecture and token in task7a and token in task8c and token in task9a
for stale in (
    "next_root_" + "train_rng",
    "root_" + "train_rng",
    "collection_" + "rng",
    "report_" + "rng",
    "evaluation_" + "rng",
):
    assert stale not in architecture, stale
assert "post-call Ninjax context remainder is discarded" in architecture
assert "collection -> evaluation -> train -> report -> checkpoint -> resume" in task5
assert "second-root" in task5

for body in (architecture, task4d, task9a, task10, task11):
    assert "Vision open-loop cursor/files are nonempty" in body
    assert "Proprio open-loop cursor is zero" in body
    assert "Proprio open-loop directory is absent" in body
    assert "evaluation cursor/files are nonempty in both modalities" in body
assert 's["next_file_cursors"]["open_loop"]==0' in task9a
assert 's["next_file_cursors"]["open_loop"]>0' in task9a

owner_sections = {
    "config": section("### Task 1a:", "### Task 1b:"),
    "replay": section("## Task 2:", "## Task 3:"),
    "world": section("## Task 3:", "## Task 4:"),
    "normalization": section("### Task 4a:", "### Task 4b:"),
    "train": section("## Task 5:", "## Task 6:"),
    "dmc_vector": task6c,
    "driver": task7a,
    "artifacts": section("### Task 8a1:", "### Task 8a2:"),
}
for label, body in owner_sections.items():
    assert "primitive" in body and "state" in body, label
assert "DreamerTrainState.state_dict" in owner_sections["train"]
assert "DreamerTrainState.from_state" in owner_sections["train"]
assert "list[DMCState]" in owner_sections["dmc_vector"]
assert "tuple[DMCState" not in architecture
assert "DMCEnvironment.restore_state" not in architecture
assert "DMCVectorEnvironment.restore_states" not in architecture
task2b = section("### Task 2b:", "## Task 3:")
assert '`{"index": int, "item_id": int}`' in task2b
print("GREEN structural Task 0 contract")
PY_CONTRACT
uv run pytest tests/test_dreamer_v3_dmc_state_inventory.py -q
uv run python - <<'PY_SOURCE'
import ast
import inspect
import importlib.metadata
import re
import subprocess
from pathlib import Path

import optax

reference = "/private/tmp/danijar-dreamerv3-20260713"
revision = "bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01"
current_revision = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"

def show(path):
    return subprocess.check_output(
        ["git", "-C", reference, "show", f"{revision}:{path}"],
        text=True,
    )

def show_at(source_revision, path):
    return subprocess.check_output(
        ["git", "-C", reference, "show", f"{source_revision}:{path}"],
        text=True,
    )

def function(source, name):
    return next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )

def initializer(source, class_name):
    class_node = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

def defaults(node):
    args = node.args.args[-len(node.args.defaults):] if node.args.defaults else []
    return {
        arg.arg: ast.literal_eval(value)
        for arg, value in zip(args, node.args.defaults)
    }

for source_revision in (revision, current_revision):
    main_source = show_at(source_revision, "dreamerv3/main.py")
    config_source = show_at(source_revision, "dreamerv3/configs.yaml")
    replay_source = show_at(source_revision, "embodied/core/replay.py")
    selector_source = show_at(source_revision, "embodied/core/selectors.py")
    dmc_source = show_at(source_revision, "embodied/envs/dmc.py")

    make_replay = function(main_source, "make_replay")
    assert not any(
        isinstance(node, ast.Attribute) and ast.unparse(node) == "config.seed"
        for node in ast.walk(make_replay)
    )
    replay_calls = [
        node for node in ast.walk(make_replay)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "embodied.replay.Replay"
    ]
    assert len(replay_calls) == 1
    assert not any(keyword.arg == "seed" for keyword in replay_calls[0].keywords)
    for selector_name in ("Uniform", "Prioritized", "Recency", "Mixture"):
        selector_calls = [
            node for node in ast.walk(make_replay)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func).endswith(f".{selector_name}")
        ]
        assert len(selector_calls) == 1, (source_revision, selector_name)
        assert not any(
            keyword.arg == "seed"
            for keyword in selector_calls[0].keywords
        ), (source_revision, selector_name)

    replay_init = initializer(replay_source, "Replay")
    assert defaults(replay_init)["seed"] == 0
    uniform_from_replay = [
        node for node in ast.walk(replay_init)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "selectors.Uniform"
    ]
    assert len(uniform_from_replay) == 1
    assert [ast.unparse(arg) for arg in uniform_from_replay[0].args] == ["seed"]
    for selector_name in ("Uniform", "Prioritized", "Recency", "Mixture"):
        assert defaults(initializer(selector_source, selector_name))["seed"] == 0

    dmc_config = re.search(r"(?m)^\s+dmc:\s*\{([^}]*)\}$", config_source)
    assert dmc_config is not None
    assert "use_seed" not in dmc_config.group(1)
    make_env = function(main_source, "make_env")
    use_seed_test = [
        node for node in ast.walk(make_env)
        if isinstance(node, ast.If)
        and "kwargs.pop('use_seed', False)" in ast.unparse(node.test)
    ]
    assert len(use_seed_test) == 1
    dmc_init = initializer(dmc_source, "DMC")
    assert "seed" not in [arg.arg for arg in dmc_init.args.args]
    suite_load_calls = [
        node for node in ast.walk(dmc_init)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "suite.load"
    ]
    assert len(suite_load_calls) == 1
    assert [ast.unparse(arg) for arg in suite_load_calls[0].args] == ["domain", "task"]
    assert not suite_load_calls[0].keywords

assert importlib.metadata.version("dm-control") == "1.0.17"
dm_control_dist = importlib.metadata.distribution("dm-control")
suite_source = Path(dm_control_dist.locate_file("dm_control/suite/__init__.py")).read_text()
base_source = Path(dm_control_dist.locate_file("dm_control/suite/base.py")).read_text()
build_environment = function(suite_source, "build_environment")
assert any(
    isinstance(node, ast.Assign)
    and [ast.unparse(target) for target in node.targets] == ["task_kwargs"]
    and ast.unparse(node.value) == "task_kwargs or {}"
    for node in ast.walk(build_environment)
)
task_init = initializer(base_source, "Task")
assert defaults(task_init)["random"] is None
random_state_calls = [
    node for node in ast.walk(task_init)
    if isinstance(node, ast.Call) and ast.unparse(node.func) == "np.random.RandomState"
]
assert len(random_state_calls) == 1
assert [ast.unparse(arg) for arg in random_state_calls[0].args] == ["random"]
print("GREEN pinned replay/DMC construction audit at both revisions")

opt = show("embodied/jax/opt.py")
agent = show("dreamerv3/agent.py")
outer_agent = show("embodied/jax/agent.py")
rssm = show("dreamerv3/rssm.py")
assert "self.scaling = (nets.COMPUTE_DTYPE == jnp.float16)" in opt
assert re.search(r"if self\.scaling:\n\s+self\.opt = optax\.apply_if_finite", opt)
for value in (
    "self.opt.update",
    "optax.apply_updates",
    "state.write(new_state)",
    "self.step.write(self.step.read() + 1)",
):
    assert value in opt
train = agent.split("  def train(self, carry, data):", 1)[1].split("  def loss(", 1)[0]
assert train.index("self.opt(") < train.index("self.slowval.update()")
assert train.index("self.slowval.update()") < train.index("outs['replay'] = updates")
make_opt = agent.split("  def _make_opt(", 1)[1]
assert make_opt.index("optax.add_decayed_weights(wd, wdmask)") < make_opt.index(
    "optax.scale_by_learning_rate(sched)"
)
assert inspect.signature(optax.scale_by_learning_rate).parameters["flip_sign"].default is True
for value in (
    ".max((2, 4))",
    "'... (g h w c) -> ... h w (g c)'",
    "jax.nn.sigmoid(x)",
    "outs.Agg(out, 3, jnp.sum)",
    "cand = jnp.tanh(reset * cand)",
    "update = jax.nn.sigmoid(update - 1)",
):
    assert value in rssm
for value in (
    "np.random.default_rng(seed=[self.config.seed, int(counter)])",
    "rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)",
    "self.n_batches = elements.Counter()",
    "self.n_actions = elements.Counter()",
    "counter = self.n_actions.value",
    "counter = self.n_batches.value",
    "np.array([self.config.seed, 0], np.uint32)",
    "self.model.init_train",
    "self.model.train",
):
    assert value in outer_agent, value
assert "for key in self.dec.imgkeys:" in agent
print("GREEN pinned source audit: optimizer/train/RNG/Vision/BlockGRU ordering")
PY_SOURCE
uv run python - <<'PY_AST'
import ast
import re
from pathlib import Path

root = Path("src/world_marl/dreamer_v3_baseline")
architecture = (root / "ARCHITECTURE.md").read_text()
section = architecture.split(
    "### 3.1 Complete live-symbol migration inventory", 1
)[1].split("#### Legacy import-site inventory", 1)[0]
rows = {
    re.fullmatch(r"`(.+)`", cells[0]).group(1): cells[3]
    for line in section.splitlines()
    if line.startswith("| `")
    for cells in [[cell.strip() for cell in line.strip().strip("|").split("|")]]
}
package_modules = {path.stem for path in root.glob("*.py")}
paths = sorted(root.glob("*.py")) + [
    Path("src/world_marl/scripts/train_dreamer_v3_baseline.py")
]
missing = set()
for path in paths:
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        module = node.module
        if not (
            module.startswith("world_marl.dreamer_v3_baseline.")
            or (node.level and module in package_modules)
        ):
            continue
        source_module = module.rsplit(".", 1)[-1] + ".py"
        site = path.name
        token = (
            "E"
            if site == "__init__.py"
            else "C"
            if site == "train_dreamer_v3_baseline.py"
            else "O"
            if "oracle" in site
            else site
        )
        for alias in node.names:
            key = f"{source_module}::{alias.name}"
            if key in rows and token not in [
                item.strip() for item in rows[key].split(",")
            ]:
                missing.add((key, site, token))
assert not missing, sorted(missing)
print("GREEN live production caller inventory")
PY_AST
uv run ruff check .
git diff --check
```

The known global formatting debt is a separately asserted nonpassing baseline;
it is not part of the passing `set -e` block:

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
FORMAT_LOG="$(mktemp /tmp/wm-marl-dreamer-v3-task0-format.XXXXXX)"
set +e
uv run ruff format --check . >"$FORMAT_LOG" 2>&1
format_status=$?
set -e
test "$format_status" -eq 1
uv run python -c 'from pathlib import Path; import re,sys; actual=Path(sys.argv[1]).read_text(); lines=actual.splitlines(); assert lines==["Would reformat: src/world_marl/evaluation.py",lines[-1]],lines; assert re.fullmatch(r"1 file would be reformatted, [0-9]+ files already formatted",lines[-1]),lines; print(actual,end="")' "$FORMAT_LOG"
```

Do not spend seven minutes rerunning unchanged component suites.

**Report/review/commit:** report all required controller fields; fresh spec
review checks every documentation requirement and authority value; fresh
quality review checks internal consistency and executable task boundaries.
Fix/re-review to Critical=0 and Important=0, then commit
`docs(dreamer): correct implementation contract`.

## Task 0b: Freeze the locked DMC state design before production edits

Task 0 cannot be accepted or committed until this unit and its fresh reviews
are clean. No production task may start until its literal architecture tables,
structured canonical fixture, isolated executable proof, and reviews are
accepted.

**Owned files/boundary:** create
`tests/test_dreamer_v3_dmc_state_inventory.py` and
`tests/dreamer_v3_dmc_state_worker.py`, and create
`tests/fixtures/dreamer_v3/dm_control_1_0_17_state_schema.json`; edit only the
DMC state section in `ARCHITECTURE.md`, this Task-0b plan section, and the
ignored Task-0 ledger/report. No production module, `pyproject.toml`, or
`uv.lock` change is owned.

**Historical and strengthened RED:** the first test failed because the fixture
did not exist. The r14 isolation RED proved importing the pytest parent changed
`MUJOCO_GL` from absent to `off`. The validator RED proved a fractional MT19937
position `1.5` combined with a changed integration time was accepted and
mutated live state. The structural RED was `0/7` for missing child isolation,
DMCSpec schema, `legacy_step`, action ownership, valid source paths, writable
cache setup, and dependency ownership.

The pytest parent stays renderer-neutral: it neither sets `MUJOCO_GL` nor
imports dm_control, MuJoCo, GLFW, or a renderer. All locked state probing lives
in the non-collected child worker and runs in a separate process with
`MUJOCO_GL=off`. The parent snapshots `os.environ` and renderer-module
membership before/after every child. This is state-only evidence; Task 6c and
Task 10 run Vision separately with a validated render-capable backend.

The fixture is a closed structured schema, not type prose. It defines exact
immutable `DMCSpec` task mapping, profile/mode, public/vector/child seed
identity, `[64,64]` image size, nullable camera override/effective camera,
repeat, and backend. It freezes `physics.legacy_step is true`; exact
runtime-versus-serialized types for Python counters/timeouts, reset bool,
`StepType`, nullable reward/discount, observations, and all MT19937 leaves;
specs, sensor gate, task/model/cache fields, and integration profiles. A bool
never satisfies an integer leaf. Integration `ctrl` plus the driver's pending
environment action are the sole behavioral action owners; no redundant raw,
clipped, or native action state exists in Task 0b or Task 6b.

`Physics.get_state()` is explicitly insufficient. Dynamically validate and
record `mjtState.mjSTATE_INTEGRATION`, numeric spec `8191`, and the result of
`mj_stateSize`; capture/restore with `mj_getState`/`mj_setState`. The schema
identifies time, physics, warmstart, control, applied-force, equality-active,
mocap, userdata, and plugin state separately from model arrays. The restore
order in fixture, architecture, Task 0b, Task 6b, and executable trace is:

`validate_closed_candidate -> construct_locked_task -> copy_complete_model_arrays -> mj_setState(INTEGRATION) -> mj_step1(legacy_step=True) -> restore_task_rng_and_mutable_task_fields -> restore_environment_counters_and_adapter_current_time_step -> clear_only_enumerated_derived_caches`.

The pure DMCSpec matrix checks the train/evaluation base-seed offset and child
`SeedSequence([base_seed, child_index])` formula without loading extra
environments. It covers both profiles/modes/roles, seed and child-index
endpoints, and non-quadruped/quadruped default and legal-override cameras; its
invalid table covers exact types, ranges, task mapping, role, base/child seed,
evaluation offset, and effective-camera derivation. Actual cold-resume
rejection remains a later production Task-8 boundary, not Task-0b evidence.
This unchanged schedule is named exactly `wm_marl_seedsequence_v1`. It is one
bounded native environment-integration/reproducibility exception shared by both
profiles, not paper/current algorithm behavior or a translation of official
DMC. The existing public/base/child fields are its only representation; Task 0b
adds no policy field, second seed owner, codec, provenance framework, or fixture
field.

**Implementation and review:** generate twice and compare bytes/SHA; compare
every architecture integration/task table cell with fixture structure; and
validate all candidate leaves before construction or mutation. Corrupt every
schema family with an independently frozen ordered corruption tuple, including
exact structured DMCSpec element types, per-task camera bounds, static task
fields, finger model state, quadruped caches, FIRST/MID/LAST cross-field
invariants, and changed integration plus fractional RNG position. The forbidden
loader remains uncalled and the source is recaptured unchanged after every
rejection. Recursive fixture tests require each encoded scalar dtype/shape to
equal its declared serialized schema. `_timeout_progress` is omitted because
the locked classes reset but never read it. The real locked worker exercises all
20 tasks at MID under the one explicitly labeled identity
`paper`/`proprio`/`train`, public seed 7, child index 0, and no camera override.
It then reaches a genuine native time-limit LAST through
ordinary stepping, compares the current `TimeStep` and complete recaptured state
after restore, and compares the following genuine FIRST/reset state. It never
manufactures a pending-reset MID. Production `DMCState` contains no fixture
path, digest, or provenance leaf; the fixture digest is only generation/test
evidence. The schema's frozen corruption tuple has exactly 41 families after
removing the former fixture-identity case. The fixture is 102198 bytes with
SHA-256 `55c1b76180e0a811c96efd0742ff972e61d5424f5de41d3ae54ea641b141dbd7`.
Task 0/0b remains unaccepted until fresh spec and quality reviews both report
zero Critical and zero Important.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv sync --frozen --extra dev --extra dmc
uv run python -c 'import importlib.metadata, mujoco; assert importlib.metadata.version("dm-control") == "1.0.17"; assert mujoco.__version__ == "3.1.3"'
uv run pytest tests/test_dreamer_v3_dmc_state_inventory.py -q
MUJOCO_GL=off uv run python tests/dreamer_v3_dmc_state_worker.py verify --fixture tests/fixtures/dreamer_v3/dm_control_1_0_17_state_schema.json
uv run ruff check tests/test_dreamer_v3_dmc_state_inventory.py tests/dreamer_v3_dmc_state_worker.py
uv run ruff format --check tests/test_dreamer_v3_dmc_state_inventory.py tests/dreamer_v3_dmc_state_worker.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/envs/dmc.py >/dev/null
```

**Commit after both Task 0 and Task 0b are accepted:**
`docs(dreamer): freeze implementation and dmc state contracts`.

## Task 1: Minimize oracle boundary and correct profiles

Task 1 is three sequential implementer-review-fixer-commit units. Their symbol
ownership does not overlap, except that Task 1a and Task 1c sequentially edit
different named imports/exports in `__init__.py`; later units otherwise consume
earlier public interfaces, and editors never run concurrently. The aggregate
inventory and regression matrix below are cross-unit reference, not permission
for one large implementation.

### Task 1a: Resolve typed profiles and configuration

**Owned files/symbols:** `config.py`, `tests/test_dreamer_v3_config.py`, and
`src/world_marl/dreamer_v3_baseline/__init__.py` only for the exact
public-config migration: remove the `ActorCriticConfig` import and its
`__all__` entry, and add imports and `__all__` entries for exactly
`DebugSnapshot`, `ResolvedDreamerRun`, `RuntimeOverrides`,
`SequenceShapeConfig`, and `resolve_dreamer_run`. Own all typed config
dataclasses/enums, canonical serialization/hash, and `SequenceShapeConfig`,
`RuntimeOverrides`, `DebugSnapshot`, `resolve_dreamer_run`, and the no-override
`resolve_dreamer_config` wrapper. No other package-root import or export is
owned: do not edit oracle/runtime-import code or fixtures.

Freeze the exact public signatures
`resolve_dreamer_run(*, mode, task, profile=DreamerProfile.PAPER, seed=0, model=None, debug_local=False, overrides=RuntimeOverrides())`
and
`resolve_dreamer_config(*, mode, task, profile=DreamerProfile.PAPER, seed=0, model=None, debug_local=False)`.
The wrapper calls
`resolve_dreamer_run(mode=mode, task=task, profile=profile, seed=seed, model=model, debug_local=debug_local, overrides=RuntimeOverrides())`.
All arguments are keyword-only: `mode` and `task` remain required, while
`profile` is genuinely omittable and defaults to `DreamerProfile.PAPER`.
DreamerV3Config.seed is the sole
canonical public-seed owner for model initialization, model-call outer roots,
and the named native DMC seed policy. Replay construction is independent and
always uses the official constant zero. Seed is a primary resolver input, not a
`RuntimeOverrides` field. Copies in canonical argv, manifests, `DMCSpec`,
summaries, and checkpoint identity are derived equality-checked projections,
not mutable owners.

Task 1a also owns the checkpoint-facing primitive projections for every named
config/runtime/debug record: exact plain-mapping `state_dict()` and closed
`from_state()` inverses, plus `ResolvedDreamerRun.identity_state()` containing
canonical config, `config_sha256`, authority revision, nullable debug snapshot,
and the separate algorithm/environment runtime-override mappings. REDs require
fresh dict/list/array ownership, exact key/type reconstruction, public-Flax
roundtrip, and rejection of tuple, dataclass, enum, and `FrozenDict` leaves.
No generic object converter is introduced. Every accepted public constructor
enforces the same exact Python primitive, finiteness, tuple-element, and
nested-record rules as its inverse. Table-drive bool-as-int, int-as-float,
NumPy integer/float scalars, list/tuple mismatch, nonfinite floats, wrong
primitive types, and `type(record).from_state(record.state_dict()) == record`
for every named constructible record.

The scalar `seed` is included in canonical resolved config JSON/hash and
`ResolvedDreamerRun.identity_state()`. Validate it as a non-bool Python integer
satisfying `0 <= seed <= 2**32 - 1 - 10_000` before NumPy conversion. Reject
NumPy integer scalars as well as Python `bool`.

**First RED:** snapshot all four profile/mode model resolutions and every
paper/current scalar. `inspect.signature` proves `mode` and `task` are required
keyword-only parameters, `profile` is keyword-only with default
`DreamerProfile.PAPER`, and all remaining parameters are keyword-only with the
frozen defaults. Calling either public API while omitting `profile` is
byte/config/hash-exactly equal to explicit `DreamerProfile.PAPER`; explicit
`DreamerProfile.UPSTREAM_CURRENT` differs, positional calls fail with
`TypeError`, and the wrapper forwards the same omitted-paper and
explicit-current behavior. Reject legacy constructor/generic source-map fields. Resolve two
distinct nondefault seeds and prove distinct stable canonical JSON/hash and
identity projections, distinct official parameter/counter roots and DMC derived identities,
but identical fresh replay selector seed-0 state and sample sequence. A direct
component oracle advances and restores the complete PCG64 record to prove the
advanced selector state resumes at the exact next sample. Prove `ReplayConfig`
has no seed field and `ResolvedDreamerRun.identity_state()` has no replay/public
seed equality projection. Prove the no-override wrapper forwards `seed`; reject
`bool`, `-1`, and `2**32 - 10_000` before construction or NumPy conversion.
The same RED imports the package root and proves that `ActorCriticConfig` is
absent and the five replacement public config interfaces are present in both
attributes and `__all__`; it initially fails because the live package imports
and exports the legacy class but none of the five replacement interfaces.
GREEN deletes the class and its config property/serialization path, removes
exactly its two package-root references, and adds exactly the five named
imports and five `__all__` entries in one atomic unit. No compatibility alias,
forwarding import, or deprecated export is retained; no other package-root edit
is permitted.
Require `SequenceShapeConfig` to be the sole owner of train
`B,T,K,consecutive` and report `report_length=32, report_consecutive=1`, with
derived train `raw_length=65` and report `report_raw_length=33` in production.
`RunConfig` separately owns the source-derived `eval_envs=4` and
`report_batches=1`. Table-drive
every Task-9 identity-bearing override (`env_steps`, `num_envs`, `batch_size`,
`batch_length`, `train_ratio`, `eval_every`, `eval_episodes`, `report_every`,
`checkpoint_every`, and environment-only `camera`) through `RuntimeOverrides`;
assert the exact merge order,
cross-field revalidation, canonical JSON/hash and sorted override map. Reject
unknown fields. Assert invocation-only out-dir/resume/dry-run/stop fields are
absent; camera is returned only in the
environment override map and is absent from Dreamer config/hash. Snapshot every
value of deterministic noncanonical `debug-local-v1`, including the literal
48-frame CPU resolution, `report_length=4`, `report_consecutive=1`,
`eval_envs=1`, and `report_batches=1`. Both production
profiles resolve `RunConfig.log_every=1_000` as a positive physical-frame
cadence; `debug-local-v1` resolves `log_every=16`. This repository-native
decision does not convert upstream's wall-clock setting and is not a public CLI
override.
The resolved run state therefore owns the log next threshold. At each crossing,
the driver emits `metric_means` and `metric_counts`, then flushes and resets the
window; the debug acceptance trace observes this at frame 16.

The locked-profile RED mutates at least one leaf in every component family and
requires rejection of every non-overridable leaf. A resolved-identity RED
reconstructs the final config from profile/mode/task/seed/base model, nullable
exact debug snapshot, and explicit overrides, then rejects every component
family patch, debug presence/value mismatch, and missing, extra, or disagreeing
override. This reconstruction is pure, nonrecursive, and independent of the
canonical hash self-check. Source-derived normalizer expectations are
`valnorm.debias=True`, `advnorm.debias=True`, and explicit
`retnorm.debias=False`, using `configs.yaml` plus
`embodied/jax/utils.py::Normalize`.

**Implementation:** translate the two resolved snapshots from the Nature
profile plus pinned `configs.yaml`, delete legacy config branches, keep one
canonical JSON/hash path, and make Task 9 only parse values and call the Task-1
resolver. Apply only the exact public-config migration at the package root:
remove the `ActorCriticConfig` import and `__all__` entry and add the five named
replacement imports and `__all__` entries; leave oracle/tooling cleanup to Task
1c and later production export replacement to Task 9. `ReplayConfig` and
`RunConfig` do not duplicate the shape fields.
**Focused acceptance:**
`uv run pytest -q tests/test_dreamer_v3_config.py` followed by scoped Ruff check
and format-check of those three files. Report RED/GREEN, exact scalar/source
mapping, and concerns; fresh spec and quality reviewers plus a fresh covering
fixer must reach 0 Critical/Important before
`fix(dreamer): align conformance profiles`.

During review this unit remains uncommitted, and its focused GREEN is not a
claim that the whole package is runnable: Task 1b owns the positional oracle
resolver migration and Task 2 owns replay construction against the new
sequence/replay split. No compatibility alias hides those explicit sequential
owners; a later accepted per-task commit still carries no whole-package parity
claim until the composed owners land.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest tests/test_dreamer_v3_config.py -q
uv run ruff check src/world_marl/dreamer_v3_baseline/config.py src/world_marl/dreamer_v3_baseline/__init__.py tests/test_dreamer_v3_config.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/config.py src/world_marl/dreamer_v3_baseline/__init__.py tests/test_dreamer_v3_config.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/configs.yaml >/dev/null
```

### Task 1b: Simplify the numerical oracle and fixture root

**Owned files/symbols:** `oracle.py`, `network_oracle.py`, `rssm_oracle.py`,
`replay_oracle.py`, `replay_oracle_contract.py`, new `fixture_generator.py`
limited to the stable root/registry/refresh symbols named in the aggregate
inventory, `__init__.py` only to remove the dead `OracleHarness` and
`OracleInvocation` imports and `__all__` entries when their definitions are
deleted, `tests/test_dreamer_v3_oracle_manifest.py`,
the obsolete manifest/request/generator assertions and imports in
`tests/test_dreamer_v3_distributions.py`, `tests/test_dreamer_v3_networks.py`,
and `tests/test_dreamer_v3_rssm_parity.py` that must follow Task 1b's compact
manifest schema (without changing any distribution, network, or RSSM equation,
tolerance, or numerical assertion),
`tests/fixtures/dreamer_v3/README.md`, and the twelve manifest-only paths named
below. No NPZ or runtime-package file is owned.

**First RED:** reject callback/source-text/interpreter authentication, require
fixture-only pinned blob hashes and a bijective parameter translator, and fail
the absent refresh parser. **Implementation:** replace the private-runtime
emulation with direct read-only `git show` translation, immutable private
source-hash/dtype lookup tables, deterministic NPZ/manifest helpers that use
exclusive random temporary files and a random staging directory, the stable
decorated parser registry, and manifest-only refresh. `OracleManifest` and
`fixture_generator.py` must not call the transitional `OracleSourceSpec`
registry. **Focused acceptance:** run the one oracle-manifest test file's owned
tests, all twelve literal refresh commands below, assert every NPZ remains
byte-identical, then scoped Ruff check/format. Report every digest; fresh spec/
quality review and fresh fixer must reach 0 Critical/Important before
`refactor(dreamer): simplify parity fixtures`.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
NPZ_BEFORE="$(mktemp /tmp/wm-marl-dreamer-task1b-npz.XXXXXX)"
uv run python -c 'from pathlib import Path; import hashlib,json,sys; stems=["paper-proprio-distributions","upstream-current-proprio-distributions","paper-proprio-replay","upstream-current-proprio-replay","paper-proprio-rssm","paper-proprio-rssm-float32","upstream-current-proprio-rssm","upstream-current-proprio-rssm-float32","paper-vision-networks","paper-vision-networks-float32","upstream-current-vision-networks","upstream-current-vision-networks-float32"]; root=Path("tests/fixtures/dreamer_v3"); Path(sys.argv[1]).write_text(json.dumps({s:hashlib.sha256((root/(s+".npz")).read_bytes()).hexdigest() for s in stems},sort_keys=True))' "$NPZ_BEFORE"
for fixture in \
  "paper|proprio|bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01|paper-proprio-distributions" \
  "upstream-current|proprio|e3f02248693a79dc8b0ebd62c93683888ddaccfe|upstream-current-proprio-distributions" \
  "paper|proprio|bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01|paper-proprio-replay" \
  "upstream-current|proprio|e3f02248693a79dc8b0ebd62c93683888ddaccfe|upstream-current-proprio-replay" \
  "paper|proprio|bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01|paper-proprio-rssm" \
  "paper|proprio|bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01|paper-proprio-rssm-float32" \
  "upstream-current|proprio|e3f02248693a79dc8b0ebd62c93683888ddaccfe|upstream-current-proprio-rssm" \
  "upstream-current|proprio|e3f02248693a79dc8b0ebd62c93683888ddaccfe|upstream-current-proprio-rssm-float32" \
  "paper|vision|bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01|paper-vision-networks" \
  "paper|vision|bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01|paper-vision-networks-float32" \
  "upstream-current|vision|e3f02248693a79dc8b0ebd62c93683888ddaccfe|upstream-current-vision-networks" \
  "upstream-current|vision|e3f02248693a79dc8b0ebd62c93683888ddaccfe|upstream-current-vision-networks-float32"
do
  IFS='|' read -r profile mode revision stem <<TASK1B_FIELDS
$fixture
TASK1B_FIELDS
  uv run python -m world_marl.dreamer_v3_baseline.fixture_generator refresh-manifest --profile "$profile" --observation-mode "$mode" --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision "$revision" --output-dir tests/fixtures/dreamer_v3 --fixture-stem "$stem"
done
uv run python -c 'from pathlib import Path; import hashlib,json,sys; before=json.loads(Path(sys.argv[1]).read_text()); root=Path("tests/fixtures/dreamer_v3"); after={s:hashlib.sha256((root/(s+".npz")).read_bytes()).hexdigest() for s in before}; assert after==before,(before,after); print(len(after),"NPZ payloads byte-identical")' "$NPZ_BEFORE"
uv run pytest tests/test_dreamer_v3_oracle_manifest.py -q
uv run ruff check src/world_marl/dreamer_v3_baseline/oracle.py src/world_marl/dreamer_v3_baseline/network_oracle.py src/world_marl/dreamer_v3_baseline/rssm_oracle.py src/world_marl/dreamer_v3_baseline/replay_oracle.py src/world_marl/dreamer_v3_baseline/replay_oracle_contract.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_oracle_manifest.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/oracle.py src/world_marl/dreamer_v3_baseline/network_oracle.py src/world_marl/dreamer_v3_baseline/rssm_oracle.py src/world_marl/dreamer_v3_baseline/replay_oracle.py src/world_marl/dreamer_v3_baseline/replay_oracle_contract.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_oracle_manifest.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/agent.py >/dev/null
```

### Task 1c: Enforce the runtime/tooling import boundary

**Owned files/symbols:** `__init__.py` only for oracle/tooling imports and exports,
`rssm.py` only for the registration/source-spec/fixture scan-key removal,
`oracle.py` only after those import edges are removed and only to delete the
temporary `OracleSourceSpec`/source-spec registry interfaces retained across
the Task-1b-to-1c boundary; the immutable fixture source-hash/dtype tables,
direct lookup helpers, and thin source-name constants remain and need no
Task-1c edit,
`tests/test_dreamer_v3_distributions.py`, `tests/test_dreamer_v3_networks.py`,
`tests/test_dreamer_v3_rssm_parity.py` only for direct tooling imports and
boundary assertions, and new `tests/test_dreamer_v3_runtime_imports.py`.
No numerical equation, fixture, manifest, or parser is owned.
Task 1c owns `__init__.py` only for oracle/tooling imports and exports; the Task
1a package-root removal is already complete, and Task 1c must not restore,
alias, or otherwise edit `ActorCriticConfig`. Task 9 separately owns later
replacement of the remaining production exports.

**First RED:** a fresh interpreter enumerates tooling-named package modules and
fails because runtime import eagerly loads them; component tests fail until the
single scan-key import is `rssm_oracle.py::ninjax_scan_sample_keys`.
**Implementation:** remove runtime registrations/exports and migrate only those
test imports. The three component files must collect, proving their import seam,
but Task 1c does not execute their numerical tests: the network/RSSM files still
use the removed legacy config constructor and Tasks 3b/3c own that migration and
full execution. **Focused acceptance:** collect the three component files, run
the runtime-import test and literal fresh-interpreter `pkgutil.walk_packages`
check below, then scoped Ruff check/format. Report the before/after import graph
and collection count; fresh spec/quality review and fresh fixer must reach 0
Critical/Important before `refactor(dreamer): isolate parity tooling`.

Task 1c intentionally owns three executable shell fences: its scoped unit gate,
the integrated Task-1 regression gate, and the manifest-regeneration commands.
Every other leaf execution unit owns one. The Task-0 parser asserts these exact
counts from anchored Markdown heading spans; it does not search for a heading
substring inside its own source.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest --collect-only -q tests/test_dreamer_v3_distributions.py tests/test_dreamer_v3_networks.py tests/test_dreamer_v3_rssm_parity.py
uv run pytest tests/test_dreamer_v3_runtime_imports.py -q
uv run ruff check src/world_marl/dreamer_v3_baseline/__init__.py src/world_marl/dreamer_v3_baseline/rssm.py tests/test_dreamer_v3_distributions.py tests/test_dreamer_v3_networks.py tests/test_dreamer_v3_rssm_parity.py tests/test_dreamer_v3_runtime_imports.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/__init__.py src/world_marl/dreamer_v3_baseline/rssm.py tests/test_dreamer_v3_distributions.py tests/test_dreamer_v3_networks.py tests/test_dreamer_v3_rssm_parity.py tests/test_dreamer_v3_runtime_imports.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/rssm.py >/dev/null
```

#### Integrated Task 1 regression inventory

**Aggregate sequential ownership:**
`src/world_marl/dreamer_v3_baseline/config.py`, `oracle.py`,
`network_oracle.py`, `rssm_oracle.py`, `replay_oracle.py`,
`replay_oracle_contract.py`, new `fixture_generator.py` limited to
`main`, `_parse_args`, `_validate_reference`, `_canonical_request`,
`_write_pair`, `_PARSER_REGISTRY`, `_register_parser`, `refresh_manifest`, and
`_register_refresh_manifest_parser`; `__init__.py` is split by symbol: Task 1a
removes the `ActorCriticConfig` import and `__all__` entry and adds exactly the
five named replacement public config imports and `__all__` entries, Task 1b
removes only the dead `OracleHarness` and `OracleInvocation` imports and
`__all__` entries, while Task 1c later removes the remaining eager
oracle/tooling imports and exports. Task 9 later replaces
the remaining production exports;
`tests/test_dreamer_v3_config.py`, `tests/test_dreamer_v3_oracle_manifest.py`,
new `tests/test_dreamer_v3_runtime_imports.py`, and the following narrow
runtime-boundary exceptions: `rssm.py` only to remove
`OracleSourceSpec`/`register_oracle_source_spec`, `RSSM_SOURCE_SPEC`, and the
fixture-only `ninjax_scan_sample_keys`; and
`tests/test_dreamer_v3_distributions.py`, `tests/test_dreamer_v3_networks.py`,
and `tests/test_dreamer_v3_rssm_parity.py` only to replace runtime-package
oracle imports/registrations with direct fixture-tooling imports and assert the
package boundary. Task 1b separately owns their obsolete compact-manifest,
request, command, and removed-generator assertions/imports so the literal Task
1c component collection gate is executable before Task 1c begins; it does not
own their config constructors, equations, tolerances, or numerical assertions.
Tasks 3a-3c own full component execution, with Tasks 3b and 3c also owning the
network/RSSM test migration from the removed legacy config constructor. Task 1c
deletes the transitional
`OracleSourceSpec` and source-spec registry definitions from `oracle.py` only
after removing the `rssm.py` and package-root import edges in the same unit.
Task 1b proves by AST/reference inventory that no manifest, fixture generator,
or thin oracle wrapper uses those transitional definitions, so Task 1c's
allowlist is sufficient without editing `fixture_generator.py` or an oracle
wrapper.
`rssm_oracle.py::ninjax_scan_sample_keys` is the single owner
and import path for that fixture helper. It is not a runtime RSSM algorithm,
is not exported from `rssm.py` or `__init__.py`, and fixture tests import it
directly. No distribution, network, or RSSM equation is owned by Task 1.
Task 1 also owns
`tests/fixtures/dreamer_v3/README.md`. Task 1 may update exactly these manifest
metadata files: `paper-proprio-distributions.manifest.json`,
`upstream-current-proprio-distributions.manifest.json`,
`paper-proprio-replay.manifest.json`,
`upstream-current-proprio-replay.manifest.json`,
`paper-proprio-rssm.manifest.json`, `paper-proprio-rssm-float32.manifest.json`,
`upstream-current-proprio-rssm.manifest.json`,
`upstream-current-proprio-rssm-float32.manifest.json`,
`paper-vision-networks.manifest.json`,
`paper-vision-networks-float32.manifest.json`,
`upstream-current-vision-networks.manifest.json`, and
`upstream-current-vision-networks-float32.manifest.json`. It does not own any
NPZ payload or distribution/network/RSSM/replay algorithm.
The parser acceptance function owned in
`tests/test_dreamer_v3_oracle_manifest.py` is exactly
`test_fixture_generator_refresh_manifest_parser`; later tasks do not edit it.

**Integrated interfaces/dependencies:** produce immutable canonical config resolution and
hashes for Nature paper and explicit current profiles; one profile-selected
`authority_revision`; a small fixture-only source-hash/tensor-schema
`OracleManifest`; strict one-to-one `ParameterTranslator`; and an explicit
fixture-generation command. Config/runtime manifests contain no source maps or
live implementation hashes. Runtime package import must not import oracle code.
Depends on Task 0.

**Integrated RED matrix:** snapshot tests assert all four model resolutions: paper Vision
200M, paper Proprio 200M, current Vision 200M, current Proprio `size1m`; paper
also has 20 tasks in both modes, 1,000,000 steps, repeat 1, strided convolution,
and beta2 0.99, while current is explicit, pooling/upsampling, beta2 0.999, and
1.1e6 steps. A fresh-interpreter test enumerates every package module, derives
the tooling set from any leaf containing `oracle`, `fixture`, `generator`, or
`contract`, asserts all six current helpers are in that set, and fails if any
such module is imported by the runtime package; an eagerly imported future
similarly named helper therefore fails without updating a hard-coded denylist.
Scope tests fail on callback/interpreter-authentication APIs. Both-profile
runtime-identity tests require the exact selected revision, prove config
resolution needs no live reference checkout, and reject authority-source maps,
implementation revisions/maps, and generic source fields in production config.
Each numerical fixture manifest separately records and recomputes only the
official `git show <selected revision>:<path>` blobs that generated that case.
`test_fixture_generator_refresh_manifest_parser` fails before the stable root
parser and refresh registration helper exist. Task 1b first migrates the three
component suites off obsolete manifest/request/generator contracts. Task 1c
requires their collection/import seam, not their numerical execution, and fails
until all oracle registrations/exports disappear from the runtime import graph
and the fixture-only scan-key helper has exactly the direct `rssm_oracle.py`
import path above. Tasks 3a-3c then require complete numerical execution after
the owning test/config migrations. Replay's removed
`run_replay_case` caller remains Task 2 ownership and is not restored through a
compatibility worker.

**Integrated implementation constraints:** (1) add profile and import-boundary tests; (2) correct
typed snapshots and remove legacy production constructor/config paths without
claiming ownership of the stale numerical-test constructors; (3) reduce oracle
validation to source hashes, canonical request/schema, deterministic fixtures,
and parameter bijection; (4) delete callback fingerprints, reload games,
interpreter aliases, and source-text contracts; (5) regenerate only the exact
owned manifests whose config/source metadata changed through an explicitly
supplied read-only checkout; (6) refactor common
canonical JSON/hash helpers; (7) bind both profiles to their exact authority
revision while keeping fixture-only blob hashes out of runtime config; (8)
remove the runtime registration block from `rssm.py`, move the scan-key fixture
helper to `rssm_oracle.py`, migrate the three component-test imports, and prove
neither the package root nor any production module imports tooling. The direct
fixture source tables/helpers remain after Task 1c; only the temporary runtime
class and global registry are deleted there.

**Integrated regression acceptance after 1a-1c:**

```shell
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest -q tests/test_dreamer_v3_config.py tests/test_dreamer_v3_oracle_manifest.py
uv run pytest --collect-only -q tests/test_dreamer_v3_distributions.py tests/test_dreamer_v3_networks.py tests/test_dreamer_v3_rssm_parity.py
uv run pytest -q tests/test_dreamer_v3_runtime_imports.py
uv run python -c "import pkgutil,sys,world_marl.dreamer_v3_baseline as p; names={m.name for m in pkgutil.walk_packages(p.__path__,p.__name__+'.')}; tooling={n for n in names if any(t in n.rsplit('.',1)[-1] for t in ('oracle','fixture','generator','contract'))}; required={p.__name__+'.oracle',p.__name__+'.network_oracle',p.__name__+'.rssm_oracle',p.__name__+'.replay_oracle',p.__name__+'.replay_oracle_contract',p.__name__+'.fixture_generator'}; assert required<=tooling,(required-tooling); loaded=tooling&set(sys.modules); assert not loaded,sorted(loaded)"
uv run ruff check src/world_marl/dreamer_v3_baseline tests/test_dreamer_v3_config.py tests/test_dreamer_v3_oracle_manifest.py tests/test_dreamer_v3_runtime_imports.py tests/test_dreamer_v3_distributions.py tests/test_dreamer_v3_networks.py tests/test_dreamer_v3_rssm_parity.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline tests/test_dreamer_v3_config.py tests/test_dreamer_v3_oracle_manifest.py tests/test_dreamer_v3_runtime_imports.py tests/test_dreamer_v3_distributions.py tests/test_dreamer_v3_networks.py tests/test_dreamer_v3_rssm_parity.py
```

Regenerate Task 1's metadata with these literal commands (the existing NPZ
payloads are inputs and must remain byte-identical):

```shell
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator refresh-manifest --profile paper --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3 --fixture-stem paper-proprio-distributions
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator refresh-manifest --profile upstream-current --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3 --fixture-stem upstream-current-proprio-distributions
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator refresh-manifest --profile paper --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3 --fixture-stem paper-proprio-replay
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator refresh-manifest --profile upstream-current --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3 --fixture-stem upstream-current-proprio-replay
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator refresh-manifest --profile paper --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3 --fixture-stem paper-proprio-rssm
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator refresh-manifest --profile paper --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3 --fixture-stem paper-proprio-rssm-float32
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator refresh-manifest --profile upstream-current --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3 --fixture-stem upstream-current-proprio-rssm
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator refresh-manifest --profile upstream-current --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3 --fixture-stem upstream-current-proprio-rssm-float32
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator refresh-manifest --profile paper --observation-mode vision --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3 --fixture-stem paper-vision-networks
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator refresh-manifest --profile paper --observation-mode vision --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3 --fixture-stem paper-vision-networks-float32
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator refresh-manifest --profile upstream-current --observation-mode vision --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3 --fixture-stem upstream-current-vision-networks
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator refresh-manifest --profile upstream-current --observation-mode vision --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3 --fixture-stem upstream-current-vision-networks-float32
```

**Aggregate regression report:** append the three already accepted commit SHAs
and the integrated regression output; it creates no fourth Task-1 commit.

## Task 2: Define carry/row contract and simplify replay

**Owned files and boundary:** `src/world_marl/dreamer_v3_baseline/replay.py`, new
`src/world_marl/dreamer_v3_baseline/agent.py` limited to `AgentCarry`,
`validate_action_tree`, `validate_replay_row`, `DreamerAgent.initial`, and
`DreamerAgent.apply_replay_context`, Task 1's `fixture_generator.py` limited to
`generate_replay` and `_register_replay_parser`,
`tests/test_dreamer_v3_replay.py`, new
`tests/test_dreamer_v3_agent_contract.py`, and exactly
`paper-proprio-replay.npz`, `paper-proprio-replay.manifest.json`,
`upstream-current-proprio-replay.npz`, and
`upstream-current-proprio-replay.manifest.json` under
`tests/fixtures/dreamer_v3/`. Do not implement losses, optimizer, environment,
or driver. Sequential overlap with Task 1 is allowed; simultaneous editors are
not. Task 2 owns exactly
`tests/test_dreamer_v3_replay.py::test_fixture_generator_replay_parser` for the
subcommand contract.

**Interfaces/dependencies:** `AgentCarry`, model/replay action tree validation,
`ReplayKey`, `ReplayBatch`, `ReplayChunk`, `ReplayWriter`, `OnlineQueue`,
`UniformSelector`, `ConsecutiveStream`, bounded immutable replay mutation plans,
`DreamerReplay.prepare_add/commit_add`, `prepare_sample/commit_sample`, `state_dict`,
transactional `DreamerReplay.from_state_dict(state, replay_config,
sequence_shape, transition_spaces, latent_spaces)`, immutable public
train `raw_length = K + T * consecutive`, report
`report_raw_length = K + report_length * report_consecutive`,
`can_sample_batch(mode)`, `validate`, and
exact context writeback. The closed inverse APIs are exactly
`AgentCarry.from_state(state, agent, expected_leading_shape)` and
`ReplayBatch.from_state(state, transition_spaces, latent_spaces, expected_batch_size, expected_time_length)`.
The already-bound Agent supplies encoder, decoder, RSSM, and action schemas;
the stream/replay owner supplies spaces and exact batch/raw-time dimensions.
Task 2 creates and owns
`DreamerReplay.can_sample_batch(mode)`: the only modes are `"train"` and
`"report"`; it separately checks that all `B` requested sequence items are
currently available from the named stream without sampling or changing the
selector, stream cursor, online queue, replay RNG, identity/sample counters, or
any limiter. Depends on Task
1's config and runtime boundary.

Task 2 owns exact primitive checkpoint records and inverse validators for
`AgentCarry`, its then-available encoder/decoder carry and `RSSMState`
projection, `ReplayBatch`, every chunk/writer/queue/selector/stream record, and
the complete `DreamerReplay` state. A live `ConsecutiveStream` stores its
current batch through `ReplayBatch.state_dict()` and restores it only through
the exact `ReplayBatch.from_state` call above; all state APIs allocate fresh
plain dict/list containers and copied numeric arrays, reject tuple/dataclass/
`FrozenDict`/object leaves and aliases, and reconstruct exact runtime classes.
Task 2 first adds a representative replay-with-live-stream public-Flax RED.
Task 2b later changes only the two integer-key mapping representations found by
that RED; it does not postpone the rest of the replay owner schema.

`ReplayConfig` contains no selector seed, and the public `DreamerReplay`
constructor takes no seed. Fresh replay internally constructs exactly
`UniformSelector(seed=0)`. `UniformSelector(seed)` remains available only as the
narrow component constructor used by selector tests. REDs bind two distinct
public run seeds to distinct official parameter/counter roots and DMC derived identities,
but require identical fresh replay selector seed-0 state and sample sequence.
After selector advancement, complete PCG64 state roundtrip proves the advanced
selector state resumes at the exact next sample; resume never reconstructs
progress from zero.

**First RED:** table-driven tests construct ordinary, first, terminal,
truncated, and auto-reset rows and fail the current same-row action assumption;
an action sample with a component outside `[-1,1]` must pass Agent/replay
validation and remain byte-identical in the replay row (except that every
`is_last` next-action leaf is already zero). A separate table accepts every
already-zero terminal/truncated action tree, rejects a nonzero final-row action
with the exact validation error, and proves the rejected caller tree and replay
state remain byte-identical; replay never masks or rewrites it. Online-phase
cases assert zero-based
logical starts `1+nR` for `R>1`, every start for `R=1`, global FIFO add order,
independent writer phases, and an initial/partially-advanced restore. These
tests are compared directly with the old-length-before-increment ordering in
official `embodied/core/replay.py::Replay.add`.
The complexity RED measures that the existing implementation scans
lifetime/capacity state. The GREEN measures mutation/work bounded by the valid
sampled sequence/context and independent of lifetime/capacity; restore
tests exercise valid online queue, independent train and training report
streams, selector RNG, context mutation, and an evicted-but-valid writer
continuation. There is no evaluation replay stream: policy-return evaluation
never owns or mutates replay, and reports use the training report stream.
Readiness REDs cover both modes, all `B` items present, one of `B` absent, zero
items, separate stream progress, and byte/tree equality of the complete replay
state before and after every query. They call readiness before any selector,
stream, queue, replay RNG, or limiter mutation. Task 7b consumes this API and is
forbidden to edit replay.
With nonzero context and consecutive counts greater than one, source-derived
REDs assert the immutable train and report properties are respectively
`K + T * consecutive` and
`K + report_length * report_consecutive`, differ from their trimmed lengths,
and survive restore; Task 7a consumes only the train property.
Replay-context reconstruction and its negative-index slice run only when
`K>0`. Tests prove that at `K=0` every consecutive slice uses the incoming
carry and normal prepended-previous-action alignment. Separate zero-context and
nonzero-context cases assert carry, previous actions, observations, and step
IDs.
Identity tests distinguish allocatable chunk IDs `[1, 2**128 - 1]` from the
Python cursor domain `[1, 2**128]`, whose `2**128` is the exhausted sentinel,
and allocatable item IDs `[0, 2**63 - 1]` from the Python cursor domain
`[0, 2**63]`, whose `2**63` is the exhausted sentinel. Both cursors are explicit
exceptions to ordinary signed-int64 counters. Tests cover big-endian 16-byte
encoding, collision/exhaustion preflight, final allocation, exhausted-state
roundtrip, and the next request rejecting without mutation; reverse the live
test that rejects a valid exhausted restore. A checkpoint immediately
before a chunk rollover and item allocation must reproduce all later IDs,
samples, queues, links, and context writeback after restore.
The named parser test fails until `_register_replay_parser` is registered by the
unchanged Task 1 root parser.
An API-declaration RED imports both classes and uses `inspect.signature` to
require exactly `state,agent,expected_leading_shape` and
`state,transition_spaces,latent_spaces,expected_batch_size,expected_time_length`
after classmethod binding. Empty inverses, generic dependency bags, and
self-described batch dimensions fail before behavioral restore cases run.

**Red-green-refactor:** (1) freeze row/carry shapes, raw unbounded
normalized-coordinate actions, `TensorSpace.low/high` as metadata rather than
Agent/replay bounds, final-row zero validation without mutation, and the
previous-action formula; (2) retain
chunk/link/selector/online-first semantics while replacing lifetime history with
bounded valid-state data; (3) implement one `int64` online phase per writer by
testing the old length, enqueueing, then incrementing, with stable global FIFO
order; (4) make add/evict mutation-local and transactional through prevalidated
bounded plans whose commit is nonallocating/no-fail, never a full replay copy;
(5) persist all
valid operational state, including every writer phase and initial phase, and
validate candidate restore before swap; (6) put exhaustive checks only in
explicit `validate()`; (7) remove hostile private-container,
exact-Python-type, callback, and source text tests; (8) add seeded official
happy-path and exact-resume fixtures; (9) make both identity cursors sole-owned,
serialize the wider Python integer without int64 coercion, and perform an
owner-local wraparound preflight at the actual mutation boundary.

**Focused acceptance:**

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator replay --profile paper --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator replay --profile upstream-current --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run pytest -q tests/test_dreamer_v3_agent_contract.py tests/test_dreamer_v3_replay.py
uv run pytest -q tests/test_dreamer_v3_config.py tests/test_dreamer_v3_oracle_manifest.py
uv run python -c 'import inspect; from world_marl.dreamer_v3_baseline.agent import AgentCarry; from world_marl.dreamer_v3_baseline.replay import ReplayBatch; assert list(inspect.signature(AgentCarry.from_state).parameters)==["state","agent","expected_leading_shape"]; assert list(inspect.signature(ReplayBatch.from_state).parameters)==["state","transition_spaces","latent_spaces","expected_batch_size","expected_time_length"]'
uv run ruff check src/world_marl/dreamer_v3_baseline/agent.py src/world_marl/dreamer_v3_baseline/replay.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_agent_contract.py tests/test_dreamer_v3_replay.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/agent.py src/world_marl/dreamer_v3_baseline/replay.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_agent_contract.py tests/test_dreamer_v3_replay.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/core/replay.py >/dev/null
```

**Report/review/commit:** report required fields plus measured insertion scaling
and serialized-state inventory. Spec review checks official chronology/context;
quality review checks bounded work, transactionality, and deletion of security
scope. Fix/re-review to zero Critical/Important before
`refactor(dreamer): simplify online replay state`.

### Task 2b: Canonical replay checkpoint records for public Flax restore

**Why this bounded follow-up exists:** the Task-0 Flax 0.10.4 mechanical RED
proved that `msgpack_serialize` can encode the current integer-key
`selector.indices` and `writers` maps, but public `msgpack_restore` rejects
them with `ValueError: int is not allowed for map key when
strict_map_key=True`. Fix this at the replay serialization owner rather than
adding a fifth/general mapping adapter to Task 8b.

**Owned files/symbols:** edit only `UniformSelector.state_dict`/
`from_state_dict`, `DreamerReplay.state_dict`/restore validation for `writers`,
and their narrow helpers in `replay.py`; edit only
`tests/test_dreamer_v3_replay.py`. Runtime `indices: dict[int,int]` and
`writers: dict[int,ReplayWriter]` remain unchanged. No sampling, insertion,
identity allocation, fixture, config, dependency, or checkpoint-code edit is
owned.

**First RED:** serialize a replay containing multiple nonmonotonic item and
worker IDs with public `flax.serialization.msgpack_serialize`, then restore it
with public `msgpack_restore`; the current state raises on the integer map key.
The corrected checkpoint form is exact:

- `selector.indices` is a list of records
  `{"index": int, "item_id": int}` in increasing `item_id` order;
- `writers` is a list of records
  `{"state": ReplayWriterState, "worker_id": int}` in increasing `worker_id`
  order. Record keys are inserted in the shown canonical lexical order.

Inverse validation requires exact list/record types and keys, exact Python
non-bool integers in declared ID/index domains, strict increasing IDs, no
duplicates, `index` values forming exactly `0..len(keys)-1`, exact agreement
with selector `keys`, and each nested writer state's `worker_id` matching the
outer record. Missing, extra, reordered, duplicate, wrong-range, float/bool,
and cross-record mismatches reject before live replay mutation. `state_dict`
returns fresh records and restore retains no candidate containers.

**Green/refactor:** prove two insertion histories with the same logical replay
state produce identical serialized bytes, public-Flax roundtrip reconstructs
the exact runtime maps, and the next sample/add plus all resulting state equals
an uninterrupted branch. A recursive walk over a representative complete
replay state (multiple chunks, writers, selector items, queue, and both streams)
asserts no mapping anywhere retains an integer key; this guards future replay
subtrees instead of checking only the two repaired paths. Keep bytes-key `refs` unchanged because Flax public
restore accepts bytes keys and canonicalizes their order. Fresh spec/quality
review must reach zero Critical/Important before
`fix(dreamer): canonicalize replay checkpoint state`.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest -q tests/test_dreamer_v3_replay.py::test_public_flax_roundtrips_canonical_replay_state tests/test_dreamer_v3_replay.py
uv run python -c 'from importlib.metadata import version; assert version("flax")=="0.10.4"'
uv run ruff check src/world_marl/dreamer_v3_baseline/replay.py tests/test_dreamer_v3_replay.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/replay.py tests/test_dreamer_v3_replay.py
git diff --check
```

## Task 3: Prove composed replay-to-world-model parity

Task 3 is four sequential implementer-review-fixer-commit units. Each unit owns
one numerical layer and its generator/test/fixture slice; no editors overlap.

### Task 3a: Dreamer distributions

**Owned files/symbols:** `distributions.py`; only
`generate_distributions`/`_register_distributions_parser` in
`fixture_generator.py`; `tests/test_dreamer_v3_distributions.py`; and both
files for `paper-proprio-distributions` and
`upstream-current-proprio-distributions`. **First RED:** official values and
gradients for symlog/symexp, MSE, Binary, Categorical, OneHot, TwoHot, and Agg,
including direct `prob(event)` and event reduction, plus the absent parser.
Direct head REDs require `MSEOutput(mean, squash=None)`, a stopped optionally
squashed target, Proprio `squash=symlog` while `pred()` remains the encoded
mean, bounded Normal `tanh(raw_mean)` and
`(maxstd-minstd)*sigmoid(raw_std + 2)+minstd`, and categorical mixing that is
opt-in and defaults off: ordinary categorical heads use none while OneHot/RSSM
use configured unimix.
**Implementation:** translate each pinned output directly and register only its
subcommand. **Focused acceptance:** generate both stems, run the one test file,
then scoped Ruff check/format. Report per-operation errors and fixture digests;
fresh spec/quality review and fresh fixer must reach 0 Critical/Important before
`feat(dreamer): port output distributions`.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator distributions --profile paper --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator distributions --profile upstream-current --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run pytest tests/test_dreamer_v3_distributions.py -q
uv run ruff check src/world_marl/dreamer_v3_baseline/distributions.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_distributions.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/distributions.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_distributions.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/jax/outs.py >/dev/null
```

### Task 3b: Network primitives and Vision encoder/decoder

**Owned files/symbols:** `networks.py`; only
`generate_networks`/`_register_networks_parser`; the networks test; and both
dtype files for `paper-vision-networks` and
`upstream-current-vision-networks`. This task removes the network test's
`_legacy=True` RSSM-config construction and replaces it with the public resolved
config shape before requiring full file execution. **First RED:** supplied-parameter value/
gradient cases cover initialization, RMSNorm, Linear/BlockLinear, convolution,
MLP/head, exact paper stride-2 versus current conv-then-2x2-max-pool encoder,
and exact spatial projection/upsampling/transposed-conv/final-sigmoid decoder.
**Implementation:** repair primitives first, then encoder/decoder in pinned
operation order. **Focused acceptance:** generate the four cases, run the
network test, then scoped Ruff check/format. Report shapes, parameter bijection,
and maximum errors; fresh spec/quality review and fixer must reach 0
Critical/Important before `feat(dreamer): port vision networks`.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator networks --profile paper --observation-mode vision --compute-dtype bfloat16 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator networks --profile paper --observation-mode vision --compute-dtype float32 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator networks --profile upstream-current --observation-mode vision --compute-dtype bfloat16 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator networks --profile upstream-current --observation-mode vision --compute-dtype float32 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run pytest tests/test_dreamer_v3_networks.py -q
uv run ruff check src/world_marl/dreamer_v3_baseline/networks.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_networks.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/networks.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_networks.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/jax/nets.py >/dev/null
```

### Task 3c: RSSM and BlockGRU

**Owned files/symbols:** `rssm.py`; only
`generate_rssm`/`_register_rssm_parser`; the RSSM parity test; and both dtype
files for `paper-proprio-rssm` and `upstream-current-proprio-rssm`.
This task removes the RSSM parity test's `_legacy=True` config construction and
replaces it with the public resolved config shape before requiring full file
execution.
Task 3c finalizes the primitive `RSSMState.state_dict()` and closed
`RSSMState.from_state(state, config, expected_leading_shape)` APIs consumed by
all Agent carries. They copy the exact deter/stoch arrays, reject wrong keys,
shape, dtype, aliases, tuple/dataclass/`FrozenDict` leaves, and roundtrip through
public Flax before reconstruction.
**First RED:** non-singleton B/T/S/C supplied-noise cases cover mid-sequence
reset, exact reset/candidate/update equations, prior/posterior values, KLs,
scan order, final state, and gradients. **Implementation:** translate
`initial/observe/imagine/_observe/_core` without fixture branches. **Focused
acceptance:** generate all four RSSM cases, run its test, then scoped Ruff
check/format. Report equations, sampling order, and errors; fresh spec/quality
review and fixer must reach 0 Critical/Important before
`feat(dreamer): port recurrent state model`.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator rssm --profile paper --observation-mode proprio --compute-dtype bfloat16 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator rssm --profile paper --observation-mode proprio --compute-dtype float32 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator rssm --profile upstream-current --observation-mode proprio --compute-dtype bfloat16 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator rssm --profile upstream-current --observation-mode proprio --compute-dtype float32 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run pytest tests/test_dreamer_v3_rssm_parity.py -q
uv run ruff check src/world_marl/dreamer_v3_baseline/rssm.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_rssm_parity.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/rssm.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_rssm_parity.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/rssm.py >/dev/null
```

### Task 3d: Composed replay-to-world model

**Owned files/symbols:** only `DreamerAgent.preprocess/world_model` in
`agent.py`; only `generate_world_model`/`_register_world_model_parser`; the
world-model parity test; and both files for
`{paper,upstream-current}-{proprio,vision}-world-model` (eight files). No
component equation is owned. **First RED:** one shifted-action replay batch per
profile/mode compares preprocessing, tokens, priors/posteriors, decoder
predictions, reward/continuation heads, KLs, final carry, and gradients; Vision
cases additionally prove final sigmoid and HWC aggregate reconstruction loss.
**Implementation:** compose only the accepted 3a-3c APIs and require complete
parameter consumption. **Focused acceptance:** generate all four composed
cases, run the world-model test plus Tasks 2/3 component regressions, then
scoped Ruff check/format. Report parameter bijection and value/gradient errors;
fresh spec/quality review and fixer must reach 0 Critical/Important before
`feat(dreamer): compose replay world model`.

#### Integrated Task 3 regression inventory

**Aggregate sequential ownership:**
`src/world_marl/dreamer_v3_baseline/distributions.py`, `networks.py`, `rssm.py`,
only `DreamerAgent.preprocess` and `DreamerAgent.world_model` in `agent.py`, and
only `generate_distributions`/`_register_distributions_parser`,
`generate_networks`/`_register_networks_parser`,
`generate_rssm`/`_register_rssm_parser`, and
`generate_world_model`/`_register_world_model_parser` in
`fixture_generator.py`; existing
`tests/test_dreamer_v3_distributions.py`, `tests/test_dreamer_v3_networks.py`,
`tests/test_dreamer_v3_rssm_parity.py`, and new
`tests/test_dreamer_v3_world_model_parity.py`. Under
`tests/fixtures/dreamer_v3/`, fixture ownership is exactly both `.npz` and
`.manifest.json` for stems `paper-proprio-distributions`,
`upstream-current-proprio-distributions`, `paper-vision-networks`,
`paper-vision-networks-float32`, `upstream-current-vision-networks`,
`upstream-current-vision-networks-float32`, `paper-proprio-rssm`,
`paper-proprio-rssm-float32`, `upstream-current-proprio-rssm`, and
`upstream-current-proprio-rssm-float32`, plus new
`paper-proprio-world-model.npz`, `paper-proprio-world-model.manifest.json`,
`upstream-current-proprio-world-model.npz`, and
`upstream-current-proprio-world-model.manifest.json`, plus
`paper-vision-world-model.npz`, `paper-vision-world-model.manifest.json`,
`upstream-current-vision-world-model.npz`, and
`upstream-current-vision-world-model.manifest.json` under
`tests/fixtures/dreamer_v3/`. Do not add actor/critic losses or optimizer logic.
The exact parser tests are
`test_fixture_generator_distributions_parser` in the distributions test,
`test_fixture_generator_networks_parser` in the networks test,
`test_fixture_generator_rssm_parser` in the RSSM test, and
`test_fixture_generator_world_model_parser` in the world-model test.

**Integrated interfaces/dependencies:** preprocessing, `DictEncoder`, `RSSM.observe`,
`DictDecoder`, reward/continuation heads, and parameter translation compose on
one `[B,T]` replay batch with shifted previous actions, mid-sequence reset, and
non-singleton dimensions. Depends on Tasks 1-2.

**Integrated RED matrix:** an official supplied-parameter/noise fixture compares encoder
tokens, priors/posteriors, decoder distributions, reward/continuation outputs,
KL values, final carry, and gradients. Distribution parity includes direct
`prob(event)` values, the pinned summed `AggregateOutput.prob`, and continuation
`prob(1)`. It must initially fail because no native composed path exists; a
separate test prevents singleton B/T from masking scan or reduction errors, and
the four named parser tests fail until their registration helpers exist.

**Integrated implementation constraints:** (1) add the composed fixture/test without changing
components; (2) implement the smallest composition API; (3) fix source-backed
component mismatches one at a time with per-mismatch RED; (4) require complete
source/destination parameter consumption; (5) refactor duplicated preprocessing
only after all outputs and gradients are green.

**Integrated regression acceptance after 3a-3d:**

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator distributions --profile paper --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator distributions --profile upstream-current --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator networks --profile paper --observation-mode vision --compute-dtype bfloat16 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator networks --profile paper --observation-mode vision --compute-dtype float32 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator networks --profile upstream-current --observation-mode vision --compute-dtype bfloat16 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator networks --profile upstream-current --observation-mode vision --compute-dtype float32 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator rssm --profile paper --observation-mode proprio --compute-dtype bfloat16 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator rssm --profile paper --observation-mode proprio --compute-dtype float32 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator rssm --profile upstream-current --observation-mode proprio --compute-dtype bfloat16 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator rssm --profile upstream-current --observation-mode proprio --compute-dtype float32 --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator world-model --profile paper --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator world-model --profile upstream-current --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator world-model --profile paper --observation-mode vision --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator world-model --profile upstream-current --observation-mode vision --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run pytest -q tests/test_dreamer_v3_distributions.py tests/test_dreamer_v3_networks.py tests/test_dreamer_v3_rssm_parity.py tests/test_dreamer_v3_world_model_parity.py
uv run pytest -q tests/test_dreamer_v3_agent_contract.py tests/test_dreamer_v3_replay.py
uv run ruff check src/world_marl/dreamer_v3_baseline/distributions.py src/world_marl/dreamer_v3_baseline/networks.py src/world_marl/dreamer_v3_baseline/rssm.py src/world_marl/dreamer_v3_baseline/agent.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_distributions.py tests/test_dreamer_v3_networks.py tests/test_dreamer_v3_rssm_parity.py tests/test_dreamer_v3_world_model_parity.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/distributions.py src/world_marl/dreamer_v3_baseline/networks.py src/world_marl/dreamer_v3_baseline/rssm.py src/world_marl/dreamer_v3_baseline/agent.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_distributions.py tests/test_dreamer_v3_networks.py tests/test_dreamer_v3_rssm_parity.py tests/test_dreamer_v3_world_model_parity.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/agent.py >/dev/null
```

**Aggregate regression report:** append the four accepted commit SHAs and
integrated fixture/test output; it creates no fifth Task-3 commit.

## Task 4: Implement normalizers and the unified Agent objective

Task 4 is four sequential implementer-review-fixer-commit units. They may
overlap `agent.py` and `fixture_generator.py` only sequentially; no two editors
run together. None owns optimizer, DMC, driver, artifacts, or checkpoints.

### Task 4a: Normalizer, slow state, and lambda-return primitives

**Owned files and boundary:** create
`src/world_marl/dreamer_v3_baseline/normalization.py`; create the complete `AgentLoss` declaration
and `lambda_return` in `agent.py`; create
`tests/test_dreamer_v3_normalization.py` and
`tests/test_dreamer_v3_returns.py`. No fixture or generator change.
`AgentLoss` is one frozen result record whose required constructor fields, in
order and with no defaults, are `total_loss`, `named_losses`, `metrics`,
`carry`, `context_entries`, `tokens`, `replay_features`, `normalizer_states`,
and no RNG field. `total_loss` is scalar float32; `carry` is `AgentCarry`;
context/tokens/features retain their declared batch/time pytrees;
`normalizer_states` has exactly `retnorm`, `valnorm`, and `advnorm`. Every field is declared here even though
Tasks 4b-4d populate or consume their objective-specific values later. Those
tasks must not edit the `AgentLoss` declaration.

**Interfaces/dependencies:** `PercentileNormalizerState`,
`PercentileNormalizer.stats/update`, `SlowValueState`, `update_slow_value`, and
`lambda_return(last, terminal, reward, value, bootstrap, disc, lambda_)`.
Depends on Task 3.

The inverse signatures are exactly
`PercentileNormalizerState.from_state(state, config)` and
`SlowValueState.from_state(state, online_critic_params, config)`. The trusted
resolved normalizer/slow configs close conditional leaves, rates, and count;
the already-validated online critic closes every slow-tree key, shape, and
dtype. Neither inverse accepts an empty argument list, generic dependency bag,
or candidate-described tree.

Task 4a owns primitive `state_dict()`/closed `from_state()` records for every
normalizer state and `SlowValueState`. The slow parameter tree is recursively
unfrozen to a fresh plain mapping and reconstructed only after exact online-
critic key/shape/dtype validation. Public-Flax roundtrip and input-nonretention
REDs reject tuple, either dataclass kind, `FrozenDict`, aliases, missing/extra
leaves, and the config-dependent wrong `corr` schema.
An API RED uses `inspect.signature` to require those exact bound parameter
names before the state roundtrips run.

**First RED and red-green-refactor:** first add
`test_agent_loss_declaration_is_complete`, which fails until the exact frozen
field order, required constructor, shapes/dtypes, and mutation rejection exist;
then add value/state/stop-gradient tests
for percentile debias/rate/limits (including `{lo,hi}` with no `corr` when
`debias=false` and `{lo,hi,corr}` when true), slow rate/every/count, termination
versus truncation, and reverse lambda recursion; confirm absent APIs fail. The
slow-state RED constructs an initialized online critic and an empty slow critic,
then requires a key/shape/dtype/value-preserving initial copy with scalar
`int32(0)` count and the pinned first train update at `count % every == 0`.
It also proves a failure before the slow update leaves tree/count unchanged.
Implement one
pure transition at a time, compare pinned
`utils.py::Normalize/SlowModel._initonce/update` and
`agent.py::lambda_return`, then remove duplicated tree helpers.

**Focused acceptance:**

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest -q tests/test_dreamer_v3_normalization.py tests/test_dreamer_v3_returns.py
uv run pytest -q tests/test_dreamer_v3_world_model_parity.py
uv run python - <<'PY_AST'
import ast
from pathlib import Path
tree = ast.parse(Path("src/world_marl/dreamer_v3_baseline/agent.py").read_text())
node = next(value for value in tree.body if isinstance(value, ast.ClassDef) and value.name == "AgentLoss")
fields = [value.target.id for value in node.body if isinstance(value, ast.AnnAssign) and isinstance(value.target, ast.Name)]
assert fields == ["total_loss", "named_losses", "metrics", "carry", "context_entries", "tokens", "replay_features", "normalizer_states"], fields
PY_AST
uv run python -c 'import inspect; from world_marl.dreamer_v3_baseline.normalization import PercentileNormalizerState,SlowValueState; assert list(inspect.signature(PercentileNormalizerState.from_state).parameters)==["state","config"]; assert list(inspect.signature(SlowValueState.from_state).parameters)==["state","online_critic_params","config"]'
uv run ruff check src/world_marl/dreamer_v3_baseline/normalization.py src/world_marl/dreamer_v3_baseline/agent.py tests/test_dreamer_v3_normalization.py tests/test_dreamer_v3_returns.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/normalization.py src/world_marl/dreamer_v3_baseline/agent.py tests/test_dreamer_v3_normalization.py tests/test_dreamer_v3_returns.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/jax/utils.py >/dev/null
```

**Report/review/commit:** report state inventories and maximum value/gradient
errors. Spec review checks the three exact pinned symbols; quality review checks
pure JIT-safe pytrees. Fix/re-review to 0 Critical/Important, then
`feat(dreamer): add return normalization primitives`.

### Task 4b: World-model objective

**Owned files and boundary:** edit only `DreamerAgent.world_loss` in `agent.py`
and only
`generate_world_loss` plus `_register_world_loss_parser` in
`fixture_generator.py`; create
`tests/test_dreamer_v3_world_loss.py` and exactly
`paper-proprio-world-loss.npz`, `paper-proprio-world-loss.manifest.json`,
`upstream-current-proprio-world-loss.npz`, and
`upstream-current-proprio-world-loss.manifest.json`, plus
`paper-vision-world-loss.npz`, `paper-vision-world-loss.manifest.json`,
`upstream-current-vision-world-loss.npz`, and
`upstream-current-vision-world-loss.manifest.json` in the fixture directory.
Task 4b owns exactly
`tests/test_dreamer_v3_world_loss.py::test_fixture_generator_world_loss_parser`.
Task 4b populates the predeclared `context_entries`, `tokens`,
`replay_features`, `named_losses`, and `metrics` fields and must not edit the `AgentLoss` declaration.

**Interfaces/dependencies:** the fixed `AgentLoss` fields and
`DreamerAgent.world_loss` with per-key reconstruction, reward, continuation,
dynamics KL, representation KL, carry, entries, and metrics. Depends on 4a.

**First RED and red-green-refactor:** add official supplied-parameter/noise
value and gradient tests per observation key and term, including exact scales,
reductions, free nats, `reward_grad` both ways, continuation target, and target
stops. Both Vision cases prove
`AggregateOutput(MSEOutput, event_ndims=3, reduction=jnp.sum)` produces one
HWC-summed `[B,T]` loss per image key and compare encoder/RSSM/decoder gradients;
confirm no unified method and no world-loss parser helper. Add each term
and the named parser registration separately, compose the scalar, then refactor
only while the named fixture remains exact.

**Focused acceptance:**

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator world-loss --profile paper --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator world-loss --profile upstream-current --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator world-loss --profile paper --observation-mode vision --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator world-loss --profile upstream-current --observation-mode vision --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run pytest -q tests/test_dreamer_v3_world_loss.py tests/test_dreamer_v3_world_model_parity.py
uv run pytest -q tests/test_dreamer_v3_agent_contract.py tests/test_dreamer_v3_replay.py
uv run ruff check src/world_marl/dreamer_v3_baseline/agent.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_world_loss.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/agent.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_world_loss.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/agent.py >/dev/null
```

**Report/review/commit:** report every fixture digest, per-term error, and
gradient matrix. Spec review checks `Agent.loss` world-term order; quality review
checks no global reconstruction averaging or fixture branches. Fix/re-review to
0 Critical/Important, then `feat(dreamer): port world model objective`.

### Task 4c: Imagination, actor, critic, and replay value

**Owned files and boundary:** edit only `DreamerAgent.imagine`,
`DreamerAgent.imag_loss`, and `DreamerAgent.repl_loss` in `agent.py`, and only
`generate_agent_loss` plus `_register_agent_loss_parser` in
`fixture_generator.py`;
create `tests/test_dreamer_v3_imagination.py` and exactly
`paper-proprio-agent-loss.npz`, `paper-proprio-agent-loss.manifest.json`,
`upstream-current-proprio-agent-loss.npz`, and
`upstream-current-proprio-agent-loss.manifest.json` in the fixture directory.
Task 4c owns exactly
`tests/test_dreamer_v3_imagination.py::test_fixture_generator_agent_loss_parser`.
It populates the already-declared imagination/replay-value losses, metrics, and
normalizer proposals and must not edit the `AgentLoss` declaration.

**Interfaces/dependencies:** `DreamerAgent.imagine`, `imag_loss`, `repl_loss`,
every-valid-state starts, continuation weights, stopped-action/stopped-advantage
REINFORCE, entropy, value likelihood, replay value, and slow regularization.
Depends on 4b.

**First RED and red-green-refactor:** add independent official value/gradient
tests for every replay start, imagination order, lambda targets/weights, actor,
critic, replay value, entropy, and each of `contdisc`, `slowtar`, `repval_grad`,
and `repval_loss`. Test both `ac_grads` branches: false stops the initial replay
feature; true connects only it; subsequent features and imagined policy carries
are stopped in both. Confirm absent methods and parser helper fail, implement in
official order, register the parser, then share only proved stop-target helpers.
The separate
`test_rssm_imagine_shape_separate_from_concatenation` first records the RSSM
call start as `[B*T,...]` and its outputs as `[B*T,H,...]`, then independently
asserts that only the prepended replay feature/final action rows are
`[B*T,1,...]` and the concatenated sequences are `[B*T,H+1,...]`; a spy fails
if a singleton time axis is passed into RSSM.
Replay discount is unconditionally `disc = 1 - 1 / horizon`. The focused
two-branch discount test proves imagination switches between continuation-only
`disc=1` and `disc = 1 - 1 / horizon`, while replay uses the fixed finite-horizon value
in both `contdisc` branches before replay
`lambda_return`.
The counter RED proves the official Agent start axis exactly:
`Kstart = min(imag_last if imag_last != 0 else T, T)` after context trimming.
It covers zero and nonzero `imag_last`, then proves every committed update adds
`B*Kstart*H` to the persistent summary counter.

**Focused acceptance:**

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator agent-loss --profile paper --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator agent-loss --profile upstream-current --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run pytest -q tests/test_dreamer_v3_imagination.py tests/test_dreamer_v3_returns.py tests/test_dreamer_v3_normalization.py
uv run pytest -q tests/test_dreamer_v3_world_loss.py tests/test_dreamer_v3_world_model_parity.py
uv run ruff check src/world_marl/dreamer_v3_baseline/agent.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_imagination.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/agent.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_imagination.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/agent.py >/dev/null
```

**Report/review/commit:** report all-start coverage, errors, and a complete
switch/gradient matrix. Spec review checks `Agent.imag_loss/repl_loss` ordering;
quality review checks detach clarity and shapes. Fix/re-review to 0
Critical/Important, then `feat(dreamer): port actor critic objectives`.

### Task 4d: Agent composition, policy, and report

**Owned files and boundary:** complete only `DreamerAgent.initial`,
`DreamerAgent.policy`, `DreamerAgent.loss`, and `DreamerAgent.report`
composition in `agent.py`; create `tests/test_dreamer_v3_agent.py`.
No fixture or generator change.

**Interfaces/dependencies:** one public functional Agent path returning
`AgentLoss`, context writeback, sampled train/eval policy actions, finite
diagnostics, and
`DreamerAgent.policy(params, carry, observation, mode, outer_seed)` plus
`report(params, carry, batch, outer_seed) -> (arrays, returned carry)`, a
parameter/replay-state-pure posterior-prefix/prior-suffix report whose outer
legacy `uint32[2]` key is call-local and whose post-call Ninjax state is
discarded.
Depends on 4c.

**First RED and red-green-refactor:** add end-to-end Agent tests that fail on the
absent composition, deterministic RNG reuse, mean/argmax evaluation, hidden
state mutation, or posterior leakage in the open-loop suffix. Small official
Ninjax probes freeze policy train/evaluation posterior then sorted action-leaf
draws and report's loss draw sequence followed by the extra prefix observe and
recorded-action suffix imagine draws; tests compare every child key and a
composed stochastic output, not a returned cursor. Report REDs accept an
explicit legacy `uint32[2]` key and prove failure returns no carry/output.

Official-source modality cases require outputs only for decoder image keys.
Vision open-loop cursor/files are nonempty; Proprio open-loop cursor is zero;
Proprio open-loop directory is absent; evaluation cursor/files are nonempty in
both modalities. At this Agent-only boundary the cursor/file phrases describe
the Task-9 observable contract: Vision returns one or more nonempty video
arrays, Proprio returns an empty open-loop mapping, and a fabricated Proprio
numeric/video leaf is rejected. Compose established primitives without new
equations, then simplify only duplicate plumbing.

- Vision open-loop cursor/files are nonempty.
- Proprio open-loop cursor is zero.
- Proprio open-loop directory is absent.
- evaluation cursor/files are nonempty in both modalities.

**Focused acceptance:**

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest -q tests/test_dreamer_v3_agent.py tests/test_dreamer_v3_imagination.py tests/test_dreamer_v3_world_loss.py
uv run pytest -q tests/test_dreamer_v3_agent_contract.py tests/test_dreamer_v3_world_model_parity.py tests/test_dreamer_v3_replay.py
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/agent.py | rg 'for key in self.dec.imgkeys'
uv run ruff check src/world_marl/dreamer_v3_baseline/agent.py tests/test_dreamer_v3_agent.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/agent.py tests/test_dreamer_v3_agent.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/agent.py >/dev/null
```

**Report/review/commit:** report complete API/state/RNG inventory and open-loop
leakage test. Spec review checks official Agent composition; quality review
checks public API and state purity. Fix/re-review to 0 Critical/Important, then
`feat(dreamer): compose unified agent`.

## Task 5: Implement AGC, LaProp, RNG ownership, and `train_step`

**Owned files and boundary:** create
`src/world_marl/dreamer_v3_baseline/optimizer.py`,
`tests/test_dreamer_v3_optimizer.py`, and
`tests/test_dreamer_v3_train_step_parity.py`; edit only
`DreamerAgent.loss`'s functional return boundary in `agent.py` and only
`generate_optimizer_five_step`, `generate_all`, and the
`_register_optimizer_five_step_parser`/`_register_generate_all_parser`
registrations in `fixture_generator.py`.
Task 5 owns adding the exact test/oracle dependency `ninjax==3.6.3` to the
development extra, updating `uv.lock`, and auditing its installed source. Task
0 intentionally performs no host-private cache lookup. Production runtime code
must not import Ninjax. This version is fixture/oracle identity, not algorithm-profile identity
because the official project declares a wider compatible range.
Own exactly
`paper-proprio-optimizer-five-step.npz`,
`paper-proprio-optimizer-five-step.manifest.json`,
`upstream-current-proprio-optimizer-five-step.npz`, and
`upstream-current-proprio-optimizer-five-step.manifest.json` in the fixture
directory. Do not add driver or checkpoint policy. Task 5 owns exactly
`tests/test_dreamer_v3_optimizer.py::test_fixture_generator_optimizer_and_generate_all_parsers`.

**Interfaces/dependencies:** tensorwise AGC, RMS and momentum transforms,
schedules, the unconditional bfloat16/float32 update path,
`DreamerOptimizerState`, `DreamerTrainStateSchema`, `DreamerTrainState`,
`validate_next_update_capacity`, the exact official counter-seed helper, and
`train_step(agent, state, carry, batch, outer_seed)`. Depends
on Task 4's unified loss/state.

Task 5 explicitly owns `DreamerOptimizerState.state_dict()` and
`DreamerOptimizerState.from_state(state, params, config)`,
`DreamerTrainState.initialize(agent, observation_spaces, action_spaces, resolved_config)`,
`DreamerTrainState.schema(agent, observation_spaces, action_spaces, resolved_config)`,
`DreamerTrainState.state_dict()`, and
`DreamerTrainState.from_state(state, agent, observation_spaces, action_spaces, resolved_config)`.
The latter orchestrates the exact optimizer, slow-value, and three normalizer
inverses declared earlier. Parameter,
optimizer-moment, slow-value, and normalizer trees are recursively copied and
unfrozen into plain string-keyed mappings with numeric array leaves; the inverse
validates every key/shape/dtype/count/config relation before reconstructing
fresh runtime pytrees. REDs pass the complete train-state primitive record
through public Flax, prove caller mutation/nonretention, and reject tuple,
Python/Flax dataclasses, `FrozenDict`, arbitrary objects, and aliases. The codec
never repairs an owner-produced train record.

Fresh initialization is the direct `_init_params` translation: use the raw
parameter seed `uint32([resolved_config.seed, 0])`; construct zero complete
observation/action/extra data `[B,T+K,...]`; trace/materialize train carry and
then the complete train path; only afterward initialize optimizer moments,
copy online critic parameters into slow value, initialize normalizers and fixed
counters, and store config identity. Schema uses the same bound agent/spaces
through abstract shape evaluation, creates no live runtime state, and must
match the initialized state key/shape/dtype/byte-for-byte schema exactly.
Here `resolved_config` is exactly `resolved.config`, and
`resolved_config.seed == resolved.config.seed` is equality-checked before
initialization. REDs use two nondefault seeds to prove distinct official parameter/counter roots and DMC derived identities, including exact initialization, but identical fresh
replay selector seed-0 state and sample sequence. Replay has no seed identity
projection to equality-check against `resolved.config.seed`. A scripted
advanced selector checkpoint proves the advanced selector state resumes at the
exact next sample from complete PCG64 state.

**First RED:** exact small pytrees fail tests for AGC floor, RMS-before-momentum,
both bias corrections, paper beta2 0.99/current 0.999, epsilon, the literal
constant/linear/cosine/warmup formulas at global train steps `0`, `W-1`, `W`,
`W+1`, and `A` (with `u=s-W` applied once and cosine alpha exactly `0.1*L`),
the pure initialized-state/schema equality, official first-transition parity,
call-local model/posterior, imagination, and action-distribution children (no
duplicate child state), exact legacy `uint32[2]` outer-key schema, and `jnp.int32[]`
optimizer/RMS/momentum/update/slow-value counters. The pinned-source test proves
`apply_if_finite` is present only under `COMPUTE_DTYPE == float16`, while both
shipped bfloat16/float32 profiles always run `opt.update`, `apply_updates`,
write optimizer state, increment the optimizer step, then update slow value and
construct replay writeback. Wrong key/counter schema and would-overflow next
update fail before limiter/replay/counter mutation. A diagnostic nonfinite case
raises before returning a new train state or writeback and must never return a
skip/retry result. The no later checkpoint integration assertion belongs to
Task 8c, which owns checkpoint publication. The named parser test first fails
on both absent registration
helpers.

The optimizer RED asserts
`parameter_delta = -schedule(s) * laprop_output`, that `schedule(s)` already
contains `L` exactly once, and additive `optax.apply_updates`. A positive scalar
gradient makes the parameter decrease with no extra `L`; constant, warmup,
linear, and cosine are checked at every step of the five-step fixture, with
weight decay inside the same scheduled descent multiplication.
In literal contract language, schedule(s) already contains `L` exactly once
and the parameter decreases. A positive scalar gradient is checked for
constant, warmup, linear, and cosine schedules.

The outer-call RED proves the exact official helper
`default_rng([public_seed, int(counter)]).integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)`
for counters zero, one, and a nontrivial value. It interleaves
`collection -> evaluation -> train -> report -> checkpoint -> resume`, with
collection/evaluation sharing one simulated policy counter and train/report one
simulated batch counter, and compares the uninterrupted and resumed outer keys.
The second-root case must differ from recursively splitting the preceding key.

The within-train RED vendors an oracle fixture from Ninjax 3.6.3 and checks its
FIFO `reserve.pop(0)` behavior and `seed(amount)` path. From one explicit outer
key it asserts the discarded gradient-access child and `loss_seed`; exact
`posterior_keys: [T,2]`, `imagination_action_keys: [H,M,2]`,
`imagination_prior_keys: [H,2]`, and `final_action_keys: [M,2]`; legacy
`uint32[2]` shape; and the complete composed stochastic train output. The
post-call Ninjax cursor/remainder is discarded and never becomes a later outer
key. The native runtime implements this schedule directly; it does not import Ninjax.

API-declaration REDs use `inspect.signature` to enforce every initializer,
schema, inverse, and bound train-step parameter above. Tests prove initializer
execution once when called, schema execution without live-state creation, and
cold inverse execution with zero initializer calls; Tasks 8c/9a repeat those
last two counts at their real bootstrap boundaries.

**Red-green-refactor:** (1) pure AGC; (2) pure RMS state; (3) pure momentum and
schedules; (4) compose LaProp in mandated order; (5) prove float16-only
`apply_if_finite` and implement the unconditional shipped-dtype call; (6)
implement train initialization/schema and one agent-bound `train_step`; (7) update slow
value and replay writeback once after the optimizer call in pinned order; (8)
validate equality of optimizer/RMS/momentum/train/slow counters and expose pure
`validate_next_update_capacity(state)` before limiter/replay consumption; (9)
make configured diagnostics terminate rather than skip and continue; (10)
never request `jnp.int64` under x64-disabled JAX; (11) refactor tree
utilities while five-step parity stays green.

**Focused acceptance:**

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
GEN_DIR="$(mktemp -d /tmp/wm-marl-dreamer-v3-task5-generate-all.XXXXXX)"
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator optimizer-five-step --profile paper --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator optimizer-five-step --profile upstream-current --observation-mode proprio --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir tests/fixtures/dreamer_v3
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator generate-all --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --current-source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir "$GEN_DIR"
uv run python -c 'from pathlib import Path; import hashlib,sys; out=Path(sys.argv[1]); tracked=Path("tests/fixtures/dreamer_v3"); stems=["paper-proprio-agent-loss","paper-proprio-distributions","paper-proprio-optimizer-five-step","paper-proprio-replay","paper-proprio-rssm","paper-proprio-rssm-float32","paper-proprio-world-loss","paper-proprio-world-model","paper-vision-networks","paper-vision-networks-float32","paper-vision-world-loss","paper-vision-world-model","upstream-current-proprio-agent-loss","upstream-current-proprio-distributions","upstream-current-proprio-optimizer-five-step","upstream-current-proprio-replay","upstream-current-proprio-rssm","upstream-current-proprio-rssm-float32","upstream-current-proprio-world-loss","upstream-current-proprio-world-model","upstream-current-vision-networks","upstream-current-vision-networks-float32","upstream-current-vision-world-loss","upstream-current-vision-world-model"]; expected={Path(stem+suf) for stem in stems for suf in (".npz",".manifest.json")}; got={p.relative_to(out) for p in out.rglob("*") if p.is_file()}; assert got==expected,(sorted(map(str,expected-got)),sorted(map(str,got-expected))); bad=[str(rel) for rel in sorted(expected) if (out/rel).read_bytes()!=(tracked/rel).read_bytes()]; assert not bad,bad; print(len(expected),hashlib.sha256(b"".join((out/p).read_bytes() for p in sorted(expected))).hexdigest())' "$GEN_DIR"
uv run pytest -q tests/test_dreamer_v3_optimizer.py tests/test_dreamer_v3_train_step_parity.py tests/test_dreamer_v3_agent.py
uv run pytest -q tests/test_dreamer_v3_world_model_parity.py tests/test_dreamer_v3_replay.py
uv run python -c 'import inspect; from world_marl.dreamer_v3_baseline.optimizer import DreamerOptimizerState,DreamerTrainState,train_step; assert list(inspect.signature(DreamerOptimizerState.from_state).parameters)==["state","params","config"]; assert list(inspect.signature(DreamerTrainState.initialize).parameters)==["agent","observation_spaces","action_spaces","resolved_config"]; assert list(inspect.signature(DreamerTrainState.schema).parameters)==["agent","observation_spaces","action_spaces","resolved_config"]; assert list(inspect.signature(DreamerTrainState.from_state).parameters)==["state","agent","observation_spaces","action_spaces","resolved_config"]; assert list(inspect.signature(train_step).parameters)==["agent","state","carry","batch","outer_seed"]'
uv run python -c 'import importlib.metadata as m; assert m.version("ninjax") == "3.6.3"'
uv run ruff check src/world_marl/dreamer_v3_baseline/optimizer.py src/world_marl/dreamer_v3_baseline/agent.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_optimizer.py tests/test_dreamer_v3_train_step_parity.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/optimizer.py src/world_marl/dreamer_v3_baseline/agent.py src/world_marl/dreamer_v3_baseline/fixture_generator.py tests/test_dreamer_v3_optimizer.py tests/test_dreamer_v3_train_step_parity.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/jax/opt.py >/dev/null
```

The report records `GEN_DIR` and its 48-file hash. It is retained through
review; after acceptance only that exact directory may be moved to trash with
`trash -- "$GEN_DIR"`, never a wildcard.

**Report/review/commit:** report required fields plus optimizer state inventory,
RNG split tree, maximum five-step errors, and one-call advancement matrix.
Spec review
compares exact `embodied/jax/opt.py` order; quality review checks atomic state
updates and JIT-safe pytrees. Fix/re-review to zero Critical/Important before
`feat(dreamer): add laprop training step`.

## Task 6: Implement DMC Vision and Proprio environments

Tasks 6a-6c are sequential implementer/review/commit units. Sequential symbol
overlap is allowed; concurrent editors are forbidden.

### Task 6a: Single-environment spec, action, boundary, and modalities

**Owned symbols/files:** create `dmc.py` and
`tests/test_dreamer_v3_dmc_contract.py`; edit
`pyproject.toml:[project.optional-dependencies].dmc` to require exactly
`dm-control==1.0.17` and `mujoco==3.1.3`, edit the pytest marker table to add
`real_dmc`, and update `uv.lock`. Task 6a is the sole owner of all dependency
and lock edits in Task 6. In `dmc.py` it is also the sole owner of `DMC_TASKS`,
the one literal ordered mapping from canonical ID to load target and camera
default/bounds; `DMC20_ORDER = tuple(DMC_TASKS)` is derived rather than a second
table. `DMC_STATE_SCHEMA` is the sole state-schema authority. Task 6a must
create `DMCState` as the public five-field `TypedDict` from Architecture
section 9 and the private closed nested TypedDicts for compatibility, spec,
mutable physics, MT19937, TimeStep, and observation trees. No field is `Any`.
This task owns the declarations and an AST/type gate; Task 6b consumes them
when it adds methods. State trees are direct primitive/NumPy serialization
values, not objects requiring a callback decoder. Task 6a also owns `DMCSpec`, single-environment
construction/spec helpers, `reset`, `step`, observation/action conversion,
`DMCEnvironment.close`, and constructor unwind. The production schema is a
reviewed literal source table, never imported from
`tests/fixtures/dreamer_v3/dm_control_1_0_17_state_schema.json`.
`test_production_schema_matches_canonical_fixture` recursively proves exact
parity with the frozen fixture. Later tasks consume but neither redefine nor
duplicate any authority. Task 6a does not own state capture/restore, vector
composition, or vector lifecycle aggregation. The shared float32 pixel adapter
is unchanged.
The schedule is named exactly `wm_marl_seedsequence_v1`, one bounded native
environment-integration/reproducibility exception shared by both profiles.
Source-conformance REDs at both pins prove DMC omits `use_seed`, `make_env`
forwards no seed, official `DMC` accepts no seed and calls
`suite.load(domain, task)`, and locked dm-control defaults to
`RandomState(None)`. These tests declare that the native schedule is not
paper/current algorithm behavior or an official-DMC translation; `DMCSpec`'s
existing public/base/child fields are its only serialized representation.
**First RED:** fail the exact public `DMCState` declaration/field order and
nested no-`Any` inventory, exact DMC20 mapping, paper repeat/cameras, Vision uint8 and
Proprio specs, terminal/truncation/following-first chronology, and clip/scale
ordering. `DMCEnvironment.step` receives the full environment-action mapping
with model-action leaves and a boolean `reset` leaf; vector step later uses the
same mapping tree with a leading environment axis and returns actual native
step count (zero reset, `1..action_repeat` for control, including early break on
last/terminal). Both production profiles require action repeat 1. Failure injection after
every acquisition proves constructor unwind closes all acquired task/physics/
viewer resources; close is idempotent and reset/step after close fail. Backend
metadata, imported MuJoCo, and the lock must all match the exact pins before an
environment is constructed. **Implementation order:** dependency/lock pins,
canonical literal task/camera/state-schema authority, concrete `DMCState` type,
and its fixture parity RED,
specs, resource acquisition with unwind, single close, suite construction,
modality conversion, full action-tree validation, boundary ordering. **Focused
acceptance:** `uv run pytest -q
tests/test_dreamer_v3_dmc_contract.py tests/test_dmc_pixel_adapter.py`, then
scoped Ruff check/format. Fresh spec and quality reviews must reach zero
Critical/Important before `feat(dreamer): add single dmc environment`.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv sync --extra dev --extra dmc
uv run python -c "import importlib.metadata as m, mujoco; assert m.version('dm-control')=='1.0.17'; assert m.version('mujoco')=='3.1.3'; assert mujoco.__version__=='3.1.3'"
uv run python -c 'from pathlib import Path; text=Path("uv.lock").read_text(); assert "name = \"dm-control\"\nversion = \"1.0.17\"" in text; assert "name = \"mujoco\"\nversion = \"3.1.3\"" in text'
uv run pytest tests/test_dreamer_v3_dmc_contract.py tests/test_dmc_pixel_adapter.py -q
uv run python -c 'import ast; from pathlib import Path; tree=ast.parse(Path("src/world_marl/dreamer_v3_baseline/dmc.py").read_text()); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="DMCState"); names=[n.target.id for n in cls.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)]; assert names==["compatibility","dmc_spec","format","format_version","mutable"]; assert "Any" not in ast.unparse(cls)'
uv run ruff check src/world_marl/dreamer_v3_baseline/dmc.py tests/test_dreamer_v3_dmc_contract.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/dmc.py tests/test_dreamer_v3_dmc_contract.py
uv lock --check
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/envs/dmc.py >/dev/null
```

### Task 6b: Complete state serialization and transactional restore

**Owned symbols/files:** edit only `DMCEnvironment.state_dict`,
`DMCEnvironment.from_state`, and private
single-environment state encode/validate/apply helpers in `dmc.py`; create only
`tests/test_dreamer_v3_dmc_state.py`. Task 6b owns no dependency, lock, task,
camera, schema-authority, vector class, vector API, or vector lifecycle edit. It
consumes the exact frozen production authority from Task 6a and may not add an
unreviewed field or wildcard.
**First RED:** dynamically validate MuJoCo 3.1.3
`mjtState.mjSTATE_INTEGRATION` numeric spec 8191 and `mj_stateSize`; enumerate
the full `mj_getState` bytes/dtype/shape, exact five-part
MT19937 state, current TimeStep, `_step_count`, `_reset_next_step`, static
compatibility, `physics.legacy_step is true`, bounds, structured `DMCSpec`, and
every literal task/model leaf in that schema. Integration `ctrl` and driver
pending action remain the sole action owners. Malformed candidates leave live
state unchanged; changed integration plus fractional RNG position is rejected
before construction. Exact Python types, task-bounded cameras, scalar encoded
shape/dtype, and reachable FIRST/MID/LAST/count/reset/reward/discount invariants
are checked before construction. A real restored next step equals uninterrupted
execution.
**Implementation order:**

`validate_closed_candidate -> construct_locked_task -> copy_complete_model_arrays -> mj_setState(INTEGRATION) -> mj_step1(legacy_step=True) -> restore_task_rng_and_mutable_task_fields -> restore_environment_counters_and_adapter_current_time_step -> clear_only_enumerated_derived_caches`.

Validation covers all leaves before construction or mutation; the one
`mj_step1` is immediate after `mj_setState`. After the final cache clear, one
validated single-environment candidate is returned.
`DMCEnvironment.from_state` is a nonmutating single-environment replacement
constructor. Candidate ownership
stays with the cleanup stack until return; rejection closes the candidate.
`state_dict` returns the Task-6a `DMCState` with newly allocated containers and
owned copies of every array. The restore entrypoint validates and copies that
exact type without mutating or retaining caller inputs; fixture hashes and
paths are rejected as unknown production keys.
Tests cover every single-environment validation, construction,
candidate-cleanup, and return boundary and prove no live partial mutation.
`Physics.get_state()` alone is forbidden. **Focused acceptance:** `uv run pytest -q
tests/test_dreamer_v3_dmc_state.py tests/test_dreamer_v3_dmc_contract.py`, then
scoped Ruff check/format. Fresh spec and quality reviews must reach zero
Critical/Important before `feat(dreamer): serialize dmc environment state`.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest tests/test_dreamer_v3_dmc_state.py tests/test_dreamer_v3_dmc_contract.py -q
uv run ruff check src/world_marl/dreamer_v3_baseline/dmc.py tests/test_dreamer_v3_dmc_state.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/dmc.py tests/test_dreamer_v3_dmc_state.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/envs/dmc.py >/dev/null
```

### Task 6c: Vector state, construction, lifecycle, and real 40-case matrix

**Owned symbols/files:** create `DMCVectorEnvironment`, its vector
construction/step/close helpers, `DMCVectorEnvironment.state_dict`, and
`DMCVectorEnvironment.from_states` in `dmc.py`; create only
`tests/test_dreamer_v3_dmc_vector.py`. Task 6c consumes Task-6a pins and literal
authorities and Task-6b single-environment state APIs; it owns no dependency,
lock, task, camera, schema-authority, or single-environment state edit.
**First RED:** stable seeds, the same batched environment-action mapping,
ordered state capture/restore, partial vector construction unwind, idempotent
vector close, all-child atomicity, and all-child aggregate close failure. Every
candidate child remains owned by one cleanup stack until the complete vector is
returned; a rejected child closes all staged candidates. The caller assigns the
new vector as active before it closes the old one: old vector closes only
after ownership transfer. If old close raises, every old child is still
attempted, the aggregate close error is reported, and the new vector remains the
active valid owner. This task owns DMC component behavior only. The named native
seed formula for policy `wm_marl_seedsequence_v1` is exactly the documented
`SeedSequence([base_seed, child_index]).generate_state(1, uint32)` formula.
The public domain is `0 <= public_seed <= 2**32 - 1 - 10_000`, checked before
`np.uint32` conversion; `train_base_seed = public_seed` and
`evaluation_base_seed = public_seed + 10_000`. Direct behavioral REDs cover the
named policy, role/child distinctness, vector-count stability, and same-seed
reproducibility. They cover vector `state_dict`/`from_states` exact continuation
and lifecycle, and prove an `expected_specs` mismatch is rejected before any
child construction. Trace-parity cases inject and record the same
explicit native child seed on both sides. They also prove different public
seeds retain distinct DMC identities while replay remains at its independent
fresh seed-zero state.
The non-skipped real test covers all 20 tasks x 2 modes, including real render/
schema, state roundtrip, and exact next-step equality. Vision runs in a separate
process whose render-capable backend is explicitly selected and validated by a
successful real render; it never inherits Task-0b's child-only
`MUJOCO_GL=off`. Installed metadata and the lock are reverified but not edited
in this task: the frozen checks assert `dm-control==1.0.17`, `mujoco==3.1.3`,
and the matching `uv.lock` entries. Its checkpoint state is a fresh
`list[DMCState]`; each child mapping/array is independently owned. The closed
inverse accepts only that primitive list and retains no candidate references;
the `expected_specs` runtime argument may remain a tuple because it is not
serialized. A public-Flax roundtrip and mutation-after-capture RED cover the
complete vector record. This component test does not implement or claim any
later payload, restore-manager, or run-coordination behavior. **Implementation
order:** vector construct/unwind, action tree and
seeds, vector `state_dict`/`from_states`, all-child atomicity
and ownership-transfer/aggregate cleanup tests, vector lifecycle, real matrix.
**Focused acceptance:**

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv sync --frozen --extra dev --extra dmc
uv run python -c "import importlib.metadata as m, mujoco; assert m.version('dm-control')=='1.0.17'; assert m.version('mujoco')=='3.1.3'; assert mujoco.__version__=='3.1.3'"
uv run python -c 'from pathlib import Path; text=Path("uv.lock").read_text(); assert "name = \"dm-control\"\nversion = \"1.0.17\"" in text; assert "name = \"mujoco\"\nversion = \"3.1.3\"" in text'
uv run pytest -q -m "not real_dmc" tests/test_dreamer_v3_dmc_contract.py tests/test_dreamer_v3_dmc_state.py tests/test_dreamer_v3_dmc_vector.py tests/test_dmc_pixel_adapter.py
uv run pytest -q -m real_dmc tests/test_dreamer_v3_dmc_vector.py::test_dmc20_all_modes_real_contract
uv run python -c 'from world_marl.dreamer_v3_baseline.dmc import DMC20_ORDER; cases=[(t,m) for t in DMC20_ORDER for m in ("vision","proprio")]; assert len(cases)==len(set(cases))==40; print("real DMC cases",len(cases))'
uv run ruff check src/world_marl/dreamer_v3_baseline/dmc.py tests/test_dreamer_v3_dmc_contract.py tests/test_dreamer_v3_dmc_state.py tests/test_dreamer_v3_dmc_vector.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/dmc.py tests/test_dreamer_v3_dmc_contract.py tests/test_dreamer_v3_dmc_state.py tests/test_dreamer_v3_dmc_vector.py
uv lock --check
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/envs/dmc.py >/dev/null
```

Fresh spec and quality reviews must reach zero Critical/Important before
`feat(dreamer): add vector dmc lifecycle`.

## Task 7: Interleave collection, evaluation, and training

Tasks 7a-7c are sequential implementer/review/commit units and own no artifact
writer or checkpoint manager.

### Task 7a: Official limiter, serializable driver state, and runner initialization

**Owned symbols/files:** create `driver.py`, `tests/test_dreamer_v3_limiter.py`,
and `tests/test_dreamer_v3_driver_state.py`; create `DreamerRunner` and own only
its fresh constructor `DreamerRunner.__init__`, cold classmethod
`DreamerRunner.from_state`, and their dependency/state wiring.
Also own
`SamplesPerInsertLimiter`, `PendingReplayRow`, `CadenceKind`, `CadenceRequest`,
`ActiveReport`, `ActiveEvaluation`, aggregation-window classes, `DriverState`,
and their primitive `state_dict()`/closed `from_state()` serializers. The
limiter record contains exact configuration plus `size`/`avail`. The complete
Driver record projects pending rows and cadence FIFOs to lists, active cadence
to a closed string-tagged mapping or `None`, and every window, carry, call
counter, summary, action, and accumulator to plain mappings/lists/primitives/
numeric arrays. Inverses validate and copy before runtime reconstruction and
reject tuple, Python/Flax dataclass, `FrozenDict`, arbitrary object, alias, and
unknown tag leaves. Public-Flax roundtrip/nonretention tests cover every active
tag and nonempty FIFO. Task 7a does not implement `DreamerRunner.advance`,
evaluation/cadence service, `run`, or `close`, and does not edit any Agent,
optimizer, replay, or DMC implementation. The exact constructor is
`DreamerRunner(agent, train_state, replay, train_environment,
evaluation_environment, limiter, run_config, sequence_shape)`. It validates
those already-created owners without deriving a model key, stepping an
environment, sampling replay, or changing a limiter. `DriverState` solely owns
`pending_rows: tuple[PendingReplayRow,...]`, signed-int64
`pending_sample_credits` in `[0,batch_size-1]`, boolean
`pending_update_demand`, ordered `pending_cadence_requests`, and scheduler
scalars `policy_call_counter: np.int64` and
`batch_seed_counter: np.int64`. The policy counter is shared by collection and
evaluation; the batch counter is shared by successful train and report calls.
No carry subtree or train state stores a recurring model key.

The exact closed inverse is
`DriverState.from_state(state, agent, run_config, sequence_shape, observation_spaces, action_spaces)`.
The bound agent closes carry schemas. `run_config` supplies train/evaluation
vector sizes and cadences; `sequence_shape` supplies `B/K/T/consecutive`; the
trusted spaces supply row/action leaves; and the bound agent/train resolved
identity supplies the canonical seed when later runner calls derive roots.
`RunConfig` owns no seed or batch/sequence shape. There is no duplicate
DriverState seed leaf: the state stores only both call counters, and later
runner calls equality-check the bound identity's `resolved.config.seed` before
deriving official outer roots. `ActiveReport` restoration must call
`ReplayBatch.from_state(state, transition_spaces, latent_spaces, expected_batch_size, expected_time_length)`
for its staged batch. An `inspect.signature` RED requires those exact arguments
and rejects an empty inverse, generic dependency record, or candidate-described
dimensions.
**First RED:** exact pinned limiter configuration is immutable
`samples_per_insert`, `tolerance`, `minsize`; state is `size` and
`avail = -minsize`; bounds are `min_avail=-tolerance` and
`max_avail=tolerance*samples_per_insert`; translate exact `want_insert()`,
`want_sample()`, `insert()`, and per-sequence `sample()`. Construction is
`train_ratio / batch_length`, `4 * batch_size`, and
`batch_size * replay.raw_length`; one batch consumes `batch_size` credits. A
nonzero-context, `consecutive > 1` case proves minsize uses train raw
`K + T * consecutive`, not trimmed `T`, while an independent report case uses
`K + report_length * report_consecutive`; both prove
`SequenceShapeConfig` is the sole owner. Cover
minsize debt, both pressure directions and boundaries, and executable paper-
default cases `samples_per_insert=4`, `tolerance=64`: insertion at
`size >= minsize, avail = 252` and a partial 16-credit batch at `avail = -63`.
Exact restore covers the FIFO, partial credits, update demand, cadence queue,
the sole scheduler `active_cadence` tagged union, every named
collection/train/report/evaluation owner, and every
`replay_rows`, `control_steps`, `env_frames`, episode/window counter.
An active cadence contains no carry, call counter, action, or aggregation
window; named report/evaluation subtrees own their carries while scheduler owns
both call counters. `DreamerTrainState.update_count`
is the sole update counter. `driver.summary` persistently owns signed-int64
`imagined_transitions`, nullable finite `last_loss`, the latest completed
evaluation mean, and evaluation window identity. Atomic train/evaluation
commits update it; a log-window reset does not change it. REDs perturb each
candidate duplicate and include mid-active restore.
Train and evaluation environments are not DriverState leaves.
Reset rows consume zero physical frames; control rows add the environment's
actual native-step count returned by the adapter to the physical control-frame budget (one in
both shipped profiles, at most `action_repeat` before an early last/terminal).

The literal first RED is
`test_dreamer_runner_initializes_exact_state`. The initial environment-action
mapping for both vectors has every model-action leaf zero with its declared
`[N,*A]` dtype/shape and `reset=np.ones([N], bool)`; no environment reset has
yet run. Collection/evaluation carries are exact `agent.initial(N)` values and
train/report carries are exact `agent.initial(batch_size)` values, including
zero previous actions. Both call counters are exact scalar `np.int64(0)` and no
outer key is materialized or stored. Episode accumulators and return windows
are zero/empty. The partial
scheduler is exactly empty `pending_rows`, zero `pending_sample_credits`, false
`pending_update_demand`, empty `pending_cadence_requests`, null
`active_cadence`; all counters, event sequence, window IDs/sums/counts, and
summary imagined transitions are signed-int64 zero; nullable loss/evaluation
summary fields are null. Each next evaluation/report/log/checkpoint threshold
equals its positive configured period. The RED also perturbs every constructor
shape/dtype/dependency mismatch and proves failure before any owned input
changes.

The second literal RED is `test_dreamer_runner_from_state_is_cold_and_exact`.
It freezes the signature
`DreamerRunner.from_state(agent, train_state, replay, train_environment,
evaluation_environment, limiter, run_config, sequence_shape, driver_state)`.
The method validates every restored driver action/carry/counter/cadence and
its dependency relationships before allocating without `__init__`; it performs
only reference assignments plus an owned driver-state copy. Spies prove it
does not initialize state, reset/step environments, derive a key, query/mutate
replay or limiter, mutate inputs, or retain caller-mutable containers. A
failure closes nothing because the Task-8c staging caller remains owner.
The constructor and cold inverse equality-check the agent/train canonical seed;
a seed mismatch fails before construction and before any outer root is derived.

**Implementation order:** limiter transition/state, scheduler records, windows,
driver schema and transactional restore, exact initial actions/carries/counters,
fresh constructor wiring, then cold `DreamerRunner.from_state`. **Focused acceptance:** `uv run pytest -q
tests/test_dreamer_v3_limiter.py tests/test_dreamer_v3_driver_state.py`, then
scoped Ruff check/format. Fresh spec and quality reviews must reach zero
Critical/Important before `feat(dreamer): add official online limiter state`.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest tests/test_dreamer_v3_limiter.py tests/test_dreamer_v3_driver_state.py -q
uv run python -c 'import inspect; from world_marl.dreamer_v3_baseline.driver import DriverState; assert list(inspect.signature(DriverState.from_state).parameters)==["state","agent","run_config","sequence_shape","observation_spaces","action_spaces"]'
uv run ruff check src/world_marl/dreamer_v3_baseline/driver.py tests/test_dreamer_v3_limiter.py tests/test_dreamer_v3_driver_state.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/driver.py tests/test_dreamer_v3_limiter.py tests/test_dreamer_v3_driver_state.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/core/limiters.py >/dev/null
```

### Task 7b: Deterministic cooperative collection/training scheduler

**Owned symbols/files:** create the complete frozen `RunnerOutput` declaration,
then edit only collection/train capacity helpers and `DreamerRunner.advance`;
create
`tests/test_dreamer_v3_driver_turn.py`. Task 7b must not edit `agent.py`,
`optimizer.py`, `replay.py`, or `dmc.py`, including already-owned Agent,
train-step, replay, and vector boundaries. A discovered missing boundary routes
to a fresh smallest-owner fixer with its own literal files, focused tests, Ruff
check/format, review, and commit before Task 7b resumes. The controller's global
ownership audit rejects any Task-7b diff outside `driver.py` and the named test;
it also requires every conditional fixer path to have its own scoped acceptance.
`RunnerOutput` has exactly five tuple fields in constructor order: `metrics`,
`scores`, `open_loop`, `evaluations`, and `cadence_requests`. Each defaults to
an empty tuple, so `RunnerOutput()` is the exact empty output; mappings and
numeric arrays are copied before a nonempty result is constructed. It is never
serialized and has no callback or methods beyond frozen record construction.
The literal declaration RED and future AST gate are
`test_runner_output_has_complete_frozen_schema`.
**First RED:** after the declaration RED, trace
step-policy-mask-FIFO-insert-limiter-sample-train-writeback, previous action,
final-row zeroing, online-first training, and one bounded quantum per call. The
deterministic cooperative scheduler uses stable insertion/sample/collection
priority, must stage and prevalidate replay append/identity/eviction before
limiter mutation, and preserves exactly one `insert()` per committed row and
exactly one `sample()` per reserved sequence. It serializes partial scheduler
state rather than blocking. REDs execute `size >= minsize, avail = 252` and
`avail = -63`, require collection to release a blocked partial batch, four
complete updates from the exact initial state `size >= minsize, avail = -64`:
16 inserts reach zero and four 16-credit batches return to `-64`, with every
intermediate value asserted. The scheduler must drain all currently owed
full updates before optional collection, stable pending-row/online-queue/sample
order, safe stop with partial credits, and uninterrupted/resumed equality.
Final-credit replay sample/train/writeback uses bounded replay mutation plans
and derives an outer key from the current `batch_seed_counter`, calls
`train_step(agent, state, carry, batch, outer_seed)`, then atomically swaps the
functional train state and increments that counter exactly once; a
diagnostic raises without a later sample/checkpoint and returns no new train
state/writeback/counter. Collection similarly derives its policy key from
`policy_call_counter` and increments it with the validated policy carry/action
commit. A later environment failure terminates the live process without
publishing a new checkpoint; a cold resume from the previous durable
`policy_call_counter` deterministically replays that uncheckpointed call.
Owner-local wraparound preflight occurs only at each actual
mutation boundary. **Implementation order:** pure guards, bounded row mutation
plan/commit, one-credit reservation, bounded final-credit replay plans plus
functional train-state swap,
collection-for-progress, copied direct `RunnerOutput`. **Focused acceptance:** `uv run pytest -q
tests/test_dreamer_v3_driver_turn.py tests/test_dreamer_v3_limiter.py
tests/test_dreamer_v3_train_step_parity.py tests/test_dreamer_v3_replay.py`,
then scoped Ruff check/format. Fresh spec and quality reviews must reach zero
Critical/Important before `feat(dreamer): interleave collection and training`.

Before limiter, replay RNG, selector, or stream mutation, the runner calls the
nonmutating `can_sample_batch(mode)`. Unavailable training falls through to
collection. REDs cover two writers with `B=1,T=4,R=4`, zero-credit/no-item,
and separate train/report readiness.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest tests/test_dreamer_v3_driver_turn.py tests/test_dreamer_v3_limiter.py tests/test_dreamer_v3_train_step_parity.py tests/test_dreamer_v3_replay.py -q
uv run python - <<'PY_AST'
import ast
from pathlib import Path
tree = ast.parse(Path("src/world_marl/dreamer_v3_baseline/driver.py").read_text())
node = next(value for value in tree.body if isinstance(value, ast.ClassDef) and value.name == "RunnerOutput")
fields = [value.target.id for value in node.body if isinstance(value, ast.AnnAssign) and isinstance(value.target, ast.Name)]
assert fields == ["metrics", "scores", "open_loop", "evaluations", "cadence_requests"], fields
PY_AST
uv run ruff check src/world_marl/dreamer_v3_baseline/driver.py tests/test_dreamer_v3_driver_turn.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/driver.py tests/test_dreamer_v3_driver_turn.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/run/parallel.py >/dev/null
```

### Task 7c: Evaluation, cadence, capacity, and lifecycle composition

**Owned symbols/files:** edit only runner evaluation/cadence/run/close methods
and their capacity helpers; create `tests/test_dreamer_v3_driver.py`.
**First RED:** policy-return evaluation advances only its own environment,
carry, episode counters, window, and the scheduler's shared
`policy_call_counter`; evaluation never owns or mutates replay,
model/optimizer, limiter, update count, batch counter, or the training report
stream. Collection and evaluation derive outer keys from consecutive values of
that one counter in stable service order. Report batches use the training
report stream and share `batch_seed_counter` with train calls: each successful
report derives the current legacy `uint32[2]` key and commits exactly one
counter advancement with returned carry/copied output; failure is no-advance.
Tests cover successive interleaved train/report calls and
mid-report resume coverage. `CadenceRequest` contains exact `CadenceKind`,
`threshold_env_frames`, `observed_env_frames`, and `event_sequence`; periods
use physical frames, first trigger one period, crossing catch-up queues every
threshold, and simultaneous kinds use evaluation/report/log/checkpoint order.
The scheduler drains already-produced rows and their earned updates, then gives
the serialized `active_cadence` and every due cadence one bounded service
quantum before optional collection. It must pause collection and training while an active report or
evaluation completes. A target-48/cadence-16 trace proves evaluation, report,
log, and checkpoint run at frame 16, not only at frame 48; a mid-evaluation
restore and mid-report resume are identical to uninterrupted execution.

`RunConfig.log_every` is a positive physical-frame cadence: 1,000 for both
production profiles and `log_every=16` for `debug-local-v1`; it is not a public
CLI override. The log next threshold is serialized. A log append emits exact
`metric_means`/`metric_counts`, then flushes and resets the window. The frame 16
trace checks this physical cadence. A report period below `R` is deferred while
collection can make its report stream ready; at a reached bound it becomes
`skipped-unavailable` without advancing its batch counter.

Sampling or computation failure advances neither report carry nor batch
counter; successful compute atomically advances both named owners and stages
copied output. The named RED is `test_report_compute_failure_no_advance`. Task 7 owns
sampling, computation, and copied staging only; writer publication, cold-resume
replay after publication failure, and failure after a reached stop bound belong
exclusively to Task 8c.

Target/debug-stop physical frames are lower bounds for full synchronous vector
steps. Stop after the first full-vector quantum whose actual native-frame sum
reaches/crosses the bound; shipped repeat-1 bounded overshoot is `[0,N-1]` and
reset-only quanta may add zero. Record requested and actual frames and queue all
crossed cadences. REDs include `N=2,target=3`, mixed reset/control, debug stop,
cadence crossing, and resumed-run equality. Terminal/reset/multi-episode traces prove
`replay_rows`, `control_steps`, and `env_frames` units; windows and pending
cadence requests restore mid-window. Normal/error/interruption exits close both
vectors in `finally`. Returned metrics/scores/open-loop/evaluation values are
copied direct runner output, never callbacks. **Implementation order:** isolated
policy-return evaluation, shared counter service, cadence/stop, owner-local capacity checks, run
composition, finally cleanup. Overshoot derives independently of `RunStatus`
and is zero only before any bound; Task 8c verifies retained summaries across
publication failures after a reached bound.
**Focused acceptance:** `uv run pytest -q tests/test_dreamer_v3_driver.py
tests/test_dreamer_v3_driver_turn.py tests/test_dreamer_v3_driver_state.py
tests/test_dreamer_v3_dmc_vector.py`, then scoped Ruff check/format. Fresh spec
and quality reviews must reach zero Critical/Important before
`feat(dreamer): compose online runner lifecycle`.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest tests/test_dreamer_v3_driver.py tests/test_dreamer_v3_driver_turn.py tests/test_dreamer_v3_driver_state.py tests/test_dreamer_v3_dmc_vector.py -q
uv run ruff check src/world_marl/dreamer_v3_baseline/driver.py tests/test_dreamer_v3_driver.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/driver.py tests/test_dreamer_v3_driver.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/run/parallel.py >/dev/null
```

## Task 8: Produce direct artifacts and exact checkpoint/resume

Tasks 8a1-8d are sequential implementer/review/commit units. No generic callback,
provenance graph, or file-authentication subsystem is permitted.

### Task 8a1: Direct manifest, summary, and append writers

**Owned symbols/files:** create `artifacts.py` and
`tests/test_dreamer_v3_artifact_logs.py`. Own `RunManifest`, `RunSummary`, the
`ArtifactWriter` constructor, manifest/summary writes, metrics/scores append/
flush/close, writer byte offsets, and lifecycle enum. **First RED:** direct repository-style
schemas, canonical JSONL, fsync, atomic summary, descriptor cleanup, and exact
`ArtifactWriter.state_dict()` offset state fail. Summary REDs cover exact
`RunStatus`, nullable pre-evaluation return/loss, most-recent completed
evaluation window in `evaluation_return`, last committed loss, actual
`B*Kstart*H` imagined-transition
accumulation per committed update, the exact learning-gate formula, evaluation
length in native control steps, exact `learning_gate_passed`, and separate `requested_target_env_steps`,
nullable `requested_stop_env_steps`, `actual_env_frames`, and
`overshoot_env_frames`, plus exact `config_sha256`, nullable `debug_snapshot`,
and closed `runtime_overrides` identity copied from the resolved run. The manifest test asserts
`initial_checkpoint_generation` is null and its canonical key order. Metrics
rows assert exact `row_type = train_metrics`, `metric_means`, `metric_counts`,
`window_start_env_frames`, and `window_end_env_frames`; scores rows assert
`row_type = evaluation_scores`, `evaluation_file_start`, and exclusive
`evaluation_file_stop`. Canonical serialization uses `allow_nan=false`. Each
append trigger is one completed cadence; checkpoint/resume tests compare exact
JSONL rows and resumed append grouping.

`ArtifactWriter.state_dict()` is an exact fresh primitive mapping of immutable
run identity, the two scalar byte offsets, and the two scalar next-file
cursors. Its closed validator rejects handles/lifecycle enums, tuples,
dataclasses, `FrozenDict`, paths, aliases, missing/extra keys, or out-of-range
cursors and retains no candidate container. A public-Flax roundtrip RED precedes
artifact mutation tests; Task 8b receives only this owner-produced record.

`driver.summary` is the literal persistent owner of signed-int64
`imagined_transitions`, nullable finite `last_loss`, latest completed evaluation
mean, and evaluation window identity. An atomic train commit and atomic
evaluation commit update it; log-window reset does not change it.
train_updates derives directly from `DreamerTrainState.update_count` and this
owner is a literal checkpoint subtree.
The summary counter follows the official Agent start axis:
`Kstart = min(imag_last if imag_last != 0 else T, T)` after context trimming.
Both zero and nonzero `imag_last` cases add exactly `B*Kstart*H` per committed
update.
**Implementation order:** schemas, atomic JSON, append
writers, temp write, file fsync, rename, parent-directory `fsync`, flush/close,
state. **Focused acceptance:** `uv run pytest -q
tests/test_dreamer_v3_artifact_logs.py tests/test_dreamer_v3_agent.py`, then
scoped Ruff check/format. Fresh spec and quality reviews must reach zero
Critical/Important before `feat(dreamer): add direct artifact logs`.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest tests/test_dreamer_v3_artifact_logs.py tests/test_dreamer_v3_agent.py -q
uv run ruff check src/world_marl/dreamer_v3_baseline/artifacts.py tests/test_dreamer_v3_artifact_logs.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/artifacts.py tests/test_dreamer_v3_artifact_logs.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/run/eval_only.py >/dev/null
```

### Task 8a2: Atomic numbered outputs, cursors, inspection, and resume reconciliation

**Owned symbols/files:** edit only numbered open-loop/evaluation writers,
`ArtifactWriter.state_dict()`, cold `ArtifactWriter.resume(run_dir,
writer_state)`, and `inspect_run_state`; create
`tests/test_dreamer_v3_artifact_files.py`.
**First RED:** each output uses temp write, file fsync, rename, and parent-
directory `fsync` before advancing cursors. Cold resume first validates identity,
offsets, cursors, file lengths, complete files below each cursor, and all files
at/above each cursor without handles. It directly truncates and `fsync`s active
JSONL to checkpointed prefixes, removes numbered outputs at/above their
checkpointed cursors and temporary siblings, `fsync`s changed directories, and
only then opens append handles at the exact offsets. Repeating cold resume is
idempotent. Representative injected failures before reconciliation and after
truncation/before handle-open leave no sparse holes and converge to the exact
prefix on retry. No archive, generalized file transaction, or extra lifecycle
state is introduced.

Mode-specific REDs bind artifact behavior to the manifest: Vision report
outputs create a lazy `open_loop/` directory, nonempty numbered files, and a
positive cursor; Proprio report output is empty, keeps its cursor exactly zero,
and leaves that directory absent. A fabricated Proprio numbered/video output
is rejected by write, resume, and `inspect_run_state`. Evaluation numbered
files and cursor are nonempty after evaluation in both modalities. Restore
requires every lower file for a positive cursor and removes only files at or
above the checkpoint cursor.
**Implementation order:** numbered schemas, atomic writer,
cursor state, closed validation, direct reconciliation, handle open, inspection.
**Focused
acceptance:** `uv run pytest -q tests/test_dreamer_v3_artifact_files.py
tests/test_dreamer_v3_artifact_logs.py`, then scoped Ruff check/format. Fresh
spec and quality reviews must reach zero Critical/Important before
`feat(dreamer): add atomic artifact cursors`.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest tests/test_dreamer_v3_artifact_files.py tests/test_dreamer_v3_artifact_logs.py -q
uv run ruff check src/world_marl/dreamer_v3_baseline/artifacts.py tests/test_dreamer_v3_artifact_files.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/artifacts.py tests/test_dreamer_v3_artifact_files.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/run/train.py >/dev/null
```

### Task 8b: Complete checkpoint payload and manager

**Owned symbols/files:** create `checkpoint.py` and
`tests/test_dreamer_v3_checkpoint.py`. Own the closed `CheckpointPayload`,
the short codec wrapper, and
`CheckpointManager.save/restore_candidate`; do not edit driver, artifacts,
`pyproject.toml`, or `uv.lock`. Task 8b owns no dependency or lock edit: it
uses the already-direct Flax dependency's public
`flax.serialization.msgpack_serialize` and
`flax.serialization.msgpack_restore` functions.

**First RED:** publish and recursively validate Architecture section 10's
literal payload tree. Train/evaluation environments each occur exactly once;
collection, train, report, and evaluation carries occur exactly once under
their named driver subtree; `policy_call_counter` and `batch_seed_counter`
occur exactly once in scheduler; active cadence occurs only in scheduler; replay
identity cursors occur only in replay. Reject duplicate, conflicting, unknown,
or missing leaves before resource construction. The asserted literal names
include `train_environment`, `evaluation_environment`,
`collection_carry`, `train_carry`, `report_carry`, and
`evaluation_carry`. `DreamerTrainState.update_count` is the sole update
counter and `driver.summary` is a literal checkpoint subtree. Tests perturb
each owner boundary and exercise mid-active state.

Before any envelope RED, assemble one representative complete payload from the
real primitive `state_dict()` output of every owner and roundtrip it directly
through public Flax 0.10.4. Mutate the restored tree and prove no source owner
changes. Separate direct tests reject a tuple, Python dataclass, Flax struct
dataclass, `FrozenDict`, and an unsupported object at the owner/payload
validator. Task 8b does not convert any of them and has no generic object
converter; it accepts only the already-canonical dict/list/string/bytes/finite-
number/bool/None/NumPy-or-JAX-array language from Architecture section 3.2.

Codec REDs require only the versioned envelope and closed payload. Exact file
layout is eight ASCII bytes `WMDRM3CK`, version byte `0x01`, unsigned
big-endian uint64 body length, 32 raw SHA-256 body bytes, the Flax MessagePack
body, then EOF. Tests reject magic/version/length/hash mismatch, truncation, a
second value, incomplete suffix, and any trailing byte. Before encode, owner
validation builds a fresh closed tree with recursively UTF-8-sorted structural
string keys. Task 2b has already converted the two integer-key replay maps to
ordered record lists; public Flax directly canonicalizes replay's remaining
bytes-key `refs` map. Decode uses
`msgpack_restore`, rejects unknown extension/schema leaves, applies only the
four exact fixed-width big-endian adapters from Architecture
(`replay.next_chunk_id`, `replay.next_item_id`, and PCG64 `state`/`inc`),
validates all owners, and requires reserialized canonical bytes to equal the
body. There is no custom tag table, arbitrary logical-value grammar, Unicode
policy, input callback, or object constructor.

The exact boundary REDs roundtrip chunk cursor `1` and `2**128` in 17 bytes,
item cursor `0` and `2**63` in 8 bytes, and both PCG64 leaves `0` and
`2**128-1` in 16 bytes. They reject the wrong width/range at each exact path
and any such byte leaf elsewhere. Fresh-process tests encode identical
NumPy/JAX values to identical bytes in two processes and roundtrip bfloat16 via
Flax/JAX; JAX import is allowed. A literal unknown MessagePack extension body is
rejected without a registered application extension hook.

`CheckpointPayload.max_body_bytes(owner_schemas)` derives its bound from
resolved parameter/optimizer/carry/environment shapes, configured replay
capacity/chunk/spaces/writer/stream/queue maxima, vector sizes, exact closed
driver/artifact nodes, identity/config UTF-8 bytes, and the four fixed-width
adapters. The formula is exact expected leaf bytes plus expected UTF-8 bytes
plus `64 * closed_node_count + 1_048_576`. Encode must fit; decode rejects a
larger regular file before body decode and owner validation rejects any
cardinality/shape/byte excess. Tests cover the exact boundary and one byte over;
there are no unrelated 8-GB/node/string/map resource constants.

Checkpoint generation is a payload leaf, but the caller supplies no generation:
`save(snapshot_without_generation) -> (CheckpointPayload, Path)`. The manager
uses zero when `latest` is absent and otherwise the validated latest generation
plus one. It injects that same value into payload/name/pointer, atomically
replaces the numbered successor, directory-fsyncs, then atomically replaces and
directory-fsyncs exact UTF-8 `latest` as `{generation:020d}.ckpt\n`.
A failed unpublished successor is replaced by the same next save; no
directory-wide allocation subsystem exists. Representative failures before
numbered publication and after numbered rename but before pointer publication
leave `latest` unchanged; a successful retry publishes one successor.
Missing/malformed pointers or referenced files are fatal.
`restore_candidate(path, expected_identity, owner_schemas)` is pure and
validates config, DMC specs/cameras/backend, tree shapes/dtypes, and every owner
without artifacts or resource construction.
Task 8b owns pure payload identity/schema rejection for both complete DMC specs
and the named native policy `wm_marl_seedsequence_v1` before any resource
construction. Its tests perturb profile, role, public/base/child seeds, child
indices, cameras, backend, and policy-derived values independently in the
primitive payload and expected identity; rejection occurs before the first
constructor spy. It neither constructs a DMC child nor tests coordinator cleanup.
The expected identity's canonical seed, canonical config/hash, and both native
DMC-spec seed mappings must agree with `resolved.config.seed`; seed mismatch
fails before construction, payload owner allocation, or environment/model/replay
creation. Replay has no construction-seed identity or public-seed equality
projection. REDs perturb each true projection independently and require
rejection before the first constructor spy. Two public seeds retain distinct
DMC identities but identical fresh replay selector seed-0 state and sample sequence;
restoring complete advanced PCG64 state proves the advanced selector state
resumes at the exact next sample.

The behavioral RED saves a complete scripted owner state immediately before a
next action/replay sample/train transition, restores it in a fresh process, and
compares that next transition and every resulting owner tree with the
uninterrupted branch. This is checkpoint semantics, not byte-only codec
evidence. **Implementation order:** representative complete payload public-Flax
roundtrip and unsupported-object rejection, envelope over public Flax
serialization, four path adapters, closed payload/owner bound, pure candidate validation,
atomic manager, behavioral resume, representative corruption/failure cases.
Fresh spec and quality reviews must reach zero Critical/Important before
`feat(dreamer): add complete checkpoint manager`.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest tests/test_dreamer_v3_checkpoint.py tests/test_dreamer_v3_replay.py tests/test_dreamer_v3_driver_state.py -q
uv run python -c 'from importlib.metadata import version; from flax import serialization; assert version("flax")=="0.10.4"; assert callable(serialization.msgpack_serialize) and callable(serialization.msgpack_restore)'
uv run python -c 'import tomllib; deps=tomllib.load(open("pyproject.toml","rb"))["project"]["dependencies"]; assert "flax" in deps; assert not any(x.startswith(("msgpack","ml-dtypes")) for x in deps)'
uv run ruff check src/world_marl/dreamer_v3_baseline/checkpoint.py tests/test_dreamer_v3_checkpoint.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/checkpoint.py tests/test_dreamer_v3_checkpoint.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:embodied/run/train.py >/dev/null
```
### Task 8c: Direct run coordinator and restore composition

**Owned symbols/files:** create `RunnerRestoreCandidate` and `DreamerRunCoordinator`
in `driver.py`; create
`tests/test_dreamer_v3_coordinator.py`. Fresh-run coordinator methods are
`consume_output(output)`, `save_safe_point()`, `advance()`, and `close()`.
The sole resume entry is the closed classmethod
`DreamerRunCoordinator.resume(checkpoint_path, expected_identity,
resolved_config, observation_spaces, action_spaces, run_dir)`. It calls only
named production classmethods; no constructor registry, input callback, or
open-ended dependency mapping exists. The contradictory instance
`restore(checkpoint_path)` on a live runner does not exist.

`RunnerRestoreCandidate` has exact private constructor fields in order:
`train_state`, `replay`, `limiter`, `driver_state`,
`train_environment`, `evaluation_environment`, `runner`,
`cleanup_stack`, and `transferred=False`. Its closed classmethod is
`stage(payload, agent, resolved_config, observation_spaces, action_spaces)`;
`transfer()` disarms the stack and returns the runner exactly once; `close()`
reverse-unwinds only an untransferred candidate.
`DreamerRunCoordinator(runner, artifact_writer, checkpoint_manager)` stores
only those three validated references and performs no I/O/RNG/state transition.
Its complete method surface is `consume_output`, `save_safe_point`,
`advance`, `resume`, and `close`; close aggregates writer/runner cleanup.
The declaration REDs are
`test_restore_candidate_has_complete_ownership_schema` and
`test_run_coordinator_has_complete_surface`.

**First RED:** after declarations, fresh order is runner quantum -> consume
immutable output -> flush/state capture -> atomic checkpoint. Cold resume uses
the literal Architecture order:

1. construct the non-resource `DreamerAgent` without live train state;
2. call `DreamerTrainState.schema(agent, observation_spaces, action_spaces,
   resolved_config)`, derive all remaining owner schemas/bound, then statically
   decode and validate the payload with expected identity;
3. stage training then evaluation
   `DMCVectorEnvironment.from_states` under one cleanup stack;
4. call, in order,
   `DreamerTrainState.from_state(state, agent, observation_spaces,
   action_spaces, resolved_config)`,
   `DreamerReplay.from_state_dict`,
   `SamplesPerInsertLimiter.from_state_dict`,
   `DriverState.from_state(state, agent, run_config, sequence_shape,
   observation_spaces, action_spaces)`, and
   Task-7a `DreamerRunner.from_state`;
5. run `ArtifactWriter.resume(run_dir, writer_state)` for idempotent prefix
   reconciliation and handle open;
6. build the coordinator by no-fail assignments, transfer the runner once, and
   disarm candidate cleanup.

The stack registers training then evaluation vector closes, so rejection closes
evaluation then training and each vector attempts every child. The staged
runner only refers to those vectors; after transfer it becomes their sole
lifecycle owner. Validation/construction failure leaves the run tree untouched.
Artifact failure closes candidate vectors and any opened handles. Failure
injection covers each named acquisition and representative before/after
artifact reconciliation boundaries, exact reverse cleanup, no input mutation,
and no live-owner mutation. There is no fallible swap after artifact mutation.
No checkpoint publishes after runner, diagnostic, writer, reconciliation, or
manager failure; partial rows/credits and active cadence remain safe-point
state.

The resume-signature AST RED requires the six parameter names above and proves
`DreamerRunner.from_state` is called exactly once while `DreamerRunner.__init__`
is never called on the cold path. It also proves the cold path calls
`DreamerTrainState.initialize` zero times and preserves the exact scalar
`policy_call_counter` and `batch_seed_counter` through static decode, staging,
and runner construction. Before step 1, the current
`resolved_config.seed == resolved.config.seed` projection is checked against
the expected identity, decoded canonical config, and both native DMC specs;
seed mismatch fails before construction and no cleanup is needed. Replay state
has no public-seed identity field. Two public seeds have identical fresh replay
selector seed-0 state and sample sequence, while advanced selector state resumes
at the exact next sample from the complete restored PCG64 record.
The cold-resume expected-identity/spec/seed-policy mismatch tests call
`DreamerRunCoordinator.resume` and exercise `RunnerRestoreCandidate` only at its
production boundary. Every mismatch is rejected before any child construction
and no cleanup is needed; after staging begins, injected failures prove exact
reverse cleanup and no live-owner or run-tree mutation.
Focused behavioral tests cover
`test_report_writer_failure_cold_resume_replays`, post-crossing artifact failure,
and post-crossing checkpoint failure. Fresh spec and quality reviews
must reach zero Critical/Important before
`test(dreamer): prove exact resume identity`.
The cold path calls `DreamerTrainState.initialize` zero times.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest tests/test_dreamer_v3_checkpoint.py tests/test_dreamer_v3_artifact_files.py tests/test_dreamer_v3_coordinator.py tests/test_dreamer_v3_driver.py tests/test_dreamer_v3_driver_state.py -q
uv run ruff check src/world_marl/dreamer_v3_baseline/driver.py tests/test_dreamer_v3_coordinator.py
uv run ruff format --check src/world_marl/dreamer_v3_baseline/driver.py tests/test_dreamer_v3_coordinator.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/agent.py >/dev/null
```
### Task 8d: Uninterrupted-versus-resumed behavioral trace

**Owned files:** create `tests/test_dreamer_v3_resume_trace.py` only. Production
mismatches route to a fresh smallest-owner fixer and separate review/commit.
**First RED:** a test-local comparator catches an intentionally perturbed
optimizer leaf. Both branches receive a shared deterministic test `run_id` and manifest
bytes. Compare actions, samples, losses, all model/optimizer/auxiliary state,
replay IDs/queues/streams, limiter, partial rows/credits/update demand, carries,
the two call counters, replay/task generator states, counters/windows, writer
offsets, next-file cursors, artifacts,
evaluations, decoded checkpoints, and final state. One interruption is
immediately before a chunk rollover/item allocation; later IDs, samples, queue
order, links, and writeback must match. Both schedules checkpoint the interruption safe point, so restart
adds no artifact/generation. Active files equal the checkpointed prefixes and
numbered files begin below the restored next cursors. The trace also
resumes from the exhausted allocator sentinels, a partial
16-credit batch, `ActiveReport`, and `ActiveEvaluation`, and recursively proves
every environment/carry/call-counter/generator leaf has one owner. **Focused
acceptance:** `uv run pytest -q tests/test_dreamer_v3_resume_trace.py
tests/test_dreamer_v3_checkpoint.py tests/test_dreamer_v3_artifact_files.py
tests/test_dreamer_v3_coordinator.py tests/test_dreamer_v3_driver.py`, then scoped Ruff check/format. Fresh spec and
quality reviews must reach zero Critical/Important before
`test(dreamer): prove exact resume identity`.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest tests/test_dreamer_v3_resume_trace.py tests/test_dreamer_v3_checkpoint.py tests/test_dreamer_v3_artifact_files.py tests/test_dreamer_v3_coordinator.py tests/test_dreamer_v3_driver.py -q
uv run ruff check tests/test_dreamer_v3_resume_trace.py
uv run ruff format --check tests/test_dreamer_v3_resume_trace.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/agent.py >/dev/null
```

## Task 9: Replace public CLI and benchmark integration

Task 9 is four sequential nonoverlapping implementer-review-fixer-commit
units. Each owns one public boundary; no unit edits another unit's files.

### Task 9a: Production CLI, exports, and obsolete-module deletion

**Owned files:** `train_dreamer_v3_baseline.py`, `[project.scripts]` only in
`pyproject.toml`, Dreamer package `__init__.py`, deletion of exactly
`losses.py`, `models.py`, `imagination.py`, `training.py`, `validation.py`,
and new CLI/runtime export tests. **First RED:**
help/default/dry-run/resume/debug subprocess tests
hit the obsolete RSSM constructor and legacy exports. Resource spies also fail
until training and evaluation environments close in `finally` on success,
partial setup, step/evaluation/checkpoint error, and interruption.
**Implementation:** wire the accepted config/agent/runner/artifact/checkpoint
APIs; parse the exact allowed fields into `RuntimeOverrides` and call Task 1's
`resolve_dreamer_run` so merge order, cross-field validation, canonical JSON,
override map, deterministic noncanonical debug snapshot, and 48-frame CPU
identity remain Task-1 behavior. Make dry-run constructor-free, migrate exports/imports, delete obsolete
modules, and own environment lifetime with `finally`. **Focused acceptance:**
run only CLI/runtime-export tests, CLI help, one-file dry run, one-step local
debug, package import, and scoped Ruff check/format. Report help/schema/import
graph/close traces; fresh spec/quality review and fixer must reach 0
Critical/Important before `feat(dreamer): expose production online cli`.

The parser declaration is exactly
`parser.add_argument("--profile", choices=("paper", "upstream-current"), default="paper")`.
After parsing all fields, the CLI calls
`resolve_dreamer_run(mode=parsed.observation_mode, task=parsed.task, profile=parsed.profile, seed=parsed.seed, model=parsed.model, debug_local=parsed.debug_local, overrides=runtime_overrides)`.
Help/default tests prove omission selects paper, while explicit
`--profile upstream-current` remains the only CLI route to current behavior.

Task 9a's camera parser uses only `--dmc-camera-id` with `default=None`.
Omission must omit `--dmc-camera-id` from canonical argv and leave `DMCSpec`
to derive the effective task camera; an explicitly supplied integer, including
zero, is preserved in the environment identity and the runtime override key
`camera`.

Task 9a passes parsed `--seed` through the resolver's primary `seed=` argument,
not through `RuntimeOverrides`. The returned `resolved.config.seed` is the one
value used or equality-checked by fresh bootstrap, parameter initialization,
official counter roots, native train/evaluation `DMCSpec`s, canonical argv,
manifest, checkpoint identity, and cold-resume expected identity. Fresh replay
instead constructs its selector with literal zero, records no construction-seed
identity, and has no replay/public-seed equality check. Tests perturb each true
derived projection independently and prove seed mismatch fails before
construction. Two distinct public seeds produce distinct official parameter/counter roots and DMC derived identities, but identical fresh replay selector seed-0 state and sample sequence. The cold-resume CLI path restores complete PCG64 state and proves advanced selector state resumes at the exact next sample.

The CLI/resolver validates parsed `seed` as a non-bool Python integer satisfying
`0 <= seed <= 2**32 - 1 - 10_000` before NumPy conversion, then defines
`public_seed = resolved.config.seed`, `train_base_seed = public_seed`, and
`evaluation_base_seed = public_seed + 10_000`. Child environments use
`SeedSequence([base_seed, child_index])`; the manifest identity and checkpoint
identity preserve this mapping and reject a cold-resume mismatch.
This policy is named exactly `wm_marl_seedsequence_v1` and is a bounded native
environment-integration/reproducibility exception, not source-derived
paper/current behavior. Task 9 owns CLI, dry-run, and run-manifest DMC policy
identity: parser/default normalization, canonical argv, and manifest projections
must agree with the two complete `DMCSpec` values and the named policy. It does
not duplicate the Task-6 component implementation or Task-8 payload/coordinator
validators. Explicit-current cases remain in the CLI tests. Trace comparisons
inject and record the same explicit native child seed on both sides; `DMCSpec`
remains the only serialized policy representation. Tests prove the omitted-profile dry run
resolves profile `paper` and authority revision
`bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01`; canonical argv may explicitly
include normalized `--profile paper`.

The stop control is invocation-local and never becomes hidden durable history.
Task-9a REDs run target 48 with a current-invocation stop at 32 and require the
interrupted summary to report that stop and overshoot from 32. A cold resume
without a stop must immediately rewrite the derived summary from checkpointed
measured counters: `requested_stop_env_steps` becomes null and overshoot is
zero until target 48 is reached. The final resumed summary must equal the
uninterrupted target-48 summary when both tests inject the same run id and
deterministic inputs. A recursive assertion proves the stop field is absent
from config, manifest, checkpoint identity/payload, `DriverState`, replay, and
writer state. Separate failure tests use the current invocation's stop only if
that invocation crossed it; otherwise they use the target only after target
crossing.

Fresh CLI bootstrap constructs the non-resource Agent, then calls
`DreamerTrainState.initialize` exactly once before the fresh runner constructor;
`test_fresh_bootstrap_initializes_train_state_once` spies across the real CLI
entrypoint and also checks initialized state against `DreamerTrainStateSchema`.
It starts `policy_call_counter` and `batch_seed_counter` at scalar
`np.int64(0)`. Resume delegates only to Task-8c and calls
`DreamerTrainState.initialize` zero times.
Fresh bootstrap calls `DreamerTrainState.initialize` exactly once.

The local artifact contract is mode-specific. Vision open-loop cursor/files are
nonempty after a due report. Proprio open-loop cursor is zero and Proprio
open-loop directory is absent; any fabricated Proprio video is a failure.
The evaluation cursor/files are nonempty in both modalities after evaluation.
CLI/integration assertions inspect those conditions separately rather than
requiring one open-loop rule for both.
The exact Vision assertion contains
`s["next_file_cursors"]["open_loop"]>0`; the exact Proprio assertion contains
`s["next_file_cursors"]["open_loop"]==0`.

- Vision open-loop cursor/files are nonempty.
- Proprio open-loop cursor is zero.
- Proprio open-loop directory is absent.
- evaluation cursor/files are nonempty in both modalities.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
TASK9A_ROOT="$(mktemp -d /tmp/wm-marl-dreamer-v3-task9a.XXXXXX)"
uv run pytest tests/test_dreamer_v3_cli.py tests/test_dreamer_v3_runtime_exports.py -q
uv run world-marl-train-dreamer-v3-baseline --help
uv run world-marl-train-dreamer-v3-baseline --out-dir "$TASK9A_ROOT/dry" --observation-mode vision --task walker_walk --dry-run
uv run python -c 'from pathlib import Path; import json,sys; d=json.loads((Path(sys.argv[1])/"manifest.json").read_text()); assert d["profile"]=="paper"; assert d["authority_revision"]=="bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01"; argv=d["canonical_argv"]; i=argv.index("--profile"); assert argv[i+1]=="paper"' "$TASK9A_ROOT/dry"
uv run world-marl-train-dreamer-v3-baseline --out-dir "$TASK9A_ROOT/debug" --profile paper --observation-mode proprio --task cartpole_balance --debug-local --env-steps 2 --stop-after-env-steps 1 --num-envs 1 --batch-size 1 --batch-length 4 --eval-every 16 --eval-episodes 1 --report-every 16 --checkpoint-every 1 --seed 0
uv run python -c 'from pathlib import Path; import sys; from world_marl.dreamer_v3_baseline.artifacts import inspect_run_state; root=Path(sys.argv[1]); s=inspect_run_state(root); assert s["next_file_cursors"]["open_loop"]==0; assert not (root/"open_loop").exists(); assert s["next_file_cursors"]["evaluation"]>=0' "$TASK9A_ROOT/debug"
uv run ruff check src/world_marl/scripts/train_dreamer_v3_baseline.py src/world_marl/dreamer_v3_baseline/__init__.py tests/test_dreamer_v3_cli.py tests/test_dreamer_v3_runtime_exports.py
uv run ruff format --check src/world_marl/scripts/train_dreamer_v3_baseline.py src/world_marl/dreamer_v3_baseline/__init__.py tests/test_dreamer_v3_cli.py tests/test_dreamer_v3_runtime_exports.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/main.py >/dev/null
```

### Task 9b: Dreamer comparison translation and strict normalization

**Owned files:** `compare_visual_wm.py` and its tests only. The parser owns one
explicit `--dreamer-debug-local` comparison-level opt-in with default false and
`--dmc-camera-id` with `default=None`. **First RED:** total DMC20
task/modality/budget/camera translation and a real Dreamer child subprocess
fail; malformed raw summaries currently pass permissive loading; unsupported
arm names reach command construction. Camera REDs prove omitted quadruped input
derives effective camera 2, omitted nonquadruped input derives effective camera
0, and explicit integer 0 is preserved for quadruped instead of treated as
missing. Command REDs prove default dispatch must omit the debug child override
and must omit `--dmc-camera-id`; only explicit `--dreamer-debug-local` forwards
`--debug-local`, and only an explicitly supplied camera forwards its child
flag. **Implementation:** make `dreamer_v3_baseline` the only launchable
genuine pixel arm. Reject every unavailable arm before subprocess creation.
Add one strict
`normalize_summary(path, expected_identity=None, allow_debug=False)` for the
exact raw Dreamer schema and the existing primary comparison-row schema;
validate exact keys, types, finiteness, and identity. Every normalized Dreamer
row carries `config_sha256`, nullable `debug_snapshot`, and the exact closed
`runtime_overrides`. Default primary normalization rejects a nonnull
`debug_snapshot`; `allow_debug=True` is used only by the explicit debug smoke
path, labels the row `comparison_role="debug_local"`, and primary table/
aggregation code excludes it. `load_summary` is a thin caller. Preserve
summary-only aggregation. **Focused acceptance:** comparison tests,
`test_real_dreamer_comparison_arm_subprocess_summary`, the literal Dreamer
comparison command below, and scoped Ruff check/format. Fresh spec/quality
review and fixer must reach zero Critical/Important before
`feat(dreamer): integrate visual comparison arm`.

The strict raw-run inspection also validates canonical manifest key order and
`initial_checkpoint_generation`, exact JSONL rows (`train_metrics` and
`evaluation_scores`), `metric_means`/`metric_counts`, metric window start/stop,
evaluation file start/stop, `allow_nan=false`, append trigger counts, and
resumed append grouping before normalization.
The literal fields are `row_type`, `window_start_env_frames`,
`window_end_env_frames`, `evaluation_file_start`, and
`evaluation_file_stop`; their order is the canonical key order.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
TASK9B_ROOT="$(mktemp -d /tmp/wm-marl-dreamer-v3-task9b.XXXXXX)"
uv run pytest tests/test_compare_visual_wm.py -q
uv run pytest tests/test_compare_visual_wm.py::test_real_dreamer_comparison_arm_subprocess_summary -q
uv run python -m world_marl.scripts.compare_visual_wm --arm dreamer_v3_baseline --env dmc-pixels:ball_in_cup/catch --out-dir "$TASK9B_ROOT" --collect-steps 2 --num-envs 1 --eval-episodes 1 --seed 0 --image-size 64 --dreamer-debug-local
uv run ruff check src/world_marl/scripts/compare_visual_wm.py tests/test_compare_visual_wm.py
uv run ruff format --check src/world_marl/scripts/compare_visual_wm.py tests/test_compare_visual_wm.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/configs.yaml >/dev/null
```

### Task 9c: Supported benchmark matrix and separate baselines

**Owned files:** `benchmark_dmc_pixels.py` and its tests only. Its parser uses
the same explicit `--dreamer-debug-local` opt-in as Task 9b and
`--dmc-camera-id` with `default=None`. **First RED:** current defaults include
unsupported point-mass tasks and an unavailable arm. Camera tests prove omitted
quadruped input derives effective camera 2, omitted nonquadruped input derives
effective camera 0, and explicit integer 0 remains an override. Default run
construction must omit `--dmc-camera-id`; it is forwarded only when supplied.
**Implementation:** the public default is exactly
`DEFAULT_TASKS=("cartpole/swingup","finger/spin")`,
`DEFAULT_ARMS=("dreamer_v3_baseline",)`, and seeds `0..4`:
`2 tasks * 1 arm * 5 seeds = 10` Dreamer runs. These ten primary commands are
canonical paper arms and omit every debug child override. Reject unavailable arm names
and unsupported tasks before any subprocess. Primary `--summary` inputs use
Task 9b's strict normalizer, require null debug identity, and include only
`comparison_role="primary"`. The explicit comparison debug opt-in exists only
for local smoke execution; its visibly labeled rows are excluded from the ten
primary runs, aggregate tables, and parity gates. Preserve the existing explicit
`--baseline-summary` privileged-vector aggregation path, label every such row
`comparison_role="privileged_state"`, and never count it as a primary arm run.
Optional random-policy baseline rows remain separately tagged
`comparison_role="random_policy"` and are not arm runs. **Focused acceptance:**
benchmark tests assert the exact ten-run set and baseline separation, run
`test_real_dreamer_benchmark_subprocess_summary` plus the literal one-task
Dreamer benchmark, and scoped Ruff check/format. Fresh spec/quality review and
fixer must reach zero Critical/Important before
`feat(dreamer): integrate dmc pixel benchmark`.

#### Integrated Task 9 regression inventory

**Aggregate sequential ownership:**
`src/world_marl/scripts/train_dreamer_v3_baseline.py`,
`src/world_marl/scripts/compare_visual_wm.py`,
`src/world_marl/scripts/benchmark_dmc_pixels.py`, only `[project.scripts]` in
`pyproject.toml`,
`src/world_marl/dreamer_v3_baseline/__init__.py`, deletion of exactly
`losses.py`, `models.py`, `imagination.py`, `training.py`, and `validation.py`.
The independent Task 9d owns only the known mechanical format repair in
`src/world_marl/evaluation.py`. Test ownership is new `tests/test_dreamer_v3_cli.py`, new
`tests/test_dreamer_v3_runtime_exports.py`, and existing
`tests/test_compare_visual_wm.py` and `tests/test_benchmark_dmc_pixels.py`. Do
not modify algorithm implementations except through a fresh owner fixer.

**Integrated interfaces/dependencies:** production CLI with default paper/explicit current,
validated overrides, dry-run manifest, resume, local debug, artifact paths, and
resolvable comparison arms. The parser contract is exactly `--out-dir PATH`,
`--profile {paper,upstream-current}`, `--observation-mode {vision,proprio}`,
`--task TASK`, `--seed INT`, optional `--dmc-camera-id INT`, `--dry-run`,
`--dry-run-matrix PATH`, `--debug-local`, `--env-steps INT`, debug-only
`--stop-after-env-steps INT`, `--num-envs INT`,
`--batch-size INT`, `--batch-length INT`,
`--train-ratio FLOAT`, `--eval-every INT`, `--eval-episodes INT`,
`--report-every INT`, `--checkpoint-every INT`, and `--resume PATH`. A debug
flag may reduce resources/budgets but not equations or DMC contracts. Task 9
parses only the declared override fields into Task 1's `RuntimeOverrides`,
passes parsed `--seed` as the resolver's primary `seed=` input, and calls
`resolve_dreamer_run`; it never patches config dataclasses or recomputes hashes.
Out-dir/resume/dry-run/stop remain invocation-only, and camera constructs
`DMCSpec` environment identity rather than Dreamer config identity. Depends
on Tasks 1-8d. `--dry-run-matrix PATH` writes
`{"schema_version":1,"profile":"paper","runs":[...]}` with exactly one entry
in task-major/mode-minor order
`[(task, mode) for task in DMC20_ORDER for mode in
("vision", "proprio")]`, where `DMC20_ORDER` is the literal Task 6 order.
Each actual run manifest has a fresh `run_id` and initial generation. After its
atomic write, summaries and checkpoint payloads repeat that `run_id`; writer
byte offsets, next-file cursors, and checkpoint generations increase
monotonically across resume. Numbered open-loop/evaluation files have direct
schemas and no companion identity files. `--out-dir` is
required for an actual run and for single-run `--dry-run`; it is not accepted as
a hidden default and is not required in `--dry-run-matrix` mode.
Every Task 9 attempt creates one fresh `TASK9_ROOT`; the single dry-run output
and matrix are children of that root and no fixed `/tmp` path is reused.

### Exact comparison and benchmark contract

- The only launchable genuine pixel arm at Task 9 completion is
  `dreamer_v3_baseline`. The default matrix is two tasks
  (`cartpole/swingup`, `finger/spin`) x that one arm x five seeds: ten runs.
  Unknown or unavailable arm names fail validation before subprocess creation.
- Direct comparison dispatch requires explicit `--env`; summary-only mode needs
  none. `normalize_summary` is the sole strict boundary for raw Dreamer and
  primary comparison summaries, including benchmark resume and aggregation.
- Task translation is a total reverse lookup through `DMC_TASKS`; no
  underscore heuristic. Dreamer `vision` maps to comparison `pixels` without
  changing either Dreamer's uint8 adapter or the shared float32 adapter.
- `collect_steps=K` per environment becomes exactly
  `env_steps=K*num_envs` because both shipped profiles have canonical
  `action_repeat=1`; environment count, evaluation episodes, seed, and an
  explicit camera pass unchanged. This is a physical control-frame budget, not
  a replay-row budget; reset rows do not consume it. A default primary child
  argv is `uv run world-marl-train-dreamer-v3-baseline --out-dir OUT --profile
  paper --observation-mode vision --task CANONICAL --env-steps FRAMES
  --num-envs N --eval-episodes E --seed S`, plus optional camera/resume. It has
  no debug override. The sole `--dreamer-debug-local` comparison opt-in adds the
  child's `--debug-local` only for a visibly labeled local smoke run.
- Both comparison/benchmark camera parsers use `default=None`. Omission must
  omit `--dmc-camera-id` from the child argv, leaving `DMCSpec` to derive
  quadruped effective camera 2 and nonquadruped effective camera 0. An explicit
  camera 0 is forwarded and preserved, including for a quadruped task.
- Comparison normalization accepts measured `actual_env_frames` and
  `overshoot_env_frames`; it validates the requested lower bound but must not
  claim exact budget equality for `num_envs > 1`.
- Offline legacy knobs are recorded as ignored with reasons; image size must be
  64. Non-DMC environments fail. `allow_fail` controls only parent behavior.
- Benchmark resume validates the exact arm/task/seed/target/profile/modality/
  run identity and completion status. Incompatible existing state is fatal.
- A raw summary has exact keys `schema_version`, `model`, `profile`, `task`,
  `observation_mode`, `seed`, `status`, `environment_backend`,
  `config_sha256`, nullable `debug_snapshot`, exact closed `runtime_overrides`,
  `requested_target_env_steps`, nullable `requested_stop_env_steps`,
  `actual_env_frames`, `overshoot_env_frames`,
  `train_updates`, `imagined_transitions`,
  `evaluation_return`, nullable `last_loss`, `learning_gate_passed`, and
  `run_id`. Normalization maps canonical task to slash env, Vision to pixels,
  last loss/evaluation return/frame/update names to the existing row schema.
  Normalized rows retain all three identity fields. Primary normalization and
  tables reject/exclude a nonnull debug snapshot; explicit local debug rows use
  `comparison_role="debug_local"` and never satisfy paper parity gates.
- The explicit `--baseline-summary` path accepts only the existing genuine
  dm_control privileged-vector schema and labels it non-primary. Optional
  random-policy baselines are generated and aggregated separately. Neither
  baseline source enters the ten primary Dreamer arm runs.

`--env-steps` is the immutable requested target budget used by initial and
resumed invocations. `--stop-after-env-steps` is accepted only with
`--debug-local`; it must be positive, below the target, aligned to
one native frame, and above a restored frame counter. Both are lower bounds:
the synchronous runner finishes the first full-vector quantum reaching/crossing
the bound by its actual native-frame result, records actual frames and bounded
overshoot `[0,N-1]`, drains rows/earned
updates/crossed cadences, then checkpoints. It is excluded
from canonical config JSON/hash, run manifest, checkpoint identity, and resume
compatibility. At the requested safe point the runner finishes that turn's
transitions/cadence decisions, flushes direct artifacts, publishes a complete
checkpoint, and exits zero without relabeling the target complete or adding a
process-history artifact.
Task-9 CLI/comparison tests include `N=2`, `target=3`, mixed reset/control,
debug stop, cadence crossing, and resumed summary normalization.

Single-run `--dry-run` validates and resolves all CLI/config/source metadata,
then atomically writes exactly `$DRY_ROOT/manifest.json` with keys
`schema_version`, `kind="dreamer_v3_dry_run"`, `profile`, `observation_mode`,
`task`, `seed`, `camera`, `resolved_config`, `config_sha256`,
`authority_revision`, `debug_snapshot`, `runtime_overrides`, and
`canonical_argv`. Runtime manifests have no
authority-source map, implementation revision/map, live-code hash, or legacy
generic source field. Fixture manifests remain the only place that hashes the
official blobs used for numerical parity. It creates no
environment, model, replay, writer, checkpoint manager, runner, run id, or
second file. The manifest digest is recomputed from canonical
`resolved_config`; camera is not a `DreamerV3Config` field or part of that hash,
but is a `DMCSpec` value present in canonical argv, the runtime override map,
manifest environment identity, and checkpoint compatibility. The test
monkeypatches every forbidden constructor to raise
and asserts the exact one-file tree. A separate `--debug-local` subprocess
runs one real production collection step.

**Integrated RED matrix:** CLI tests assert help/defaults, immutable target versus
per-invocation stop validation, exact dry-run schema/one-file atomic output,
and that dry run constructs none of the production environment/model/replay/
writer/checkpoint/runner objects. A separate debug-local subprocess first
reproduces the current obsolete RSSM constructor failure while exercising one
production runner step;
import tests prove legacy modules remain reachable; comparison tests resolve
the public registry, reject unavailable arm names before subprocess creation,
and require real Dreamer dispatch rather than accepting mocked command strings.
`test_real_dreamer_comparison_arm_subprocess_summary` invokes
the actual comparison subprocess, not a monkeypatched parser, and initially
fails before the Dreamer argv/summary translation is wired.

**Integrated implementation constraints:** (1) specify parser, minimal runtime-identity dry-run
schema, and constructor-free atomic dry-run path; (2) wire production config/
runner/artifacts/checkpoint so the debug-local safe stop follows
`direct writer flush -> atomic checkpoint -> exit`; (3) reject canonical-field mutation unless a
named noncanonical debug mode is selected; (4) remove old imports/modules and
migrate exports; (5) make comparison arms capability-aware and executable-
resolving; (6) align benchmark real-step/profile fields; (7) refactor parser
helpers only after subprocess tests pass.

**Integrated regression acceptance after 9a-9c:**

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
TASK9_ROOT="$(mktemp -d /tmp/wm-marl-dreamer-v3-task9.XXXXXX)"
DRY_ROOT="$TASK9_ROOT/dry-run"
DEBUG_ROOT="$TASK9_ROOT/debug-one-step"
COMPARE_ROOT="$TASK9_ROOT/compare-dreamer"
BENCHMARK_ROOT="$TASK9_ROOT/benchmark-dreamer"
MATRIX="$TASK9_ROOT/dmc20x2.json"
uv run pytest -q tests/test_dreamer_v3_cli.py tests/test_dreamer_v3_runtime_exports.py tests/test_compare_visual_wm.py tests/test_benchmark_dmc_pixels.py
uv run world-marl-train-dreamer-v3-baseline --help
uv run world-marl-train-dreamer-v3-baseline --out-dir "$DRY_ROOT" --profile paper --observation-mode vision --task walker_walk --dry-run
uv run python -c 'from pathlib import Path; import json,sys; root=Path(sys.argv[1]); assert {p.relative_to(root).as_posix() for p in root.rglob("*")}=={"manifest.json"}; d=json.loads((root/"manifest.json").read_text()); assert d["profile"]=="paper" and d["task"]=="walker_walk"' "$DRY_ROOT"
uv run world-marl-train-dreamer-v3-baseline --out-dir "$DEBUG_ROOT" --profile paper --observation-mode proprio --task cartpole_balance --debug-local --env-steps 2 --stop-after-env-steps 1 --num-envs 1 --batch-size 1 --batch-length 4 --eval-every 16 --eval-episodes 1 --report-every 16 --checkpoint-every 1 --seed 0
uv run python -c 'from pathlib import Path; import sys; from world_marl.dreamer_v3_baseline.artifacts import inspect_run_state; s=inspect_run_state(Path(sys.argv[1])); assert s["writer_offsets"]["metrics"]==0 and (Path(sys.argv[1])/"checkpoints/latest").exists(); print(s["run_id"])' "$DEBUG_ROOT"
uv run world-marl-train-dreamer-v3-baseline --profile paper --dry-run-matrix "$MATRIX"
uv run python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert len(d["runs"])==40 and d["profile"]=="paper"' "$MATRIX"
uv run python -m world_marl.scripts.compare_visual_wm --arm dreamer_v3_baseline --env dmc-pixels:ball_in_cup/catch --out-dir "$COMPARE_ROOT" --collect-steps 2 --num-envs 1 --eval-episodes 1 --seed 0 --image-size 64 --dreamer-debug-local
uv run python -c 'from pathlib import Path; import json,sys; rows=json.loads((Path(sys.argv[1])/"comparison.json").read_text()); assert len(rows)==1 and rows[0]["model"]=="dreamer_v3_baseline" and rows[0]["comparison_role"]=="debug_local"' "$COMPARE_ROOT"
uv run python -m world_marl.scripts.benchmark_dmc_pixels --arm dreamer_v3_baseline --task cartpole/swingup --seed 0 --no-random-baseline --out-dir "$BENCHMARK_ROOT" --collect-steps 2 --num-envs 1 --eval-episodes 1 --image-size 64 --dreamer-debug-local
uv run python -c 'from world_marl.scripts.benchmark_dmc_pixels import DEFAULT_ARMS,DEFAULT_TASKS,DEFAULT_SEEDS; assert DEFAULT_ARMS==("dreamer_v3_baseline",) and DEFAULT_TASKS==("cartpole/swingup","finger/spin") and len(DEFAULT_ARMS)*len(DEFAULT_TASKS)*len(DEFAULT_SEEDS)==10'
uv run python -c "import world_marl.dreamer_v3_baseline"
uv run ruff check src/world_marl/scripts/train_dreamer_v3_baseline.py src/world_marl/scripts/compare_visual_wm.py src/world_marl/scripts/benchmark_dmc_pixels.py src/world_marl/dreamer_v3_baseline/__init__.py tests/test_dreamer_v3_cli.py tests/test_dreamer_v3_runtime_exports.py tests/test_compare_visual_wm.py tests/test_benchmark_dmc_pixels.py
uv run ruff format --check src/world_marl/scripts/train_dreamer_v3_baseline.py src/world_marl/scripts/compare_visual_wm.py src/world_marl/scripts/benchmark_dmc_pixels.py src/world_marl/dreamer_v3_baseline/__init__.py tests/test_dreamer_v3_cli.py tests/test_dreamer_v3_runtime_exports.py tests/test_compare_visual_wm.py tests/test_benchmark_dmc_pixels.py
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/configs.yaml >/dev/null
```

Retain the literal Task 9 root through review. Report CLI schema, direct artifact
inspection, exact comparison argv/row, the ten-run primary matrix, and separate
privileged/random baseline rows. Fresh reviews/fixers must leave every Task 9
unit at zero Critical/Important. No push or merge.

### Task 9d: Mechanical repository-format hygiene

**Owned file/boundary:** edit only `src/world_marl/evaluation.py`. This is a
separate repository-hygiene unit after all Dreamer CLI/comparison/benchmark
work; it owns no Dreamer algorithm, interface, test, fixture, dependency, or
artifact behavior. The first RED is the focused
`ruff format --check src/world_marl/evaluation.py` result naming that one file.
Apply only `ruff format` to that file, run its existing evaluation tests, and
inspect the diff to confirm whitespace/layout-only change. Fresh read-only spec
and quality review must report zero Critical/Important before the conventional
style commit `style: format evaluation module`. This removes the Task-0 known
baseline before the final repository-wide format gate.

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
uv run pytest tests/test_evaluation.py tests/test_evaluation_scan.py -q
uv run ruff check src/world_marl/evaluation.py
uv run ruff format --check src/world_marl/evaluation.py
git diff --check
uv run ruff format --check .
```

## Task 10: Run local parity and real-DMC gates

**Owned files and boundary:** rewrite
`tests/test_dreamer_v3_baseline.py`; create
`tests/test_dreamer_v3_integration.py`. Task 10 owns no tracked fixture or
production file. It may run but not edit `fixture_generator.py` or any tracked
fixture. Production fixes are dispatched one at a time to the exact owning
module with a focused RED and their own review; no broad integration-test
workaround.

**Interfaces/dependencies:** package conformance gate, both-profile fixture
regeneration, local CPU online debug, real dm_control Vision/Proprio run,
checkpoint/resume/evaluation/artifact inspection, and 20-task-by-two-mode dry-run
manifest. Every attempt creates one new `GATE_ROOT`; fixture regeneration,
matrix, Vision, Proprio, checkpoint, and resume outputs are children of that
root. Depends on Tasks 1-9.

**First RED:** the obsolete baseline test currently fails collection on removed
`categorical_straight_through`; a new production integration test must initially
fail at the first missing/incorrect online boundary rather than be skipped.

Task 10 owns real online artifact/checkpoint inspection of the DMC policy
identity across both profiles and modes. The real-DMC gate names
`wm_marl_seedsequence_v1`, inspects both complete `DMCSpec` projections in the
run manifest and decoded checkpoint, and confirms them after online resume.
Trace parity injects and records the same explicit native child seed on both
sides, while a separate source-conformance test keeps the official constructor
classified as automatic and unseeded. Component seed derivation remains
Task-6-owned and pure payload/coordinator rejection remains Task-8-owned.

The modality oracle is literal: Vision open-loop cursor/files are nonempty;
Proprio open-loop cursor is zero; Proprio open-loop directory is absent;
evaluation cursor/files are nonempty in both modalities. Negative tests inject
a fabricated Proprio video/file and require Agent, writer, resume, and
inspection rejection.

- Vision open-loop cursor/files are nonempty.
- Proprio open-loop cursor is zero.
- Proprio open-loop directory is absent.
- evaluation cursor/files are nonempty in both modalities.

**Red-green-refactor:** (1) replace obsolete test with import/composition gates;
(2) generate every Task 1-5 case into a fresh temporary tree and compare the
complete literal expected relative file set and every byte/hash with tracked
fixtures, without editing those payloads; (3) run a tiny
production CPU job through collect/train/eval/save/restore; (4) run real local
dm_control reset/step/render and online debug in Vision and Proprio; (5) inspect
nonempty metrics/scores JSONL and every nonempty numbered open-loop/evaluation
file, writer byte offset, next-file cursor, checkpoint, and run identity after
frame 32 of the immutable 48-frame target. That invocation uses stop 32 and its
interrupted summary records stop 32. Resume omits the stop; before another
quantum its rewritten summary has null stop and zero overshoot, with no stop
leaf anywhere in decoded checkpoint/driver/config/manifest state. Prove resume
to frame 48 keeps the identity, preserves the checkpointed prefixes, advances
counters/cursors without duplicates, and produces the exact same final summary
and owner state as an uninterrupted target-48 branch with the same injected
run id and deterministic inputs. Inject an append tail and temporary numbered file,
then prove direct cold reconciliation truncates/removes them and an idempotent
second resume preserves the exact checkpointed prefix. Also prove a fresh run cannot replace existing run state
and the failed collision is nonmutating; (6) produce but do not launch
the scientific matrix; (7) dispatch owner-specific
fixers for failures and rerun from RED.

**Focused acceptance:** all commands are literal; no implementer-selected
substitute is accepted:

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
GATE_ROOT="$(mktemp -d /tmp/wm-marl-dreamer-v3-task10.XXXXXX)"
FIXTURE_ROOT="$GATE_ROOT/fixtures"
MATRIX="$GATE_ROOT/dmc20x2.json"
VISION_ROOT="$GATE_ROOT/vision"
PROPRIO_ROOT="$GATE_ROOT/proprio"
uv sync --extra dev --extra dmc
uv run python -c "import importlib.metadata as m; import dm_control; print(m.version('dm-control'))"
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator generate-all --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --current-source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir "$FIXTURE_ROOT"
uv run python - "$FIXTURE_ROOT" <<'PY_FIXTURES'
from pathlib import Path
import sys

stems = (
    "paper-proprio-agent-loss", "paper-proprio-distributions",
    "paper-proprio-optimizer-five-step", "paper-proprio-replay",
    "paper-proprio-rssm", "paper-proprio-rssm-float32",
    "paper-proprio-world-loss", "paper-proprio-world-model",
    "paper-vision-networks", "paper-vision-networks-float32",
    "paper-vision-world-loss", "paper-vision-world-model",
    "upstream-current-proprio-agent-loss",
    "upstream-current-proprio-distributions",
    "upstream-current-proprio-optimizer-five-step",
    "upstream-current-proprio-replay", "upstream-current-proprio-rssm",
    "upstream-current-proprio-rssm-float32",
    "upstream-current-proprio-world-loss",
    "upstream-current-proprio-world-model",
    "upstream-current-vision-networks",
    "upstream-current-vision-networks-float32",
    "upstream-current-vision-world-loss",
    "upstream-current-vision-world-model",
)
out = Path(sys.argv[1])
tracked = Path("tests/fixtures/dreamer_v3")
expected = {Path(f"{stem}{suffix}") for stem in stems for suffix in (".npz", ".manifest.json")}
assert len(expected)==48
got = {path.relative_to(out) for path in out.rglob("*") if path.is_file()}
assert got == expected
assert all((out / path).read_bytes() == (tracked / path).read_bytes() for path in expected)
assert (tracked / "dm_control_1_0_17_state_schema.json").is_file()
print(len(expected))
PY_FIXTURES
MUJOCO_GL=off uv run python tests/dreamer_v3_dmc_state_worker.py verify --fixture tests/fixtures/dreamer_v3/dm_control_1_0_17_state_schema.json
uv run pytest -q tests/test_dreamer_v3_baseline.py tests/test_dreamer_v3_integration.py
uv run pytest -q -m real_dmc tests/test_dreamer_v3_dmc_vector.py::test_dmc20_all_modes_real_contract
uv run pytest -q tests/test_dreamer_v3_*.py tests/test_dmc_pixel_adapter.py tests/test_compare_visual_wm.py tests/test_benchmark_dmc_pixels.py
uv run world-marl-train-dreamer-v3-baseline --profile paper --dry-run-matrix "$MATRIX"
uv run python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert len(d["runs"])==40' "$MATRIX"
uv run world-marl-train-dreamer-v3-baseline --out-dir "$VISION_ROOT" --profile paper --observation-mode vision --task cartpole_balance --debug-local --env-steps 48 --stop-after-env-steps 32 --num-envs 1 --batch-size 1 --batch-length 4 --eval-every 16 --eval-episodes 1 --report-every 16 --checkpoint-every 16 --seed 0
uv run world-marl-train-dreamer-v3-baseline --out-dir "$VISION_ROOT" --profile paper --observation-mode vision --task cartpole_balance --debug-local --env-steps 48 --num-envs 1 --batch-size 1 --batch-length 4 --eval-every 16 --eval-episodes 1 --report-every 16 --checkpoint-every 16 --seed 0 --resume "$VISION_ROOT/checkpoints/latest"
uv run world-marl-train-dreamer-v3-baseline --out-dir "$PROPRIO_ROOT" --profile paper --observation-mode proprio --task cartpole_balance --debug-local --env-steps 48 --stop-after-env-steps 32 --num-envs 1 --batch-size 1 --batch-length 4 --eval-every 16 --eval-episodes 1 --report-every 16 --checkpoint-every 16 --seed 0
uv run world-marl-train-dreamer-v3-baseline --out-dir "$PROPRIO_ROOT" --profile paper --observation-mode proprio --task cartpole_balance --debug-local --env-steps 48 --num-envs 1 --batch-size 1 --batch-length 4 --eval-every 16 --eval-episodes 1 --report-every 16 --checkpoint-every 16 --seed 0 --resume "$PROPRIO_ROOT/checkpoints/latest"
uv run python -c 'from pathlib import Path; import sys; from world_marl.dreamer_v3_baseline.artifacts import inspect_run_state; vr,pr=map(Path,sys.argv[1:]); v,p=inspect_run_state(vr),inspect_run_state(pr); assert all(s["writer_offsets"]["metrics"]>0 and s["writer_offsets"]["scores"]>0 and s["next_file_cursors"]["evaluation"]>0 and s["counters"]["env_frames"]>=48 for s in (v,p)); assert v["next_file_cursors"]["open_loop"]>0 and any((vr/"open_loop").glob("*.npz")); assert p["next_file_cursors"]["open_loop"]==0 and not (pr/"open_loop").exists(); assert any((vr/"evaluation").glob("*.json")) and any((pr/"evaluation").glob("*.json")); print([(s["run_id"],s["next_file_cursors"]) for s in (v,p)])' "$VISION_ROOT" "$PROPRIO_ROOT"
uv run pytest -q tests/test_dreamer_v3_resume_trace.py tests/test_dreamer_v3_checkpoint.py tests/test_dreamer_v3_artifact_files.py
uv run ruff check .
uv run ruff format --check .
git diff --check
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/agent.py >/dev/null
```

The report records the exact `GATE_ROOT`, config identities, run ids, writer
offsets, next-file cursors, checkpoint generations, and 48-file fixture hash. The root is retained through
review so results are inspectable. After acceptance, cleanup may move only that
literal root with `trash -- "$GATE_ROOT"`; a wildcard or reuse on a later
attempt is forbidden.

**Report/review/commit:** report required fields plus real/synthetic distinction,
direct artifact contents/cursors, resume equality, both-profile fixture digests, and
the unlaunched scientific manifest. Spec review checks all local completion
gates; quality review checks tests cannot pass on legacy/mocked paths.
Fix/re-review to zero Critical/Important before
`test(dreamer): verify local online parity`.

Linux EGL/CUDA throughput and full 20-task x 2-mode x multi-seed scientific
runs are deferred pending separate authorization. Their commands/manifests may
be generated locally, but Task 10 must not launch them and must not treat their
absence as local implementation success or failure.

## Task 11: Verify and independently review the whole branch

**Owned files and boundary:** verification-only: no required tracked change and
no empty commit. It owns only the Task 11 process paths defined by the controller
protocol and `.superpowers/sdd/progress.md`. Any finding is assigned to a fresh
fixer with an explicit smallest production/test path list and its own
test/review/commit; no concurrent overlapping editor.

**Interfaces/dependencies:** complete branch diff from the recorded architecture
base, final verification matrix, exact branch/commit range, remaining external
gates, and integration options. Each verification attempt creates one fresh
`GATE_ROOT` containing regenerated fixtures, matrix, Vision, Proprio,
checkpoints, and resume outputs. Depends on clean acceptance of Tasks 0-10.

**First RED:** run the full verification matrix before declaring completion;
any collection error, failure, unexpected skip, lint/format diff, dirty status,
legacy import, unresolved CLI arm, or reviewer finding is RED and blocks the
completion claim.

Final artifact inspection is modality-specific: Vision open-loop cursor/files
are nonempty; Proprio open-loop cursor is zero; Proprio open-loop directory is
absent; evaluation cursor/files are nonempty in both modalities. A fabricated
Proprio open-loop artifact is an explicit negative gate.

- Vision open-loop cursor/files are nonempty.
- Proprio open-loop cursor is zero.
- Proprio open-loop directory is absent.
- evaluation cursor/files are nonempty in both modalities.

**Red-green-refactor:** (1) generate a whole-range review package; (2) run fresh
independent spec and quality reviews against the original request,
`ARCHITECTURE.md`, and all reports; (3) dispatch fresh owner fixers for every
Critical/Important and add covering RED tests; (4) re-review each fixer-owned scope and
whole range until both reviews report zero Critical/Important; (5) invoke
verification-before-completion and rerun all gates from a clean process; (6)
record exact results/skips/external gates and invoke finishing-a-development-
branch without merging or pushing.

**Focused acceptance:** run literally from a fresh process after both whole-range
reviews are clean:

```bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wm-marl-uv-cache}"
GATE_ROOT="$(mktemp -d /tmp/wm-marl-dreamer-v3-task11.XXXXXX)"
FIXTURE_ROOT="$GATE_ROOT/fixtures"
MATRIX="$GATE_ROOT/dmc20x2.json"
VISION_ROOT="$GATE_ROOT/vision"
PROPRIO_ROOT="$GATE_ROOT/proprio"
test -n "${OFFICIAL_SAFETY_SNAPSHOT:?controller snapshot required}"
test -n "${JEPA_SAFETY_SNAPSHOT:?controller snapshot required}"
uv sync --extra dev --extra dmc
uv run python -c "import importlib.metadata as m; import dm_control; print(m.version('dm-control'))"
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator generate-all --reference-checkout /private/tmp/danijar-dreamerv3-20260713 --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 --current-source-revision e3f02248693a79dc8b0ebd62c93683888ddaccfe --output-dir "$FIXTURE_ROOT"
uv run python - "$FIXTURE_ROOT" <<'PY_FIXTURES'
from pathlib import Path
import sys

stems = (
    "paper-proprio-agent-loss", "paper-proprio-distributions",
    "paper-proprio-optimizer-five-step", "paper-proprio-replay",
    "paper-proprio-rssm", "paper-proprio-rssm-float32",
    "paper-proprio-world-loss", "paper-proprio-world-model",
    "paper-vision-networks", "paper-vision-networks-float32",
    "paper-vision-world-loss", "paper-vision-world-model",
    "upstream-current-proprio-agent-loss",
    "upstream-current-proprio-distributions",
    "upstream-current-proprio-optimizer-five-step",
    "upstream-current-proprio-replay", "upstream-current-proprio-rssm",
    "upstream-current-proprio-rssm-float32",
    "upstream-current-proprio-world-loss",
    "upstream-current-proprio-world-model",
    "upstream-current-vision-networks",
    "upstream-current-vision-networks-float32",
    "upstream-current-vision-world-loss",
    "upstream-current-vision-world-model",
)
out = Path(sys.argv[1])
tracked = Path("tests/fixtures/dreamer_v3")
expected = {Path(f"{stem}{suffix}") for stem in stems for suffix in (".npz", ".manifest.json")}
assert len(expected)==48
got = {path.relative_to(out) for path in out.rglob("*") if path.is_file()}
assert got == expected
assert all((out / path).read_bytes() == (tracked / path).read_bytes() for path in expected)
assert (tracked / "dm_control_1_0_17_state_schema.json").is_file()
print(len(expected))
PY_FIXTURES
MUJOCO_GL=off uv run python tests/dreamer_v3_dmc_state_worker.py verify --fixture tests/fixtures/dreamer_v3/dm_control_1_0_17_state_schema.json
uv run pytest -q tests/test_dreamer_v3_*.py tests/test_dmc_pixel_adapter.py tests/test_compare_visual_wm.py tests/test_benchmark_dmc_pixels.py
uv run pytest -q -m real_dmc tests/test_dreamer_v3_dmc_vector.py::test_dmc20_all_modes_real_contract
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python -c "import world_marl.dreamer_v3_baseline"
uv run world-marl-train-dreamer-v3-baseline --help
uv run world-marl-train-dreamer-v3-baseline --profile paper --dry-run-matrix "$MATRIX"
uv run world-marl-train-dreamer-v3-baseline --out-dir "$VISION_ROOT" --profile paper --observation-mode vision --task cartpole_balance --debug-local --env-steps 48 --stop-after-env-steps 32 --num-envs 1 --batch-size 1 --batch-length 4 --eval-every 16 --eval-episodes 1 --report-every 16 --checkpoint-every 16 --seed 0
uv run world-marl-train-dreamer-v3-baseline --out-dir "$VISION_ROOT" --profile paper --observation-mode vision --task cartpole_balance --debug-local --env-steps 48 --num-envs 1 --batch-size 1 --batch-length 4 --eval-every 16 --eval-episodes 1 --report-every 16 --checkpoint-every 16 --seed 0 --resume "$VISION_ROOT/checkpoints/latest"
uv run world-marl-train-dreamer-v3-baseline --out-dir "$PROPRIO_ROOT" --profile paper --observation-mode proprio --task cartpole_balance --debug-local --env-steps 48 --stop-after-env-steps 32 --num-envs 1 --batch-size 1 --batch-length 4 --eval-every 16 --eval-episodes 1 --report-every 16 --checkpoint-every 16 --seed 0
uv run world-marl-train-dreamer-v3-baseline --out-dir "$PROPRIO_ROOT" --profile paper --observation-mode proprio --task cartpole_balance --debug-local --env-steps 48 --num-envs 1 --batch-size 1 --batch-length 4 --eval-every 16 --eval-episodes 1 --report-every 16 --checkpoint-every 16 --seed 0 --resume "$PROPRIO_ROOT/checkpoints/latest"
uv run python -c 'from pathlib import Path; import sys; from world_marl.dreamer_v3_baseline.artifacts import inspect_run_state; vr,pr=map(Path,sys.argv[1:]); v,p=inspect_run_state(vr),inspect_run_state(pr); assert all(s["counters"]["env_frames"]>=48 and s["next_file_cursors"]["evaluation"]>0 for s in (v,p)); assert v["next_file_cursors"]["open_loop"]>0 and any((vr/"open_loop").glob("*.npz")); assert p["next_file_cursors"]["open_loop"]==0 and not (pr/"open_loop").exists(); assert any((vr/"evaluation").glob("*.json")) and any((pr/"evaluation").glob("*.json"))' "$VISION_ROOT" "$PROPRIO_ROOT"
uv run python -c 'import json,os,subprocess,sys; from pathlib import Path
def snap(path):
 p=Path(path); untracked=subprocess.check_output(["git","-C",str(p),"ls-files","--others","--exclude-standard","-z"]).split(b"\0"); untracked=[x for x in untracked if x]; inv=[]
 for raw in untracked:
  q=p/raw.decode(); inv.append({"path_hex":raw.hex(),"kind":"link" if q.is_symlink() else "file" if q.is_file() else "other","bytes_hex":os.readlink(q).encode().hex() if q.is_symlink() else q.read_bytes().hex() if q.is_file() else ""})
 return {"path":str(p),"head":subprocess.check_output(["git","-C",str(p),"rev-parse","HEAD"],text=True).strip(),"branch":subprocess.check_output(["git","-C",str(p),"branch","--show-current"],text=True).strip(),"status_hex":subprocess.check_output(["git","-C",str(p),"status","--porcelain=v1","-z","--untracked-files=all"]).hex(),"tracked_diff_hex":subprocess.check_output(["git","-C",str(p),"diff","HEAD","--binary"]).hex(),"untracked":inv}
for f in sys.argv[1:]:
 expected=json.loads(Path(f).read_text()); actual=snap(expected["path"]); assert actual==expected,(actual,expected)
print("GREEN accepted live reference snapshots")' "$OFFICIAL_SAFETY_SNAPSHOT" "$JEPA_SAFETY_SNAPSHOT"
git -C /private/tmp/danijar-dreamerv3-20260713 show bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01:dreamerv3/agent.py >/dev/null
git diff --check
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

The final report records the literal `GATE_ROOT`, 48-file fixture hash,
manifest/config hashes, run ids, and checkpoint generations. It retains that
root for user inspection; if cleanup is later requested, only
`trash -- "$GATE_ROOT"` is permitted. A later verification attempt must create
a different fresh root and may not accept artifacts from this one.

**Report/review/final HEAD:** the final report contains every required field, exact
test counts and unexpected/expected skips, branch and commit range, clean status,
two independent zero-Critical/zero-Important verdicts, and the honest list of
unauthorized Linux GPU/scientific gates. Fix/re-review until both reviewers are
clean, then record the existing final HEAD; Task 11 creates no verification-only
commit. Do not push, merge, or select an integration option for the user.
