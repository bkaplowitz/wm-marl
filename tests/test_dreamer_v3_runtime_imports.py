from __future__ import annotations

import ast
import json
import subprocess
import symtable
import sys
from pathlib import Path
from typing import TypedDict, cast


PACKAGE_NAME = "world_marl.dreamer_v3_baseline"
PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1] / "src" / "world_marl" / "dreamer_v3_baseline"
)
TOOLING_TOKENS = ("oracle", "fixture", "generator", "contract")
REQUIRED_TOOLING_MODULES = {
    f"{PACKAGE_NAME}.fixture_generator",
    f"{PACKAGE_NAME}.network_oracle",
    f"{PACKAGE_NAME}.oracle",
    f"{PACKAGE_NAME}.replay_oracle",
    f"{PACKAGE_NAME}.replay_oracle_contract",
    f"{PACKAGE_NAME}.rssm_oracle",
}
TOOLING_EXPORTS = {
    "DISTRIBUTIONS_SOURCE_SPEC",
    "NETWORKS_SOURCE_SPEC",
    "OracleManifest",
    "OracleSourceSpec",
    "ParameterMapping",
    "ParameterTranslator",
    "REPLAY_SOURCE_SPEC",
    "RSSM_SOURCE_SPEC",
    "TensorSpec",
}
RUNTIME_EXPORTS = {
    "AggregateOutput",
    "BlockGRU",
    "DreamerProfile",
    "DreamerReplay",
    "ResolvedDreamerRun",
    "RSSM",
    "RuntimeOverrides",
    "SequenceShapeConfig",
    "resolve_dreamer_run",
}
TRANSITIONAL_ORACLE_SYMBOLS = {
    "OracleSourceSpec",
    "_ORACLE_SOURCE_SPECS",
    "register_oracle_source_spec",
    "source_spec_for",
}
RETAINED_FIXTURE_SOURCE_SYMBOLS = {
    "DISTRIBUTIONS_SOURCE_SPEC",
    "NETWORKS_SOURCE_SPEC",
    "REPLAY_SOURCE_SPEC",
    "RSSM_SOURCE_SPEC",
    "_SOURCE_DTYPES",
    "_SOURCE_HASHES",
}


class _RuntimeImportResult(TypedDict):
    all: list[str]
    attributes: dict[str, bool]
    loaded_tooling: list[str]
    tooling: list[str]


def _fresh_runtime_import() -> _RuntimeImportResult:
    script = f"""
import json
import pkgutil
import sys
import {PACKAGE_NAME} as package

names = {{module.name for module in pkgutil.walk_packages(
    package.__path__, package.__name__ + "."
)}}
tooling = {{
    name for name in names
    if any(token in name.rsplit(".", 1)[-1] for token in {TOOLING_TOKENS!r})
}}
print(json.dumps({{
    "all": sorted(package.__all__),
    "attributes": {{
        name: hasattr(package, name)
        for name in {sorted(TOOLING_EXPORTS)!r}
    }},
    "loaded_tooling": sorted(tooling & set(sys.modules)),
    "tooling": sorted(tooling),
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(_RuntimeImportResult, json.loads(completed.stdout))


def _assigned_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            names.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
    return names


def _module_getattr_violation(source: str) -> tuple[int, str] | None:
    root = symtable.symtable(source, "<runtime-import-scan>", "exec")
    pending = [root]

    while pending:
        table = pending.pop()
        pending.extend(table.get_children())
        if "__getattr__" not in table.get_identifiers():
            continue

        symbol = table.lookup("__getattr__")
        binds_name = symbol.is_assigned() or symbol.is_imported()
        if table is root and (binds_name or symbol.is_namespace()):
            return (0, "module __getattr__ binding")
        if table is not root and symbol.is_global() and binds_name:
            return (0, "global __getattr__ binding")

    return None


def _tooling_import_violations(
    source: str, tooling_leaves: set[str]
) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in {"builtins", "importlib"}:
                    violations.append((node.lineno, alias.name))
                if (
                    alias.name.startswith(f"{PACKAGE_NAME}.")
                    and alias.name.rsplit(".", 1)[-1] in tooling_leaves
                ):
                    violations.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".", 1)[0] == "importlib":
                violations.append((node.lineno, module))
            if module == "builtins":
                violations.append((node.lineno, module))
            local_module = node.level > 0 or module.startswith(PACKAGE_NAME)
            if local_module and module.rsplit(".", 1)[-1] in tooling_leaves:
                violations.append((node.lineno, module))
            if node.level > 0 or module == PACKAGE_NAME:
                for alias in node.names:
                    if alias.name in tooling_leaves:
                        violations.append((node.lineno, alias.name))

        if isinstance(node, ast.Name) and node.id in {
            "__builtins__",
            "__import__",
            "importlib",
            "LazyLoader",
        }:
            violations.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and node.attr in {
            "__import__",
            "import_module",
            "LazyLoader",
        }:
            violations.append((node.lineno, node.attr))

    if violation := _module_getattr_violation(source):
        violations.append(violation)
    return violations


def test_scanner_rejects_static_dynamic_and_lazy_tooling_routes() -> None:
    tooling_leaves = {"oracle", "rssm_oracle"}
    outer_scope_mutants = {
        "module comprehension walrus": (
            "[(__getattr__ := lambda name: name) for _ in (0,)]"
        ),
        "function decorator walrus": """
@(__getattr__ := lambda obj: obj)
def f():
    pass
""",
        "function default walrus": """
def f(value=(__getattr__ := lambda name: name)):
    pass
""",
        "lambda default walrus": (
            "f = lambda value=(__getattr__ := lambda name: name): value"
        ),
        "class-base walrus": """
class C((__getattr__ := object)):
    pass
""",
    }
    mutants = {
        "direct absolute aliased": (
            "import world_marl.dreamer_v3_baseline.oracle as tooling"
        ),
        "relative import": "from .oracle import OracleManifest",
        "package import": "from world_marl.dreamer_v3_baseline import oracle",
        "multiline direct import": """
from world_marl.dreamer_v3_baseline import (
    oracle,
)
""",
        "aliased importlib dynamic target": """
import importlib as il

def load(tooling):
    return il.import_module(tooling)
""",
        "aliased package-relative import_module": """
from importlib import import_module as load

def load_tooling():
    return load(".oracle", __package__)
""",
        "assigned import_module alias": """
import importlib
load = importlib.import_module

def load_tooling():
    return load("world_marl.dreamer_v3_baseline.oracle")
""",
        "built-in import": (
            "def load(): return __import__("
            '"world_marl.dreamer_v3_baseline.rssm_oracle")'
        ),
        "five-argument package-relative built-in import": """
def load():
    return __import__("oracle", globals(), locals(), (), 1)
""",
        "package built-in import with tooling fromlist": """
def load():
    return __import__(
        "world_marl.dreamer_v3_baseline",
        fromlist=("oracle",),
    )
""",
        "aliased builtins import": """
import builtins as bi

def load(tooling):
    return bi.__import__(tooling)
""",
        "assigned builtins import alias": """
import builtins
load = builtins.__import__

def load_tooling():
    return load("world_marl.dreamer_v3_baseline.oracle")
""",
        "recovered builtins import": """
import builtins
load = getattr(builtins, "__import__")

def load_tooling():
    return load("world_marl.dreamer_v3_baseline.oracle")
""",
        "recovered aliased builtins import": """
import builtins as bi
load = getattr(bi, "__import__")

def load_tooling():
    return load("world_marl.dreamer_v3_baseline.oracle")
""",
        "recovered imported getattr builtins import": """
from builtins import getattr as recover
from sys import modules
load = recover(modules["builtins"], "__import__")

def load_tooling():
    return load("world_marl.dreamer_v3_baseline.oracle")
""",
        "imported built-in alias": """
from builtins import __import__ as load

def load_tooling(tooling):
    return load(tooling)
""",
        "aliased lazy loader": """
from importlib.util import LazyLoader as LL

def load(spec):
    return LL(spec.loader)
""",
        "assigned lazy loader alias": """
import importlib.util
Loader = importlib.util.LazyLoader

def load(spec):
    return Loader(spec.loader)
""",
        "lazy loader factory": """
from importlib.util import LazyLoader

factory = LazyLoader.factory
""",
        "module getattr": "def __getattr__(name): return name",
        "module getattr assignment": "__getattr__ = lambda name: name",
        "conditional module getattr definition": """
if True:
    def __getattr__(name):
        return name
""",
        "conditional module getattr assignment": """
if True:
    __getattr__ = lambda name: name
""",
        "destructured module getattr assignment": """
def load(name):
    return name

__getattr__, sentinel = load, None
""",
        **outer_scope_mutants,
    }

    for source in outer_scope_mutants.values():
        namespace: dict[str, object] = {}
        exec(compile(source, "<module-scope-mutant>", "exec"), namespace)
        assert "__getattr__" in namespace

    missed = [
        name
        for name, source in mutants.items()
        if not _tooling_import_violations(source, tooling_leaves)
    ]

    assert missed == []


def test_scanner_ignores_function_and_class_local_getattr_bindings() -> None:
    source = """
def outer():
    def __getattr__(name):
        return name
    __getattr__ = lambda name: name
    return __getattr__

class Namespace:
    def __getattr__(self, name):
        return name
    __getattr__ = lambda self, name: name
    values = [None for __getattr__ in ()]

module_values = [None for __getattr__ in ()]
local_lambda = lambda: (__getattr__ := (lambda name: name))

def nested_function_comprehension():
    return [(__getattr__ := lambda name: name) for _ in (0,)]

class NestedClassComprehension:
    values = [None for __getattr__ in ()]
"""

    assert _tooling_import_violations(source, {"oracle"}) == []


def test_runtime_package_import_does_not_load_discovered_tooling_modules() -> None:
    result = _fresh_runtime_import()

    tooling = set(result["tooling"])
    assert REQUIRED_TOOLING_MODULES <= tooling
    assert result["loaded_tooling"] == []


def test_runtime_package_retains_runtime_exports_without_tooling_aliases() -> None:
    result = _fresh_runtime_import()

    exports = set(result["all"])
    assert RUNTIME_EXPORTS <= exports
    assert not TOOLING_EXPORTS & exports
    assert not any(result["attributes"].values())

    package_tree = ast.parse((PACKAGE_ROOT / "__init__.py").read_text())
    assert "__getattr__" not in _assigned_names(package_tree)


def test_production_modules_do_not_import_discovered_tooling_modules() -> None:
    tooling_leaves = {
        name.rsplit(".", 1)[-1] for name in _fresh_runtime_import()["tooling"]
    }
    violations: list[tuple[str, int, str]] = []

    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        if path.stem in tooling_leaves:
            continue
        source = path.read_text()
        violations.extend(
            (path.name, lineno, imported)
            for lineno, imported in _tooling_import_violations(source, tooling_leaves)
        )

    assert violations == []


def test_transitional_registry_is_deleted_but_fixture_source_tables_remain() -> None:
    oracle_tree = ast.parse((PACKAGE_ROOT / "oracle.py").read_text())
    names = _assigned_names(oracle_tree)

    assert not TRANSITIONAL_ORACLE_SYMBOLS & names
    assert RETAINED_FIXTURE_SOURCE_SYMBOLS <= names
