"""agentic_cli.memory depends on nothing else in agentic-cli.

1. The package imports only the standard library, its own third-party
   libraries and ``agentic_cli.memory`` itself.
2. Its components stay apart: ``kb`` never imports ``store``, ``store`` never
   imports ``kb``, and ``_core`` imports neither.
3. The rest of ``src/`` and ``examples/`` use only the public modules
   (``agentic_cli.memory`` and ``agentic_cli.memory.kb``) and never the
   deprecated paths the compatibility layer keeps until 0.7.0.
4. ``tests/memory/`` follows rule 1, so the tests can move with the package.

Imports inside functions and ``if TYPE_CHECKING:`` blocks count.

What this scan cannot see: imports built from strings
(``importlib.import_module``, ``__import__``), and attribute access on a
parent module after importing it (e.g. ``memory_tools.MemoryStore(...)``
after ``from agentic_cli.tools import memory_tools`` — no name from the
memory package is ever imported there). The ``DeprecationWarning`` the old
paths raise is the backstop for that case. Fixtures pulled in from conftest
are not imports either, and are invisible here too. Importing
``agentic_cli.memory`` still runs ``agentic_cli/__init__.py``, which loads
other agentic_cli modules; the boundary covers only the package's own
imports, not what importing it transitively pulls in through the top-level
package.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PACKAGE = SRC / "agentic_cli" / "memory"
TESTS = ROOT / "tests" / "memory"

LIBRARIES = {"numpy", "structlog", "torch", "sentence_transformers", "faiss", "bm25s", "rank_bm25", "nltk"}
TEST_LIBRARIES = LIBRARIES | {"pytest"}
PUBLIC_MODULES = {"agentic_cli.memory", "agentic_cli.memory.kb"}

# What each component may import from inside the package.
ALLOWED_INSIDE = {
    "root": {"root", "_core", "store", "kb"},
    "_core": {"_core"},
    "store": {"_core", "store"},
    "kb": {"_core", "kb"},
}

# Old-path modules that forward to the package until 0.7.0 (rule 3 exempt).
COMPAT_DIRS: set[Path] = {SRC / "agentic_cli" / "knowledge_base"}


def _deprecated(module: str, names: list[str]) -> bool:
    """An import of a path the compatibility layer keeps until 0.7.0."""
    if module == "agentic_cli.tools.memory_tools" and "MemoryStore" in names:
        return True
    return module == "agentic_cli.knowledge_base" or module.startswith("agentic_cli.knowledge_base.")


def _is_real_module(dotted: str) -> bool:
    """Whether ``dotted`` names an actual module or package file on disk.

    Only resolves names in the trees this scan covers: ``agentic_cli.*``
    under ``src/``, and ``tests.*`` / ``examples.*`` under the repo root. A
    name outside those trees (a third-party library) is never a filesystem
    check we can make, so it is not "real" here.
    """
    parts = dotted.split(".")
    if parts[0] == "agentic_cli":
        base = SRC
    elif parts[0] in ("tests", "examples"):
        base = ROOT
    else:
        return False
    candidate = base.joinpath(*parts)
    return candidate.with_suffix(".py").is_file() or (candidate / "__init__.py").is_file()


def _resolve_from_import(module: str, names: list[str]) -> list[tuple[str, list[str]]]:
    """Split a ``from module import names`` into the targets it really names.

    A name that is itself a submodule or subpackage (``_core`` in ``from
    agentic_cli.memory import _core``) names that dotted module, not
    ``module`` — otherwise a submodule import would be judged only by
    ``module``, which can be public even when the submodule it names is not
    (rule 3), or can resolve to the wrong component (rule 2). A name that is
    a plain attribute (a class, a function) stays grouped under ``module``.
    """
    resolved: list[tuple[str, list[str]]] = []
    plain: list[str] = []
    for name in names:
        dotted = f"{module}.{name}"
        if _is_real_module(dotted):
            resolved.append((dotted, []))
        else:
            plain.append(name)
    if plain:
        resolved.append((module, plain))
    return resolved


def imports_of(path: Path, module: str | None) -> list[tuple[str, list[str]]]:
    """Every ``(module, names)`` the file imports; relative imports resolved."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = None
    if module is not None:
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, []) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            if node.level == 0:
                resolved_module = node.module
            elif package is not None:
                base = package.split(".")
                if node.level > 1:
                    base = base[: -(node.level - 1)]
                resolved_module = ".".join(base + ([node.module] if node.module else []))
            else:
                resolved_module = None
            if resolved_module is not None:
                found += _resolve_from_import(resolved_module, names)
    return found


def _module_name(path: Path, root: Path) -> str:
    """Dotted module name for ``path``, relative to ``root``."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _kind(module: str) -> str:
    if module == "agentic_cli.memory":
        return "root"
    for part in ("_core", "store", "kb"):
        name = f"agentic_cli.memory.{part}"
        if module == name or module.startswith(name + "."):
            return part
    if module == "tests.memory" or module.startswith("tests.memory."):
        return "tests"
    top = module.partition(".")[0]
    if top in ("agentic_cli", "tests"):
        return "host"
    if top in sys.stdlib_module_names:
        return "stdlib"
    return "library"


def _component(path: Path) -> str:
    first = path.relative_to(PACKAGE).parts[0]
    components = {"__init__.py": "root", "_core": "_core", "store.py": "store", "kb": "kb"}
    if first not in components:
        raise AssertionError(f"{path} belongs to no component; classify it in _component")
    return components[first]


def _violations(path: Path, allowed: set[str], libraries: set[str], module: str | None):
    bad = set()
    for target, _names in imports_of(path, module):
        kind = _kind(target)
        if kind == "stdlib" or kind in allowed:
            continue
        if kind == "library" and target.partition(".")[0] in libraries:
            continue
        bad.add((path.relative_to(ROOT).as_posix(), target))
    return bad


def _package_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _outside_files() -> list[Path]:
    files = [
        p
        for p in sorted((SRC / "agentic_cli").rglob("*.py"))
        if PACKAGE not in p.parents and not any(d in p.parents for d in COMPAT_DIRS)
    ]
    files += sorted((ROOT / "examples").rglob("*.py"))
    return files


def _test_files() -> list[Path]:
    return sorted(TESTS.rglob("*.py"))


def package_violations() -> set[tuple[str, str]]:
    bad = set()
    for path in _package_files():
        allowed = ALLOWED_INSIDE[_component(path)]
        bad |= _violations(path, allowed, LIBRARIES, _module_name(path, SRC))
    return bad


def outside_violations() -> set[tuple[str, str]]:
    bad = set()
    for path in _outside_files():
        module = _module_name(path, SRC) if SRC in path.parents else _module_name(path, ROOT)
        for target, names in imports_of(path, module):
            inside = target == "agentic_cli.memory" or target.startswith("agentic_cli.memory.")
            if (inside and target not in PUBLIC_MODULES) or _deprecated(target, names):
                bad.add((path.relative_to(ROOT).as_posix(), target))
    return bad


def test_the_package_depends_only_on_itself_and_its_libraries():
    assert package_violations() == set()


def test_the_rest_of_agentic_cli_uses_the_public_api():
    assert outside_violations() == set()


def test_the_package_tests_depend_only_on_the_package():
    bad = set()
    for path in _test_files():
        module = _module_name(path, ROOT)
        bad |= _violations(path, {"root", "_core", "store", "kb", "tests"}, TEST_LIBRARIES, module)
    assert bad == set()


def test_the_scan_roots_are_correct_and_the_scans_are_not_empty():
    """Every rule above can pass vacuously: a wrong ``ROOT``/``PACKAGE`` or a
    scan that silently covers no files (``rglob`` on a missing directory
    yields nothing) still reports zero violations. Pin the roots and require
    each scan to see at least the files we know are there.
    """
    assert (ROOT / "pyproject.toml").is_file()
    assert (PACKAGE / "__init__.py").is_file()
    assert _package_files() != []
    assert _test_files() != []

    outside_files = _outside_files()
    assert outside_files != []
    demo = ROOT / "examples" / "memory_demo.py"
    assert demo in outside_files
    demo_module = _module_name(demo, ROOT)
    assert any(target == "agentic_cli.memory" for target, _names in imports_of(demo, demo_module))


def test_imports_of_resolves_a_submodule_name_to_its_dotted_path(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("from agentic_cli.memory import _core\n")
    assert imports_of(path, None) == [("agentic_cli.memory._core", [])]


def test_imports_of_leaves_a_plain_name_grouped_under_its_module(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("from agentic_cli.memory import MemoryStore\n")
    assert imports_of(path, None) == [("agentic_cli.memory", ["MemoryStore"])]


def test_imports_of_resolves_a_relative_submodule_import(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("from . import _core\n")
    assert imports_of(path, "agentic_cli.memory.store") == [("agentic_cli.memory._core", [])]
