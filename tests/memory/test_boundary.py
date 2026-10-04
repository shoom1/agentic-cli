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
COMPAT_DIRS: set[Path] = set()

# Host imports still to be removed: (file relative to the repo, imported module).
KNOWN_VIOLATIONS: set[tuple[str, str]] = set()


def _deprecated(module: str, names: list[str]) -> bool:
    """An import of a path the compatibility layer keeps until 0.7.0."""
    return False


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
                found.append((node.module, names))
            elif package is not None:
                base = package.split(".")
                if node.level > 1:
                    base = base[: -(node.level - 1)]
                found.append((".".join(base + ([node.module] if node.module else [])), names))
    return found


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
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


def package_violations() -> set[tuple[str, str]]:
    bad = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        allowed = ALLOWED_INSIDE[_component(path)]
        bad |= _violations(path, allowed, LIBRARIES, _module_name(path))
    return bad


def outside_violations() -> set[tuple[str, str]]:
    files = [
        p
        for p in sorted((SRC / "agentic_cli").rglob("*.py"))
        if PACKAGE not in p.parents and not any(d in p.parents for d in COMPAT_DIRS)
    ]
    files += sorted((ROOT / "examples").rglob("*.py"))
    bad = set()
    for path in files:
        module = _module_name(path) if SRC in path.parents else None
        for target, names in imports_of(path, module):
            inside = target == "agentic_cli.memory" or target.startswith("agentic_cli.memory.")
            if (inside and target not in PUBLIC_MODULES) or _deprecated(target, names):
                bad.add((path.relative_to(ROOT).as_posix(), target))
    return bad


def test_the_package_depends_only_on_itself_and_its_libraries():
    assert package_violations() - KNOWN_VIOLATIONS == set()


def test_every_known_violation_still_exists():
    """Delete an entry once its import is gone, so the list only shrinks."""
    assert KNOWN_VIOLATIONS - package_violations() == set()


def test_the_rest_of_agentic_cli_uses_the_public_api():
    assert outside_violations() == set()


def test_the_package_tests_depend_only_on_the_package():
    bad = set()
    for path in sorted(TESTS.rglob("*.py")):
        bad |= _violations(path, {"root", "_core", "store", "kb", "tests"}, TEST_LIBRARIES, None)
    assert bad == set()
