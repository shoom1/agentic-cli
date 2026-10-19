"""agentic_cli.models depends on nothing else in agentic-cli, and on no provider SDK.

1. The package imports only the standard library, pydantic, httpx and
   ``agentic_cli.models`` itself: nothing from ``agentic_cli.workflow``,
   ``cli``, ``tools`` or ``memory``, and no provider SDK (``google``,
   ``anthropic``, ``openai``, ``langchain*``). Only the backend adapters touch
   provider SDKs.
2. ``tests/models/`` follows rule 1 (plus pytest), so the tests can move with
   the package.

Imports inside functions and ``if TYPE_CHECKING:`` blocks count. Imports built
from strings (``importlib.import_module``) are invisible to this scan; what
importing the package loads at runtime is checked by ``test_loads_alone.py``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PACKAGE = SRC / "agentic_cli" / "models"
TESTS = ROOT / "tests" / "models"

LIBRARIES = {"pydantic", "httpx"}
TEST_LIBRARIES = LIBRARIES | {"pytest"}


def imports_of(path: Path, module: str | None) -> list[str]:
    """Every module the file imports, relative imports resolved.

    ``from package import name`` yields ``package.name`` when that names a
    module on disk under ``src/`` or the repository root, else ``package``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = None
    if module is not None:
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module
            elif package is not None:
                parts = package.split(".")
                if node.level > 1:
                    parts = parts[: -(node.level - 1)]
                base = ".".join(parts + ([node.module] if node.module else []))
            else:
                base = None
            if base is None:
                continue
            for alias in node.names:
                dotted = f"{base}.{alias.name}"
                found.append(dotted if _is_real_module(dotted) else base)
    return found


def _is_real_module(dotted: str) -> bool:
    parts = dotted.split(".")
    if parts[0] == "agentic_cli":
        base = SRC
    elif parts[0] == "tests":
        base = ROOT
    else:
        return False
    candidate = base.joinpath(*parts)
    return candidate.with_suffix(".py").is_file() or (candidate / "__init__.py").is_file()


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _allowed(target: str, libraries: set[str], *, tests: bool) -> bool:
    top = target.partition(".")[0]
    if target == "agentic_cli.models" or target.startswith("agentic_cli.models."):
        return True
    if tests and (target == "tests.models" or target.startswith("tests.models.")):
        return True
    if top in ("agentic_cli", "tests"):
        return False
    return top in sys.stdlib_module_names or top in libraries


def _violations(files: list[Path], root: Path, libraries: set[str], *, tests: bool) -> set[tuple[str, str]]:
    bad = set()
    for path in files:
        for target in imports_of(path, _module_name(path, root)):
            if not _allowed(target, libraries, tests=tests):
                bad.add((path.relative_to(root).as_posix(), target))
    return bad


def _package_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _test_files() -> list[Path]:
    return sorted(TESTS.rglob("*.py"))


def test_the_package_depends_only_on_itself_and_its_libraries():
    assert _violations(_package_files(), SRC, LIBRARIES, tests=False) == set()


def test_the_package_tests_depend_only_on_the_package():
    assert _violations(_test_files(), ROOT, TEST_LIBRARIES, tests=True) == set()


def test_the_scan_roots_are_correct_and_the_scans_are_not_empty():
    """Each rule can pass vacuously on a wrong root or an empty scan; pin both."""
    assert (ROOT / "pyproject.toml").is_file()
    assert (PACKAGE / "__init__.py").is_file()
    assert PACKAGE / "refs.py" in _package_files()
    assert Path(__file__) in _test_files()


def test_a_provider_sdk_import_is_a_violation(tmp_path):
    for line in (
        "import anthropic",
        "from google.genai import types",
        "import openai",
        "from langchain_core.messages import AIMessage",
        "from agentic_cli.workflow.settings import WorkflowSettingsMixin",
        "from agentic_cli import logging",
        "from agentic_cli.memory import MemoryStore",
    ):
        path = tmp_path / "mod.py"
        path.write_text(line + "\n")
        assert _violations([path], tmp_path, LIBRARIES, tests=False) != set(), line


def test_function_level_and_type_checking_imports_count(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from agentic_cli.config import BaseSettings\n"
        "def build():\n"
        "    import anthropic\n"
    )

    assert {target for _, target in _violations([path], tmp_path, LIBRARIES, tests=False)} == {
        "agentic_cli.config",
        "anthropic",
    }


def test_a_relative_import_resolves_inside_the_package(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("from . import refs\nfrom .records import ModelRecord\n")

    assert imports_of(path, "agentic_cli.models.roles") == [
        "agentic_cli.models.refs",
        "agentic_cli.models.records",
    ]
