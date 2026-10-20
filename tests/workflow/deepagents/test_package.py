"""The ``deepagents`` extra and the backend package's import contract.

The backend lives in ``agentic_cli.workflow.deepagents`` and is installed by
the ``deepagents`` extra. Its package imports Deep Agents and LangChain's agent
factory eagerly, so a missing extra fails right at the import, where the
backend registry turns the ``ImportError`` into the install hint. No other
module imports Deep Agents: an application that never chooses the backend
never needs the extra.
"""

from __future__ import annotations

import importlib
import re
import sys
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SRC = _REPO / "src" / "agentic_cli"
_BACKEND = "agentic_cli.workflow.deepagents"

#: The spec's extra, pin for pin.
SPEC_EXTRA = [
    "deepagents>=0.7.28,<0.8",
    "langchain>=1.4.6",
    "langchain-core>=1.6.11",
    "langchain-openai>=1.7",
    "langchain-anthropic>=1.7.6",
    "langchain-google-genai>=4.4.2",
    "langchain-openrouter>=0.2",
    "langgraph-checkpoint-sqlite>=3.1",
    "langgraph-checkpoint-postgres>=3.1",
    "langchain-mcp-adapters>=0.3",
]


def _extras() -> dict[str, list[str]]:
    with (_REPO / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["optional-dependencies"]


class TestTheExtra:
    def test_it_declares_the_spec_pins(self):
        assert _extras()["deepagents"] == SPEC_EXTRA

    def test_it_installs_beside_the_langgraph_extra(self):
        """Both extras can be installed together until 0.8.0 removes LangGraph."""
        assert "langgraph" in _extras()


class TestEagerImport:
    """Without the extra, importing the backend fails, and fails at once."""

    @pytest.fixture
    def fresh_backend(self, monkeypatch):
        """Forget the backend package so the next import runs it again.

        The import binds the new package on ``agentic_cli.workflow`` too, so
        that attribute is restored with ``sys.modules``: otherwise later tests
        would reach two copies of every backend module, one by import and one
        by attribute.
        """
        import agentic_cli.workflow

        for name in [m for m in sys.modules if m == _BACKEND or m.startswith(_BACKEND + ".")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.delattr(agentic_cli.workflow, "deepagents", raising=False)

    @pytest.mark.parametrize("missing", ["deepagents", "langchain.agents"])
    def test_a_missing_dependency_fails_the_import(self, fresh_backend, monkeypatch, missing):
        # None in sys.modules makes `import <missing>` raise ModuleNotFoundError,
        # as it would without the extra.
        monkeypatch.setitem(sys.modules, missing, None)

        with pytest.raises(ImportError) as raised:
            importlib.import_module(_BACKEND)

        # The dependency is what is missing, not the backend package itself.
        assert raised.value.name == missing

    def test_with_the_extra_the_package_imports(self, fresh_backend):
        pytest.importorskip("deepagents")

        module = importlib.import_module(_BACKEND)

        assert module.__name__ == _BACKEND


_IMPORTS_DEEPAGENTS = re.compile(r"^\s*(from|import)\s+deepagents\b", re.MULTILINE)


def test_nothing_outside_the_backend_imports_deepagents():
    backend = _SRC / "workflow" / "deepagents"
    offenders = [
        str(path.relative_to(_SRC))
        for path in sorted(_SRC.rglob("*.py"))
        if "__pycache__" not in path.parts
        and not path.is_relative_to(backend)
        and _IMPORTS_DEEPAGENTS.search(path.read_text())
    ]
    assert offenders == [], (
        "Only workflow/deepagents/ may import deepagents; an application that "
        f"never chooses the backend must not need the extra. Offenders: {offenders}"
    )
