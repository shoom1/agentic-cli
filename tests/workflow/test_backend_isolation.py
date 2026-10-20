"""Import-hygiene tests enforcing backend isolation.

ADK, LangGraph and Deep Agents are pluggable orchestrators behind
BaseWorkflowManager. No backend should depend on another — shared logic lives
in workflow.* (e.g. workflow.events, workflow.service_registry,
workflow.permissions, workflow.checkpointers). If one backend imports from
another, adding or removing an orchestrator would drag the dependency along.
Deep Agents may use the LangGraph *library* state tools
(``agentic_cli.tools.langgraph``), which are not the LangGraph backend.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("langgraph")

import agentic_cli.workflow.langgraph
import agentic_cli.workflow.adk


_ADK_IMPORT = re.compile(
    r"^\s*(from\s+agentic_cli\.workflow\.adk|import\s+agentic_cli\.workflow\.adk)",
    re.MULTILINE,
)
_LG_IMPORT = re.compile(
    r"^\s*(from\s+agentic_cli\.workflow\.langgraph|import\s+agentic_cli\.workflow\.langgraph)",
    re.MULTILINE,
)
_DA_IMPORT = re.compile(
    r"^\s*(from\s+agentic_cli\.workflow\.deepagents|import\s+agentic_cli\.workflow\.deepagents)",
    re.MULTILINE,
)


def _python_files(pkg) -> list[Path]:
    root = Path(pkg.__file__).parent
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_langgraph_does_not_import_from_adk():
    offenders: list[str] = []
    for pyfile in _python_files(agentic_cli.workflow.langgraph):
        text = pyfile.read_text()
        if _ADK_IMPORT.search(text):
            offenders.append(str(pyfile))
    assert not offenders, (
        "LangGraph backend must not import from workflow.adk.*. "
        f"Offending files: {offenders}. "
        "Move shared logic to workflow/<name>.py (e.g. workflow/confirmation.py)."
    )


def test_adk_does_not_import_from_langgraph():
    offenders: list[str] = []
    for pyfile in _python_files(agentic_cli.workflow.adk):
        text = pyfile.read_text()
        if _LG_IMPORT.search(text):
            offenders.append(str(pyfile))
    assert not offenders, (
        "ADK backend must not import from workflow.langgraph.*. "
        f"Offending files: {offenders}."
    )


# Scanned by path, not imported: the scan needs no ``deepagents`` extra.
_DEEPAGENTS = SimpleNamespace(
    __file__=str(Path(agentic_cli.workflow.adk.__file__).parents[1] / "deepagents" / "__init__.py")
)


def _offenders(pkg, pattern: re.Pattern) -> list[str]:
    return [str(f) for f in _python_files(pkg) if pattern.search(f.read_text())]


def test_deepagents_imports_neither_other_backend():
    assert _python_files(_DEEPAGENTS), "the Deep Agents backend was not found"
    offenders = _offenders(_DEEPAGENTS, _ADK_IMPORT) + _offenders(_DEEPAGENTS, _LG_IMPORT)
    assert not offenders, (
        "Deep Agents backend must not import from workflow.adk.* or "
        f"workflow.langgraph.*. Offending files: {offenders}."
    )


def test_no_other_backend_imports_deepagents():
    offenders = _offenders(agentic_cli.workflow.adk, _DA_IMPORT) + _offenders(
        agentic_cli.workflow.langgraph, _DA_IMPORT
    )
    assert not offenders, (
        "ADK and LangGraph must not import from workflow.deepagents.*. "
        f"Offending files: {offenders}."
    )
