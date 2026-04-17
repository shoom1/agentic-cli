"""Import-hygiene tests enforcing backend isolation.

ADK and LangGraph are pluggable orchestrators behind BaseWorkflowManager.
Neither backend should depend on the other — shared logic lives in
workflow.* (e.g. workflow.confirmation, workflow.events, workflow.service_registry).
If one backend imports from the other, adding a third orchestrator would
drag the dependency along.
"""

from __future__ import annotations

import re
from pathlib import Path

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
