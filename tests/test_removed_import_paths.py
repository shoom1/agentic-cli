"""The import paths 0.7.0 removed fail instead of forwarding."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "agentic_cli.knowledge_base",
        "agentic_cli.knowledge_base.manager",
        "agentic_cli.tools.kb_convert",
        "agentic_cli.workflow.adk.transfer_tool_description",
    ],
)
def test_a_removed_module_cannot_be_imported(module, monkeypatch):
    monkeypatch.delitem(sys.modules, module, raising=False)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_memory_tools_no_longer_exports_a_memory_store():
    with pytest.raises(ImportError):
        from agentic_cli.tools.memory_tools import MemoryStore  # noqa: F401


def test_kb_ingest_url_is_gone():
    import agentic_cli.tools as tools
    from agentic_cli.tools.registry import get_registry

    with pytest.raises(ImportError):
        from agentic_cli.tools import kb_ingest_url  # noqa: F401
    assert "kb_ingest_url" not in tools.__all__
    assert get_registry().get("kb_ingest_url") is None


def test_tools_google_search_tool_is_gone_and_loads_no_adk():
    """Asked for in a fresh interpreter: the lookup must not import ADK first."""
    body = textwrap.dedent("""
        import json, sys
        import agentic_cli.tools as tools
        try:
            tools.google_search_tool
            raised = False
        except AttributeError:
            raised = True
        print(json.dumps({
            "raised": raised,
            "adk": sorted(m for m in sys.modules if m.startswith("google.adk")),
        }))
    """)
    proc = subprocess.run([sys.executable, "-c", body], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    assert result == {"raised": True, "adk": []}
