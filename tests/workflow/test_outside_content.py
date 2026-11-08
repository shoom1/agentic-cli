"""Which tool calls bring outside content into a conversation (spec §9.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

import agentic_cli.tools  # noqa: F401  (registers the framework's tools)
import agentic_cli.tools.execution_tools  # noqa: F401
import agentic_cli.tools.memory_tools  # noqa: F401
import agentic_cli.tools.webfetch_tool  # noqa: F401
from agentic_cli.config import BaseSettings, set_context_settings
from agentic_cli.tools.registry import ToolCategory, ToolRegistry, get_registry
from agentic_cli.workflow.outside_content import brings_outside_content
from agentic_cli.workflow.permissions import EXEMPT

MARKING = {
    "web_fetch", "web_search", "search_arxiv", "fetch_arxiv_paper", "ingest_arxiv_paper",
    "kb_search", "kb_read", "kb_list", "kb_search_concepts", "sandbox_execute",
    "execute_python", "job_result", "job_logs", "job_status", "run_skill_script",
}
NOT_MARKING = {
    "kb_ingest_text", "kb_ingest_file", "kb_write_concept", "ask_clarification", "write_file",
    "edit_file", "glob", "list_dir", "save_memory", "read_memory", "search_memory",
}


def _definition(name):
    if name == "sandbox_execute":
        import agentic_cli.tools.sandbox  # noqa: F401
    if name == "run_skill_script":
        import agentic_cli.tools.skills  # noqa: F401
    return get_registry().get(name)


@pytest.mark.parametrize("name", sorted(MARKING))
def test_tools_that_bring_outside_content(name):
    assert brings_outside_content(_definition(name), {}) is True


@pytest.mark.parametrize("name", sorted(NOT_MARKING))
def test_tools_that_do_not(name):
    assert brings_outside_content(_definition(name), {}) is False


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("name, args, marks", [
    ("read_file", {"path": "notes.md"}, False),
    ("read_file", {"path": ".agentic_cli/fetched/page.md"}, True),
    ("read_file", {"path": ".agentic_cli/tool-output/r.txt"}, True),
    ("read_file", {"path": ".agentic_cli/settings.json"}, False),
    ("grep", {"pattern": "x", "path": ".agentic_cli/fetched"}, True),
    ("grep", {"pattern": "x", "path": "."}, False),
    ("diff_compare", {"source_a": "a.md", "source_b": ".agentic_cli/fetched/p.md"}, True),
    ("diff_compare", {"source_a": "a.md", "source_b": "b.md"}, False),
    # Every spelling of the folder is the folder: on a case-insensitive
    # filesystem (macOS, Windows) these read the same page.
    ("read_file", {"path": ".AGENTIC_CLI/FETCHED/page.md"}, True),
    ("read_file", {"path": ".Agentic_Cli/Tool-Output/r.txt"}, True),
    ("diff_compare", {"source_a": "a.md", "source_b": ".agentic_cli/FETCHED/p.md"}, True),
    ("kb_ingest_file", {"path": ".agentic_cli/fetched/page.md"}, True),
    ("kb_ingest_file", {"path": ".agentic_cli/tool-output/r.txt"}, True),
    ("kb_ingest_file", {"path": "notes.md"}, False),
    # A folder is not a page: only grep, which searches below its path, reads
    # what is under a path that holds a saved folder.
    ("read_file", {"path": ".agentic_cli"}, False),
    # grep marks when its path is a saved folder, inside one, or above one
    # with no hidden entry between (it searches no hidden entry below its path).
    ("grep", {"pattern": "x", "path": ".agentic_cli"}, True),
    ("grep", {"pattern": "x", "path": ".AGENTIC_CLI"}, True),
    ("grep", {"pattern": "x", "path": ".agentic_cli/fetched/page.md"}, True),
    ("grep", {"pattern": "x", "path": ".agentic_cli/tool-output"}, True),
    ("grep", {"pattern": "x", "path": ".agentic_cli/knowledge_base"}, False),
    ("grep", {"pattern": "x", "path": ".."}, False),
    ("grep", {"pattern": "x"}, False),
])
def test_reading_a_saved_page_or_output_marks(project, name, args, marks):
    token = set_context_settings(BaseSettings(app_name="agentic_cli", workspace_dir=project / "w"))
    try:
        assert brings_outside_content(_definition(name), args) is marks
    finally:
        token.var.reset(token)


def test_an_unknown_tool_brings_nothing():
    assert brings_outside_content(None, {}) is False


def test_a_flag_that_raises_counts_as_outside_content():
    registry = ToolRegistry()

    def broken(args):
        raise RuntimeError("bug")

    @registry.register(category=ToolCategory.OTHER, capabilities=EXEMPT,
                       description="probe", outside_content=broken)
    def probe() -> dict:
        return {"success": True}

    assert brings_outside_content(registry.get("probe"), {}) is True


def test_a_variant_must_declare_the_same_flag():
    registry = ToolRegistry()
    registry.declare("shared", description="d", capabilities=EXEMPT, outside_content=True)
    with pytest.raises(ValueError, match="outside_content"):
        @registry.register(category=ToolCategory.OTHER, capabilities=EXEMPT, variant_of="shared")
        def shared_variant() -> dict:
            return {}


def test_the_langgraph_manager_marks_every_conversation(mock_context):
    """LangGraph's gate tracks no mark (spec 9.1): every memory write there asks."""
    pytest.importorskip("langgraph")
    from agentic_cli.workflow.config import AgentConfig
    from agentic_cli.workflow.langgraph.manager import LangGraphWorkflowManager
    from agentic_cli.workflow.outside_content import LANGGRAPH_MARK
    from agentic_cli.workflow.sessions import SessionRef

    manager = LangGraphWorkflowManager(
        agent_configs=[AgentConfig(name="a", prompt="p")], settings=mock_context.settings,
        model="gemini-2.5-flash",
    )
    assert manager._outside_marks.get(SessionRef(app_name='app', user_id='u', session_id='s')) == LANGGRAPH_MARK
    assert manager._outside_marks.get(None) == LANGGRAPH_MARK
