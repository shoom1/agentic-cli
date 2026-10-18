"""The workflow exposes one ``KnowledgeBase`` and the memory store.

It built two ``KnowledgeBaseManager``s under ``kb_manager`` and
``user_kb_manager``, and every tool merged them by hand. It now builds one
``KnowledgeBase`` under ``knowledge_base``, which the framework's tools
declare, and publishes no other knowledge-base key.
"""

from __future__ import annotations

import warnings
from typing import AsyncGenerator

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.memory.kb import KnowledgeBase, SourceType
from agentic_cli.tools.registry import ToolRegistry, identify_tool
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.permissions import EXEMPT
from agentic_cli.workflow.service_registry import KNOWLEDGE_BASE, MEMORY_STORE


class _Manager(BaseWorkflowManager):
    @property
    def backend_type(self) -> str:
        return "test"

    async def _do_initialize(self) -> None:
        pass

    async def process(
        self, message: str, user_id: str, session_id: str | None = None
    ) -> AsyncGenerator[WorkflowEvent, None]:
        if False:
            yield  # type: ignore[misc]

    async def reinitialize(self, model: str | None = None, preserve_sessions: bool = True) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    def _get_state_tools(self) -> list:
        return []


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")


def _manager(tmp_path, required, workspace=None) -> _Manager:
    settings = BaseSettings(workspace_dir=workspace or tmp_path / "workspace", knowledge_base_use_mock=True)
    mgr = _Manager(agent_configs=[], settings=settings)
    mgr._required_managers = set(required)
    mgr._ensure_managers_initialized()
    return mgr


def test_one_knowledge_base_over_both_scopes(tmp_path):
    mgr = _manager(tmp_path, {"knowledge_base"})
    kb = mgr.services[KNOWLEDGE_BASE]

    assert isinstance(kb, KnowledgeBase)
    assert kb.scopes == ("project", "user")


async def test_service_init_publishes_the_knowledge_base_and_no_0_6_3_keys(tmp_path):
    from agentic_cli.tools.knowledge_tools import kb_search
    from agentic_cli.workflow.config import AgentConfig

    settings = BaseSettings(workspace_dir=tmp_path / "workspace", knowledge_base_use_mock=True)
    config = AgentConfig(name="a", prompt="p", tools=[kb_search])
    mgr = _Manager(agent_configs=[config], settings=settings)

    await mgr.initialize_services(validate=False)

    assert isinstance(mgr.services[KNOWLEDGE_BASE], KnowledgeBase)
    assert "kb_manager" not in mgr.services
    assert "user_kb_manager" not in mgr.services


def test_one_scope_when_both_are_one_directory(tmp_path):
    mgr = _manager(tmp_path, {"knowledge_base"}, workspace=tmp_path / "project" / ".agentic_cli")

    assert mgr.services[KNOWLEDGE_BASE].scopes == ("project",)


def test_declaring_knowledge_base_does_not_warn():
    registry = ToolRegistry()

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        @registry.register(capabilities=EXEMPT, requires="knowledge_base")
        def kb_thing() -> dict:
            """A tool written for 0.6.4."""
            return {"success": True}


def test_the_framework_tools_declare_the_knowledge_base():
    from agentic_cli.tools.arxiv_tools import ingest_arxiv_paper
    from agentic_cli.tools.knowledge_tools import kb_search

    assert identify_tool(kb_search).requires == ("knowledge_base",)
    assert identify_tool(ingest_arxiv_paper).requires == ("arxiv_source", "knowledge_base")


def test_the_new_properties(tmp_path):
    mgr = _manager(tmp_path, {"knowledge_base", "memory_store"})

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert mgr.knowledge_base is mgr.services[KNOWLEDGE_BASE]
        assert mgr.memory_store is mgr.services[MEMORY_STORE]


@pytest.mark.parametrize("name", ["kb_manager", "user_kb_manager", "memory_manager"])
def test_the_0_6_3_properties_are_gone(tmp_path, name):
    mgr = _manager(tmp_path, {"knowledge_base", "memory_store"})

    assert not hasattr(mgr, name)


def test_a_model_switch_carries_the_knowledge_base(tmp_path):
    mgr = _manager(tmp_path, {"knowledge_base"})
    kb = mgr.services[KNOWLEDGE_BASE]

    with mgr._carrying_services(True):
        mgr._cleanup_managers()

    assert mgr.services[KNOWLEDGE_BASE] is kb


async def test_the_bound_tools_use_the_workflows_knowledge_base(tmp_path):
    mgr = _manager(tmp_path, {"knowledge_base"})
    kb = mgr.services[KNOWLEDGE_BASE]
    kb.manager("user").ingest_document(content="Glacier notes.", title="Glacier", source_type=SourceType.USER)
    tools = mgr._get_service_tool_map()

    result = tools["kb_search"]("glacier")

    assert [(r["document_title"], r["scope"]) for r in result["results"]] == [("Glacier", "user")]
    from agentic_cli.tools.knowledge_tools import kb_search

    assert identify_tool(tools["kb_search"]) is identify_tool(kb_search)


def _injected(tmp_path, required, **services):
    """A manager with services published before the build, as tests inject them."""
    settings = BaseSettings(workspace_dir=tmp_path / "workspace", knowledge_base_use_mock=True)
    mgr = _Manager(agent_configs=[], settings=settings)
    mgr._required_managers = set(required)
    mgr._services.update(services)
    mgr._ensure_managers_initialized()
    return mgr


def _mock_manager(path, title):
    from agentic_cli.memory.kb import KnowledgeBaseManager

    manager = KnowledgeBaseManager(path, use_mock=True)
    manager.ingest_document(content=f"{title} body.", title=title, source_type=SourceType.USER)
    return manager


def test_an_injected_knowledge_base_is_the_one_the_tools_use(tmp_path):
    injected = KnowledgeBase({"project": _mock_manager(tmp_path / "injected", "Injected doc")})
    mgr = _injected(tmp_path, {"knowledge_base"}, **{KNOWLEDGE_BASE: injected})

    assert mgr.knowledge_base is injected
    listed = mgr._get_service_tool_map()["kb_list"]()
    assert [d["title"] for d in listed["documents"]] == ["Injected doc"]


def test_an_injected_knowledge_base_gives_ingest_arxiv_paper_its_knowledge_base(tmp_path):
    injected = KnowledgeBase({"project": _mock_manager(tmp_path / "injected", "Injected doc")})
    mgr = _injected(
        tmp_path, {"knowledge_base", "arxiv_source"}, **{KNOWLEDGE_BASE: injected, "arxiv_source": object()}
    )

    assert "ingest_arxiv_paper" in mgr._get_service_tool_map()


def _app_with(mgr):
    from tests.cli.test_papers_command import _App

    class _WithWorkflow(_App):
        workflow = mgr

    return _WithWorkflow(mgr._settings)


async def test_papers_lists_the_injected_knowledge_base(tmp_path):
    from agentic_cli.cli.builtin_commands import PapersCommand
    injected = KnowledgeBase({"project": _mock_manager(tmp_path / "injected", "Injected doc")})
    mgr = _injected(tmp_path, {"knowledge_base"}, **{KNOWLEDGE_BASE: injected})
    app = _app_with(mgr)
    await PapersCommand().execute("", app)

    [table] = app.session.tables
    assert [str(cell) for cell in table.columns[1]._cells] == ["Injected doc"]


async def test_papers_global_falls_back_when_only_the_project_scope_is_injected(tmp_path):
    from agentic_cli.cli.builtin_commands import PapersCommand
    injected = KnowledgeBase({"project": _mock_manager(tmp_path / "injected", "Injected doc")})
    mgr = _injected(tmp_path, {"knowledge_base"}, **{KNOWLEDGE_BASE: injected})
    assert mgr.knowledge_base.scopes == ("project",)
    app = _app_with(mgr)
    await PapersCommand().execute("--global", app)

    assert app.session.messages == ["No documents found in user knowledge base."]


def test_an_injected_knowledge_base_gets_no_0_6_3_keys(tmp_path):
    project = _mock_manager(tmp_path / "project", "P")
    user = _mock_manager(tmp_path / "user", "U")
    kb = KnowledgeBase({"project": project, "user": user})
    mgr = _injected(tmp_path, {"knowledge_base"}, **{KNOWLEDGE_BASE: kb})

    assert mgr.services[KNOWLEDGE_BASE] is kb
    assert "kb_manager" not in mgr.services
    assert "user_kb_manager" not in mgr.services
