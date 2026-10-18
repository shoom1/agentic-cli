"""The workflow exposes one ``KnowledgeBase`` and the memory store.

It built two ``KnowledgeBaseManager``s under ``kb_manager`` and
``user_kb_manager``, and every tool merged them by hand. It now builds one
``KnowledgeBase`` under ``knowledge_base``, which the framework's tools
declare. Until 0.7.0 the old names keep working: the two keys still hold
the scopes' managers, ``requires="kb_manager"`` builds the same services
(with a warning when the tool registers), and the old properties warn.
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
from agentic_cli.workflow.service_registry import (
    KB_MANAGER,
    KNOWLEDGE_BASE,
    MEMORY_STORE,
    USER_KB_MANAGER,
)


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
    assert mgr.services[KB_MANAGER] is kb.manager("project")
    assert mgr.services[USER_KB_MANAGER] is kb.manager("user")


def test_one_scope_when_both_are_one_directory(tmp_path):
    mgr = _manager(tmp_path, {"knowledge_base"}, workspace=tmp_path / "project" / ".agentic_cli")

    assert mgr.services[KNOWLEDGE_BASE].scopes == ("project",)
    assert mgr.services[KB_MANAGER] is mgr.services[USER_KB_MANAGER]


def test_a_legacy_declaration_builds_the_same_services(tmp_path):
    mgr = _manager(tmp_path, {"kb_manager"})

    assert isinstance(mgr.services[KNOWLEDGE_BASE], KnowledgeBase)
    assert mgr.services[KB_MANAGER] is mgr.services[KNOWLEDGE_BASE].manager("project")


def test_a_legacy_declaration_and_a_new_one_build_one_knowledge_base(tmp_path):
    mgr = _manager(tmp_path, {"kb_manager", "knowledge_base"})
    kb = mgr.services[KNOWLEDGE_BASE]

    assert mgr.services[KB_MANAGER] is kb.manager("project")
    assert mgr.services[USER_KB_MANAGER] is kb.manager("user")


def test_declaring_kb_manager_warns_at_registration():
    registry = ToolRegistry()

    with pytest.warns(DeprecationWarning, match="knowledge_base.*0.7.0|0.7.0.*knowledge_base"):

        @registry.register(capabilities=EXEMPT, requires="kb_manager")
        def kb_thing() -> dict:
            """A downstream tool written for 0.6.3."""
            return {"success": True}

    assert registry.get("kb_thing").requires == ("kb_manager",)


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


@pytest.mark.parametrize(("name", "key"), [
    ("kb_manager", KB_MANAGER), ("user_kb_manager", USER_KB_MANAGER), ("memory_manager", MEMORY_STORE),
])
def test_the_old_properties_warn_and_still_work(tmp_path, name, key):
    mgr = _manager(tmp_path, {"knowledge_base", "memory_store"})

    with pytest.warns(DeprecationWarning, match="0.7.0"):
        value = getattr(mgr, name)

    assert value is mgr.services[key]


def test_a_model_switch_carries_the_knowledge_base(tmp_path):
    mgr = _manager(tmp_path, {"knowledge_base"})
    kb = mgr.services[KNOWLEDGE_BASE]

    with mgr._carrying_services(True):
        mgr._cleanup_managers()

    assert mgr.services[KNOWLEDGE_BASE] is kb
    assert mgr.services[KB_MANAGER] is kb.manager("project")


async def test_the_bound_tools_use_the_workflows_knowledge_base(tmp_path):
    mgr = _manager(tmp_path, {"knowledge_base"})
    kb = mgr.services[KNOWLEDGE_BASE]
    kb.manager("user").ingest_document(content="Glacier notes.", title="Glacier", source_type=SourceType.USER)
    tools = mgr._get_service_tool_map()

    result = tools["kb_search"]("glacier")

    assert [(r["document_title"], r["scope"]) for r in result["results"]] == [("Glacier", "user")]
    from agentic_cli.tools.knowledge_tools import kb_search

    assert identify_tool(tools["kb_search"]) is identify_tool(kb_search)


def _deprecations(caught):
    return [w for w in caught if issubclass(w.category, DeprecationWarning) and "kb_manager" in str(w.message)]


def test_the_warning_points_at_the_decorator_site():
    registry = ToolRegistry()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @registry.register(capabilities=EXEMPT, requires="kb_manager")
        def kb_thing() -> dict:
            """A downstream tool."""
            return {"success": True}

    assert [w.filename for w in _deprecations(caught)] == [__file__]


def test_the_warning_points_at_a_direct_register_call():
    registry = ToolRegistry()

    def kb_thing() -> dict:
        """A downstream tool."""
        return {"success": True}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        registry.register(kb_thing, capabilities=EXEMPT, requires="kb_manager")

    assert [w.filename for w in _deprecations(caught)] == [__file__]


def test_the_warning_points_at_register_tool():
    from agentic_cli.tools.registry import get_registry, register_tool

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @register_tool(capabilities=EXEMPT, requires="kb_manager")
        def kb_wiring_probe_tool() -> dict:
            """A downstream tool."""
            return {"success": True}

    get_registry()._tools.pop("kb_wiring_probe_tool", None)
    assert [w.filename for w in _deprecations(caught)] == [__file__]


def test_the_warning_points_at_declare_tool():
    from agentic_cli.tools.registry import declare_tool

    registry = ToolRegistry()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        declare_tool("kb_declared", description="d", capabilities=EXEMPT, requires="kb_manager", registry=registry)

    assert [w.filename for w in _deprecations(caught)] == [__file__]
