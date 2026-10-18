"""Tests for tool reference resolution (Phase 2).

Covers ``resolve_tool``/``resolve_tools`` (callable / bare-name / dotted-path)
and the base-manager integration that resolves config tool refs before
service detection.
"""

from __future__ import annotations

import math
import sys
import types
import warnings
from typing import AsyncGenerator

import pytest

from agentic_cli.tools.registry import ToolCategory, ToolRegistry
from agentic_cli.tools.tool_resolver import resolve_tool, resolve_tools
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.permissions import EXEMPT


# ---------------------------------------------------------------------------
# resolve_tool / resolve_tools
# ---------------------------------------------------------------------------


class TestResolveTool:
    def test_callable_passes_through(self):
        def my_tool():
            pass

        assert resolve_tool(my_tool) is my_tool

    def test_non_string_object_passes_through(self):
        # e.g. an already-built toolset object
        sentinel = object()
        assert resolve_tool(sentinel) is sentinel

    def test_dotted_path_imports(self):
        assert resolve_tool("math.sqrt") is math.sqrt

    def test_dotted_path_bad_module_raises(self):
        with pytest.raises(ValueError, match="Cannot import module"):
            resolve_tool("nonexistent_module_xyz.foo")

    def test_dotted_path_bad_attr_raises(self):
        with pytest.raises(ValueError, match="has no attribute"):
            resolve_tool("math.nonexistent_attr_xyz")

    def test_bare_name_via_explicit_registry(self):
        reg = ToolRegistry()

        def custom():
            pass

        reg.register(custom, capabilities=EXEMPT, category=ToolCategory.OTHER)
        assert resolve_tool("custom", registry=reg) is custom

    def test_unknown_bare_name_raises_with_hint(self):
        reg = ToolRegistry()

        def kb_search():
            pass

        reg.register(kb_search, capabilities=EXEMPT, category=ToolCategory.OTHER)
        with pytest.raises(ValueError, match="Did you mean: kb_search"):
            resolve_tool("kb_searh", registry=reg)  # typo

    def test_empty_ref_raises(self):
        with pytest.raises(ValueError, match="Empty tool reference"):
            resolve_tool("   ")

    def test_bare_name_resolves_builtin_via_default_registry(self):
        # Triggers lazy import of agentic_cli.tools to populate the registry.
        tool = resolve_tool("read_file")
        assert callable(tool)
        assert getattr(tool, "__name__", None) == "read_file"


class TestResolveTools:
    def test_none_returns_empty(self):
        assert resolve_tools(None) == []

    def test_mixed_list(self):
        def cb():
            pass

        out = resolve_tools([cb, "math.sqrt"])
        assert out == [cb, math.sqrt]


# ---------------------------------------------------------------------------
# Base-manager integration
# ---------------------------------------------------------------------------


class TestDeprecationsInDottedReferences:
    """A dotted tool path comes from configuration, so no frame of the
    application's code is on the stack while it resolves, and Python's default
    filters would hide a DeprecationWarning raised there. The resolver re-issues
    it as a FutureWarning that names the reference."""

    @pytest.fixture
    def module(self, monkeypatch):
        module = types.ModuleType("deprecating_tools_fixture")

        def kept_tool() -> str:
            return "kept"

        def __getattr__(name: str):
            if name == "old_tool":
                warnings.warn(
                    "old_tool is deprecated; use kept_tool", DeprecationWarning, stacklevel=2
                )
                return kept_tool
            raise AttributeError(name)

        module.kept_tool = kept_tool
        module.__getattr__ = __getattr__
        monkeypatch.setitem(sys.modules, module.__name__, module)
        return module

    def test_a_deprecation_is_reissued_as_a_future_warning_naming_the_reference(self, module):
        with pytest.warns(FutureWarning) as caught:
            obj = resolve_tool("deprecating_tools_fixture.old_tool")
        assert obj is module.kept_tool
        assert [str(w.message) for w in caught if issubclass(w.category, FutureWarning)] == [
            "tool reference 'deprecating_tools_fixture.old_tool': "
            "old_tool is deprecated; use kept_tool"
        ]
        assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]

    def test_an_error_filter_on_deprecations_does_not_break_resolution(self, module):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            assert resolve_tool("deprecating_tools_fixture.old_tool") is module.kept_tool

    def test_a_reference_that_warns_nothing_stays_silent(self, module):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_tool("deprecating_tools_fixture.kept_tool") is module.kept_tool


class _MiniManager(BaseWorkflowManager):
    """Minimal concrete manager to exercise base-class resolution/detection."""

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

    async def reinitialize(self, model=None, preserve_sessions=True) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    async def _extract_session_data(self, session_id: str):
        return [], None

    async def _inject_session_messages(
        self, session_id: str, messages, current_agent=None
    ) -> None:
        pass

    def _get_state_tools(self) -> list:
        return []


class TestBaseManagerIntegration:
    def test_string_tool_resolved_and_service_detected(self, mock_context):
        # "kb_search" is a registered service tool -> resolution must make it a
        # callable AND detection must flag the knowledge_base service.
        cfg = AgentConfig(
            name="a", prompt="p", tools=["kb_search"], include_state_tools=False
        )
        mgr = _MiniManager(agent_configs=[cfg], settings=mock_context.settings)

        assert callable(cfg.tools[0])
        assert getattr(cfg.tools[0], "__name__", None) == "kb_search"
        assert "knowledge_base" in mgr.required_managers

    def test_callable_tools_unaffected(self, mock_context):
        def my_tool():
            pass

        cfg = AgentConfig(
            name="a", prompt="p", tools=[my_tool], include_state_tools=False
        )
        mgr = _MiniManager(agent_configs=[cfg], settings=mock_context.settings)
        assert cfg.tools == [my_tool]
        assert mgr.required_managers == set()

    def test_dotted_path_tool_resolved(self, mock_context):
        cfg = AgentConfig(
            name="a", prompt="p", tools=["math.sqrt"], include_state_tools=False
        )
        _MiniManager(agent_configs=[cfg], settings=mock_context.settings)
        assert cfg.tools[0] is math.sqrt
