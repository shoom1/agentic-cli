"""Tests for BaseWorkflowManager._build_tools() — tool assembly and state tool swapping."""

from typing import Any, AsyncGenerator

import pytest

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import WorkflowEvent


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing base class methods
# ---------------------------------------------------------------------------


class _TestWorkflowManager(BaseWorkflowManager):
    """Concrete subclass for testing _build_tools()."""

    def __init__(self, *args, state_tools=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_state_tools = state_tools or []

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

    async def _extract_session_data(self, session_id: str) -> tuple[list[dict], str | None]:
        return [], None

    async def _inject_session_messages(
        self, session_id: str, messages: list[dict], current_agent: str | None = None
    ) -> None:
        pass

    def _get_state_tools(self) -> list:
        if self._custom_state_tools:
            return self._custom_state_tools
        return super()._get_state_tools()


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _dummy_tool():
    """A dummy non-state tool."""
    pass


_dummy_tool.__name__ = "my_custom_tool"


def _another_dummy():
    """Another non-state tool."""
    pass


_another_dummy.__name__ = "another_tool"


def _fake_save_plan(content: str) -> dict:
    """Fake state tool."""
    return {"fake": True}


_fake_save_plan.__name__ = "save_plan"


def _fake_get_plan() -> dict:
    """Fake state tool."""
    return {"fake": True}


_fake_get_plan.__name__ = "get_plan"


def _fake_save_tasks(tasks: list) -> dict:
    """Fake state tool."""
    return {"fake": True}


_fake_save_tasks.__name__ = "save_tasks"


def _fake_get_tasks() -> dict:
    """Fake state tool."""
    return {"fake": True}


_fake_get_tasks.__name__ = "get_tasks"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildTools:
    """Tests for _build_tools() method."""

    def test_passes_through_non_state_tools(self, mock_context):
        mgr = _TestWorkflowManager(agent_configs=[], settings=mock_context.settings)
        config = AgentConfig(
            name="agent",
            prompt="test",
            tools=[_dummy_tool, _another_dummy],
        )
        result = mgr._build_tools(config)
        assert result == [_dummy_tool, _another_dummy]

    def test_replaces_state_tools_with_backend_specific(self, mock_context):
        from agentic_cli.tools.planning_tools import save_plan, get_plan
        from agentic_cli.tools.task_tools import save_tasks, get_tasks

        fake_tools = [_fake_save_plan, _fake_get_plan, _fake_save_tasks, _fake_get_tasks]
        mgr = _TestWorkflowManager(
            agent_configs=[],
            settings=mock_context.settings,
            state_tools=fake_tools,
        )
        config = AgentConfig(
            name="agent",
            prompt="test",
            tools=[_dummy_tool, save_plan, get_plan, save_tasks, get_tasks],
        )
        result = mgr._build_tools(config)
        # Non-state tool passes through
        assert result[0] is _dummy_tool
        # State tools are replaced
        assert result[1] is _fake_save_plan
        assert result[2] is _fake_get_plan
        assert result[3] is _fake_save_tasks
        assert result[4] is _fake_get_tasks

    def test_handles_empty_tool_list(self, mock_context):
        mgr = _TestWorkflowManager(agent_configs=[], settings=mock_context.settings)
        config = AgentConfig(name="agent", prompt="test", tools=[])
        result = mgr._build_tools(config)
        assert result == []

    def test_handles_none_tool_list(self, mock_context):
        mgr = _TestWorkflowManager(agent_configs=[], settings=mock_context.settings)
        config = AgentConfig(name="agent", prompt="test", tools=None)
        result = mgr._build_tools(config)
        assert result == []

    def test_mixed_state_and_regular_tools(self, mock_context):
        from agentic_cli.tools.planning_tools import save_plan

        fake_tools = [_fake_save_plan, _fake_get_plan, _fake_save_tasks, _fake_get_tasks]
        mgr = _TestWorkflowManager(
            agent_configs=[],
            settings=mock_context.settings,
            state_tools=fake_tools,
        )
        config = AgentConfig(
            name="agent",
            prompt="test",
            tools=[_dummy_tool, save_plan, _another_dummy],
        )
        result = mgr._build_tools(config)
        assert len(result) == 3
        assert result[0] is _dummy_tool
        assert result[1] is _fake_save_plan  # replaced
        assert result[2] is _another_dummy

    def test_state_tool_not_in_override_map_passes_through(self, mock_context):
        """If backend provides only some state tools, others pass through."""
        # Only provide save_plan replacement, not get_plan
        mgr = _TestWorkflowManager(
            agent_configs=[],
            settings=mock_context.settings,
            state_tools=[_fake_save_plan],  # only save_plan
        )

        from agentic_cli.tools.planning_tools import save_plan, get_plan
        config = AgentConfig(
            name="agent",
            prompt="test",
            tools=[save_plan, get_plan],
        )
        result = mgr._build_tools(config)
        assert result[0] is _fake_save_plan  # replaced
        assert result[1] is get_plan  # NOT replaced (no fake_get_plan in state_tools)


class TestDefaultGetStateTools:
    """Tests for default _get_state_tools() implementation."""

    def test_returns_legacy_service_registry_tools(self, mock_context):
        mgr = _TestWorkflowManager(agent_configs=[], settings=mock_context.settings)
        # Don't pass custom state_tools, so we get the base class default
        mgr._custom_state_tools = []
        tools = mgr._get_state_tools()
        names = {t.__name__ for t in tools}
        assert names == {"save_plan", "get_plan", "save_tasks", "get_tasks"}
