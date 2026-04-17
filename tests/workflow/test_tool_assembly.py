"""Tests for BaseWorkflowManager._build_tools() — tool assembly and state tool injection."""

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
        return self._custom_state_tools


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
        fake_tools = [_fake_save_plan, _fake_get_plan, _fake_save_tasks, _fake_get_tasks]
        mgr = _TestWorkflowManager(
            agent_configs=[], settings=mock_context.settings, state_tools=fake_tools,
        )
        config = AgentConfig(
            name="agent",
            prompt="test",
            tools=[_dummy_tool, _another_dummy],
            include_state_tools=False,
        )
        result = mgr._build_tools(config)
        assert result == [_dummy_tool, _another_dummy]

    def test_auto_injects_state_tools_when_enabled(self, mock_context):
        fake_tools = [_fake_save_plan, _fake_get_plan, _fake_save_tasks, _fake_get_tasks]
        mgr = _TestWorkflowManager(
            agent_configs=[], settings=mock_context.settings, state_tools=fake_tools,
        )
        config = AgentConfig(
            name="agent",
            prompt="test",
            tools=[_dummy_tool],
        )
        result = mgr._build_tools(config)
        assert result[0] is _dummy_tool
        assert result[1] is _fake_save_plan
        assert result[2] is _fake_get_plan
        assert result[3] is _fake_save_tasks
        assert result[4] is _fake_get_tasks

    def test_no_state_tools_when_disabled(self, mock_context):
        fake_tools = [_fake_save_plan, _fake_get_plan, _fake_save_tasks, _fake_get_tasks]
        mgr = _TestWorkflowManager(
            agent_configs=[], settings=mock_context.settings, state_tools=fake_tools,
        )
        config = AgentConfig(
            name="agent",
            prompt="test",
            tools=[_dummy_tool],
            include_state_tools=False,
        )
        result = mgr._build_tools(config)
        assert result == [_dummy_tool]

    def test_handles_empty_tool_list_with_state_tools(self, mock_context):
        fake_tools = [_fake_save_plan, _fake_get_plan, _fake_save_tasks, _fake_get_tasks]
        mgr = _TestWorkflowManager(
            agent_configs=[], settings=mock_context.settings, state_tools=fake_tools,
        )
        config = AgentConfig(name="agent", prompt="test", tools=[])
        result = mgr._build_tools(config)
        # Empty tool list + state tools injected
        assert len(result) == 4
        assert result[0] is _fake_save_plan

    def test_handles_empty_tool_list_without_state_tools(self, mock_context):
        mgr = _TestWorkflowManager(
            agent_configs=[], settings=mock_context.settings, state_tools=[],
        )
        config = AgentConfig(name="agent", prompt="test", tools=[], include_state_tools=False)
        result = mgr._build_tools(config)
        assert result == []

    def test_handles_none_tool_list(self, mock_context):
        mgr = _TestWorkflowManager(
            agent_configs=[], settings=mock_context.settings, state_tools=[],
        )
        config = AgentConfig(name="agent", prompt="test", tools=None, include_state_tools=False)
        result = mgr._build_tools(config)
        assert result == []

    def test_default_include_state_tools_is_true(self, mock_context):
        """AgentConfig defaults to include_state_tools=True."""
        config = AgentConfig(name="agent", prompt="test")
        assert config.include_state_tools is True
