"""Tests for ADK TaskProgressPlugin.

Uses mock ToolContext with state containing tasks.
"""

import pytest

from agentic_cli.workflow.adk.task_progress_plugin import TaskProgressPlugin
from agentic_cli.workflow.events import EventType


class MockState(dict):
    """Dict-like state object mimicking ToolContext.state."""
    pass


class MockToolContext:
    """Minimal mock for google.adk.tools.tool_context.ToolContext."""

    def __init__(self, state: dict | None = None):
        self.state = MockState(state or {})


class MockTool:
    """Minimal mock for google.adk.tools.BaseTool."""

    def __init__(self, name: str = "some_tool"):
        self.name = name


class TestTaskProgressPlugin:
    """Tests for TaskProgressPlugin."""

    @pytest.fixture
    def plugin(self):
        return TaskProgressPlugin()

    async def test_no_tasks_no_events(self, plugin):
        ctx = MockToolContext()
        result = await plugin.after_tool_callback(
            tool=MockTool(),
            tool_args={},
            tool_context=ctx,
            result={"success": True},
        )
        assert result is None
        assert plugin.drain_events() == []

    async def test_buffers_progress_event(self, plugin):
        tasks = [
            {"id": "1", "description": "Active task", "status": "in_progress", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
            {"id": "2", "description": "Pending task", "status": "pending", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
        ]
        ctx = MockToolContext({"tasks": tasks})
        await plugin.after_tool_callback(
            tool=MockTool(),
            tool_args={},
            tool_context=ctx,
            result={"success": True},
        )
        events = plugin.drain_events()
        assert len(events) == 1
        event = events[0]
        assert event.type == EventType.TASK_PROGRESS
        assert "[▸] Active task" in event.content
        assert "[ ] Pending task" in event.content
        assert event.metadata["progress"]["total"] == 2
        assert event.metadata["progress"]["in_progress"] == 1

    async def test_drain_clears_buffer(self, plugin):
        tasks = [
            {"id": "1", "description": "Task 1", "status": "pending", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
        ]
        ctx = MockToolContext({"tasks": tasks})
        await plugin.after_tool_callback(
            tool=MockTool(),
            tool_args={},
            tool_context=ctx,
            result={"success": True},
        )
        assert len(plugin.drain_events()) == 1
        # Second drain should be empty
        assert plugin.drain_events() == []

    async def test_auto_clear_when_all_done(self, plugin):
        tasks = [
            {"id": "1", "description": "Done 1", "status": "completed", "priority": "medium", "tags": [], "created_at": "", "completed_at": "2024-01-01"},
            {"id": "2", "description": "Done 2", "status": "completed", "priority": "medium", "tags": [], "created_at": "", "completed_at": "2024-01-01"},
        ]
        ctx = MockToolContext({"tasks": tasks})
        await plugin.after_tool_callback(
            tool=MockTool(),
            tool_args={},
            tool_context=ctx,
            result={"success": True},
        )
        # Tasks should be auto-cleared
        assert ctx.state["tasks"] == []
        # But we still get the final progress event
        events = plugin.drain_events()
        assert len(events) == 1
        assert events[0].type == EventType.TASK_PROGRESS

    async def test_does_not_clear_when_not_all_done(self, plugin):
        tasks = [
            {"id": "1", "description": "Done", "status": "completed", "priority": "medium", "tags": [], "created_at": "", "completed_at": "2024-01-01"},
            {"id": "2", "description": "Pending", "status": "pending", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
        ]
        ctx = MockToolContext({"tasks": tasks})
        await plugin.after_tool_callback(
            tool=MockTool(),
            tool_args={},
            tool_context=ctx,
            result={"success": True},
        )
        # Tasks should NOT be cleared
        assert len(ctx.state["tasks"]) == 2

    async def test_multiple_tool_calls_buffer_multiple_events(self, plugin):
        tasks = [
            {"id": "1", "description": "Task", "status": "in_progress", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
        ]
        ctx = MockToolContext({"tasks": tasks})
        # Call twice
        await plugin.after_tool_callback(
            tool=MockTool("tool_a"), tool_args={}, tool_context=ctx, result={"success": True},
        )
        await plugin.after_tool_callback(
            tool=MockTool("tool_b"), tool_args={}, tool_context=ctx, result={"success": True},
        )
        events = plugin.drain_events()
        assert len(events) == 2

    async def test_does_not_modify_tool_result(self, plugin):
        tasks = [
            {"id": "1", "description": "Task", "status": "in_progress", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
        ]
        ctx = MockToolContext({"tasks": tasks})
        result = await plugin.after_tool_callback(
            tool=MockTool(),
            tool_args={},
            tool_context=ctx,
            result={"success": True, "data": "value"},
        )
        # Plugin should return None (don't modify tool result)
        assert result is None

    async def test_current_task_in_event_metadata(self, plugin):
        tasks = [
            {"id": "active-1", "description": "Working on this", "status": "in_progress", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
        ]
        ctx = MockToolContext({"tasks": tasks})
        await plugin.after_tool_callback(
            tool=MockTool(), tool_args={}, tool_context=ctx, result={"success": True},
        )
        events = plugin.drain_events()
        event = events[0]
        assert event.metadata.get("current_task_id") == "active-1"
        assert event.metadata.get("current_task_description") == "Working on this"

    async def test_max_events_cap(self):
        """Plugin respects max_events limit."""
        plugin = TaskProgressPlugin(max_events=3)
        tasks = [
            {"id": "1", "description": "Task", "status": "pending", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
        ]
        ctx = MockToolContext({"tasks": tasks})
        # Add 5 events
        for _ in range(5):
            await plugin.after_tool_callback(
                tool=MockTool(), tool_args={}, tool_context=ctx, result={"success": True},
            )
        # Only last 3 should remain (deque maxlen=3)
        events = plugin.drain_events()
        assert len(events) == 3
