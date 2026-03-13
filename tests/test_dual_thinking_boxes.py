"""Tests for dual thinking boxes (Tasks 2, 3, 4).

Covers:
- _EventProcessingState simplification (no task_progress_display fields)
- MessageProcessor._task_box and _last_task_content attributes
- _handle_task_progress rewrite (manages own ThinkingContext box)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_cli.cli.message_processor import (
    MessageProcessor,
    _EventProcessingState,
)
from agentic_cli.workflow.events import EventType, WorkflowEvent


# ---------------------------------------------------------------------------
# Task 2: _EventProcessingState simplification
# ---------------------------------------------------------------------------


class TestEventProcessingStateSimplified:
    """State should no longer carry task_progress_display fields."""

    def test_no_task_progress_display_field(self):
        """task_progress_display should not exist on the state."""
        state = _EventProcessingState()
        assert not hasattr(state, "task_progress_display")

    def test_no_last_task_display_content_field(self):
        """_last_task_display_content should not exist on the state."""
        state = _EventProcessingState()
        assert not hasattr(state, "_last_task_display_content")

    def test_get_status_returns_status_line_only(self):
        """get_status() should return only the status_line string."""
        state = _EventProcessingState()
        state.status_line = "Calling: web_search"
        assert state.get_status() == "Calling: web_search"

    def test_get_status_default(self):
        """get_status() returns default 'Processing...' status."""
        state = _EventProcessingState()
        assert state.get_status() == "Processing..."

    def test_reset_for_retry_does_not_clear_task_fields(self):
        """reset_for_retry() should not reference task display fields."""
        state = _EventProcessingState()
        state.status_line = "Something"
        state.thinking_content.append("thought")
        state.response_content.append("response")
        state.reset_for_retry()
        assert state.status_line == "Processing..."
        assert state.thinking_content == []
        assert state.response_content == []


# ---------------------------------------------------------------------------
# Task 3: MessageProcessor._task_box and _last_task_content
# ---------------------------------------------------------------------------


class TestMessageProcessorTaskAttributes:
    """MessageProcessor should have _task_box and _last_task_content."""

    def test_has_task_box_attribute(self):
        """MessageProcessor should have _task_box initialized to None."""
        processor = MessageProcessor()
        assert hasattr(processor, "_task_box")
        assert processor._task_box is None

    def test_has_last_task_content_attribute(self):
        """MessageProcessor should have _last_task_content initialized to None."""
        processor = MessageProcessor()
        assert hasattr(processor, "_last_task_content")
        assert processor._last_task_content is None

    def test_clear_history_resets_task_box_and_content(self):
        """clear_history() should set _task_box and _last_task_content to None."""
        processor = MessageProcessor()
        processor._last_task_content = "some content"
        processor.clear_history()
        assert processor._task_box is None
        assert processor._last_task_content is None

    def test_clear_history_finishes_active_task_box(self):
        """clear_history() should call finish() on an active task box."""
        processor = MessageProcessor()
        mock_box = MagicMock()
        processor._task_box = mock_box
        processor.clear_history()
        mock_box.finish.assert_called_once_with(
            add_to_history=False, echo_to_console=False
        )
        assert processor._task_box is None

    def test_clear_history_still_clears_message_history(self):
        """clear_history() should still clear message history."""
        processor = MessageProcessor()
        processor._message_history.add("hello", "user")
        processor.clear_history()
        assert len(processor._message_history) == 0


# ---------------------------------------------------------------------------
# Task 4: _handle_task_progress rewrite
# ---------------------------------------------------------------------------


def _make_task_progress_event(
    content: str,
    completed: int,
    total: int,
    current_task_description: str | None = None,
) -> WorkflowEvent:
    """Helper to build a TASK_PROGRESS event."""
    return WorkflowEvent.task_progress(
        display=content,
        progress={"completed": completed, "total": total, "pending": total - completed},
        current_task_description=current_task_description,
    )


class TestHandleTaskProgress:
    """Tests for the rewritten _handle_task_progress method."""

    @pytest.fixture
    def processor(self):
        return MessageProcessor()

    @pytest.fixture
    def state(self):
        return _EventProcessingState()

    @pytest.fixture
    def ui(self):
        mock_ui = MagicMock()
        mock_ctx = MagicMock()
        mock_ui.start_thinking.return_value = mock_ctx
        return mock_ui

    @pytest.fixture
    def settings(self):
        return MagicMock()

    @pytest.fixture
    def workflow(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_creates_box_on_first_progress(
        self, processor, state, ui, settings, workflow
    ):
        """Should create a task box via ui.start_thinking on first progress event."""
        event = _make_task_progress_event("[▸] Task A\n[ ] Task B", completed=0, total=2)

        await processor._handle_task_progress(event, state, ui, settings, workflow)

        ui.start_thinking.assert_called_once()
        call_kwargs = ui.start_thinking.call_args
        assert call_kwargs[1].get("order") == 100
        assert call_kwargs[1].get("content_format") == "ansi"
        assert "0/2" in call_kwargs[1].get("title", "")
        assert processor._task_box is not None

    @pytest.mark.asyncio
    async def test_updates_existing_box(
        self, processor, state, ui, settings, workflow
    ):
        """Should update title and content on subsequent progress events."""
        # First event: create box
        event1 = _make_task_progress_event("[▸] Task A\n[ ] Task B", completed=0, total=2)
        await processor._handle_task_progress(event1, state, ui, settings, workflow)
        task_box = processor._task_box
        assert task_box is not None

        # Second event: update
        event2 = _make_task_progress_event("[✓] Task A\n[▸] Task B", completed=1, total=2)
        await processor._handle_task_progress(event2, state, ui, settings, workflow)

        task_box.set_title.assert_called()
        # Check that the title was updated to 1/2
        last_title = task_box.set_title.call_args[0][0]
        assert "1/2" in last_title
        # Should have cleared and appended new content
        task_box.clear.assert_called()
        task_box.append.assert_called()

    @pytest.mark.asyncio
    async def test_destroys_box_when_all_complete(
        self, processor, state, ui, settings, workflow
    ):
        """Should finish and remove task box when completed >= total."""
        # Create box
        event1 = _make_task_progress_event("[▸] Task A", completed=0, total=1)
        await processor._handle_task_progress(event1, state, ui, settings, workflow)
        assert processor._task_box is not None

        # Complete
        event2 = _make_task_progress_event("[✓] Task A", completed=1, total=1)
        await processor._handle_task_progress(event2, state, ui, settings, workflow)

        # Task box should be finished and cleared
        assert processor._task_box is None

    @pytest.mark.asyncio
    async def test_destroys_box_when_total_is_zero(
        self, processor, state, ui, settings, workflow
    ):
        """Should finish and remove task box when total becomes 0."""
        # Create box
        event1 = _make_task_progress_event("[▸] Task A", completed=0, total=1)
        await processor._handle_task_progress(event1, state, ui, settings, workflow)
        assert processor._task_box is not None
        task_box = processor._task_box

        # Total becomes 0 (tasks cleared)
        event2 = _make_task_progress_event("", completed=0, total=0)
        await processor._handle_task_progress(event2, state, ui, settings, workflow)

        task_box.finish.assert_called_once_with(
            add_to_history=False, echo_to_console=False
        )
        assert processor._task_box is None

    @pytest.mark.asyncio
    async def test_deduplicates_unchanged_content(
        self, processor, state, ui, settings, workflow
    ):
        """Should skip content update when content is unchanged."""
        event = _make_task_progress_event("[▸] Task A", completed=0, total=1)
        await processor._handle_task_progress(event, state, ui, settings, workflow)
        task_box = processor._task_box

        # Reset call counts
        task_box.clear.reset_mock()
        task_box.append.reset_mock()

        # Same event again
        await processor._handle_task_progress(event, state, ui, settings, workflow)

        # Content methods should NOT be called again
        task_box.clear.assert_not_called()
        task_box.append.assert_not_called()

    @pytest.mark.asyncio
    async def test_updates_status_line_with_current_task(
        self, processor, state, ui, settings, workflow
    ):
        """Should set state.status_line from current_task_description."""
        event = _make_task_progress_event(
            "[▸] Implement feature",
            completed=0,
            total=2,
            current_task_description="Implement feature",
        )
        await processor._handle_task_progress(event, state, ui, settings, workflow)
        assert "Implement feature" in state.status_line

    @pytest.mark.asyncio
    async def test_no_box_created_when_total_is_zero(
        self, processor, state, ui, settings, workflow
    ):
        """Should not create a box when total is 0 from the start."""
        event = _make_task_progress_event("", completed=0, total=0)
        await processor._handle_task_progress(event, state, ui, settings, workflow)
        assert processor._task_box is None
        ui.start_thinking.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_box_created_when_already_all_complete(
        self, processor, state, ui, settings, workflow
    ):
        """Should not create a box when completed >= total from the start."""
        event = _make_task_progress_event("[✓] Done", completed=1, total=1)
        await processor._handle_task_progress(event, state, ui, settings, workflow)
        assert processor._task_box is None
        ui.start_thinking.assert_not_called()

    @pytest.mark.asyncio
    async def test_content_update_uses_richify(
        self, processor, state, ui, settings, workflow
    ):
        """Should call _richify_task_display on content before appending."""
        event = _make_task_progress_event("[✓] Done task\n[▸] Current", completed=1, total=2)

        with patch(
            "agentic_cli.cli.message_processor._richify_task_display",
            return_value="richified",
        ) as mock_richify:
            await processor._handle_task_progress(event, state, ui, settings, workflow)
            mock_richify.assert_called_once_with(event.content)
            task_box = processor._task_box
            task_box.append.assert_called_once_with("richified")


# ---------------------------------------------------------------------------
# Task 5: process() dual-box management
# ---------------------------------------------------------------------------


class TestProcessDualBoxes:
    """Verify process() manages events box independently from task box."""

    @pytest.fixture
    def setup(self):
        processor = MessageProcessor()
        ui = MagicMock()
        events_ctx = MagicMock()
        ui.start_thinking.return_value = events_ctx
        settings = MagicMock()
        settings.log_activity = False
        settings.default_user = "test"

        workflow_ctrl = MagicMock()
        workflow_ctrl.ensure_initialized = AsyncMock(return_value=True)

        # workflow.process must be an async generator
        async def empty_gen(*a, **kw):
            return
            yield  # make it an async generator

        workflow_ctrl.workflow.process = empty_gen
        workflow_ctrl.workflow.set_input_callback = MagicMock()
        workflow_ctrl.workflow.clear_input_callback = MagicMock()
        return processor, ui, settings, workflow_ctrl, events_ctx

    @pytest.mark.asyncio
    async def test_events_box_finished_task_box_untouched(self, setup):
        """Events box is finished at end, task box is NOT."""
        processor, ui, settings, wf_ctrl, events_ctx = setup
        task_ctx = MagicMock()
        processor._task_box = task_ctx

        await processor.process("hi", wf_ctrl, ui, settings)

        events_ctx.finish.assert_called()
        task_ctx.finish.assert_not_called()

    @pytest.mark.asyncio
    async def test_cold_start_recreates_task_box(self, setup):
        """Cached _last_task_progress triggers task box recreation."""
        processor, ui, settings, wf_ctrl, events_ctx = setup
        processor._last_task_progress = "[▸] Cached"

        await processor.process("hi", wf_ctrl, ui, settings)

        # Two calls: one for task box (order=100), one for events box
        assert ui.start_thinking.call_count == 2
        task_call = ui.start_thinking.call_args_list[0]
        assert task_call.kwargs.get("order") == 100

    @pytest.mark.asyncio
    async def test_cold_start_appends_cached_content_directly(self, setup):
        """Cold start appends cached content without re-richifying."""
        processor, ui, settings, wf_ctrl, events_ctx = setup
        task_box_mock = MagicMock()
        ui.start_thinking.return_value = task_box_mock
        processor._last_task_progress = "[▸] Already richified"

        await processor.process("hi", wf_ctrl, ui, settings)

        # The first start_thinking is the task box cold start
        task_box_mock.append.assert_called_once_with("[▸] Already richified")

    @pytest.mark.asyncio
    async def test_no_cold_start_when_task_box_active(self, setup):
        """No extra box created if task box already exists."""
        processor, ui, settings, wf_ctrl, events_ctx = setup
        processor._task_box = MagicMock()
        processor._last_task_progress = "[▸] Cached"

        await processor.process("hi", wf_ctrl, ui, settings)

        assert ui.start_thinking.call_count == 1  # Only events box

    @pytest.mark.asyncio
    async def test_task_progress_cached_on_exit(self, setup):
        """Plain-text _last_task_content cached to _last_task_progress on exit."""
        processor, ui, settings, wf_ctrl, events_ctx = setup
        task_ctx = MagicMock()
        processor._task_box = task_ctx
        processor._last_task_content = "[✓] Done task"

        await processor.process("hi", wf_ctrl, ui, settings)

        assert processor._last_task_progress == "[✓] Done task"

    @pytest.mark.asyncio
    async def test_no_session_finish_thinking(self, setup):
        """Session-level finish_thinking() must NOT be called."""
        processor, ui, settings, wf_ctrl, events_ctx = setup

        await processor.process("hi", wf_ctrl, ui, settings)

        ui.finish_thinking.assert_not_called()

    @pytest.mark.asyncio
    async def test_last_task_progress_none_when_no_task_box(self, setup):
        """_last_task_progress is None when no task box active."""
        processor, ui, settings, wf_ctrl, events_ctx = setup

        await processor.process("hi", wf_ctrl, ui, settings)

        assert processor._last_task_progress is None
