# Dual Thinking Boxes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the single thinking box into a persistent task progress box (order=100, near prompt) and an ephemeral events box (order=0) using thinking-prompt's multi-box API.

**Architecture:** MessageProcessor gains a `_task_box: ThinkingContext | None` instance attribute. The events box uses per-box `ctx.finish()` instead of session-level `finish_thinking()`. Task progress display is moved from `_EventProcessingState` composite into its own thinking box.

**Tech Stack:** thinking-prompt (multi-box API: `start_thinking(order=...)`, `ThinkingContext.finish()`)

**Design doc:** `docs/plans/2026-03-12-dual-thinking-boxes-design.md`

---

### Task 1: Bump thinking-prompt dependency

**Files:**
- Modify: `pyproject.toml:24`

**Step 1: Update dependency version**

In `pyproject.toml`, change line 24:
```python
# FROM:
    "thinking-prompt>=0.2.5",
# TO:
    "thinking-prompt>=0.2.6",
```

Note: thinking-prompt v0.2.6 must be released first (multi-box feature is merged but untagged). If not yet released, use `>=0.2.5` and ensure the local dev install has the multi-box feature.

**Step 2: Reinstall**

Run: `conda run -n agenticcli pip install -e .`
Expected: success, thinking-prompt with multi-box API available

**Step 3: Verify multi-box API available**

Run: `conda run -n agenticcli python -c "from thinking_prompt import ThinkingPromptSession; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: bump thinking-prompt dependency for multi-box support"
```

---

### Task 2: Simplify _EventProcessingState — remove task progress fields

**Files:**
- Modify: `src/agentic_cli/cli/message_processor.py:116-145`
- Test: `tests/test_dual_thinking_boxes.py` (new)

**Step 1: Write failing tests for simplified state**

Create `tests/test_dual_thinking_boxes.py`:

```python
"""Tests for dual thinking boxes — persistent task progress + ephemeral events."""

import pytest
from unittest.mock import MagicMock


class TestEventProcessingStateSimplified:
    """Verify _EventProcessingState no longer composites task progress."""

    def test_get_status_returns_only_status_line(self):
        from agentic_cli.cli.message_processor import _EventProcessingState

        state = _EventProcessingState()
        state.status_line = "Calling: search"
        assert state.get_status() == "Calling: search"

    def test_get_status_no_task_progress_field(self):
        from agentic_cli.cli.message_processor import _EventProcessingState

        state = _EventProcessingState()
        assert not hasattr(state, "task_progress_display")

    def test_reset_for_retry_no_task_fields(self):
        from agentic_cli.cli.message_processor import _EventProcessingState

        state = _EventProcessingState()
        state.status_line = "Working on: task 1"
        state.reset_for_retry()
        assert state.status_line == "Processing..."
        assert not hasattr(state, "_last_task_display_content")
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n agenticcli python -m pytest tests/test_dual_thinking_boxes.py::TestEventProcessingStateSimplified -v`
Expected: FAIL — `task_progress_display` still exists, `get_status()` still composites

**Step 3: Modify _EventProcessingState**

In `src/agentic_cli/cli/message_processor.py`, replace lines 116-145:

```python
@dataclass
class _EventProcessingState:
    """Mutable state shared across event handlers during process()."""

    usage_tracker: "UsageTracker | None" = field(default=None, repr=False)
    workflow_controller: "WorkflowController | None" = field(default=None, repr=False)
    status_line: str = "Processing..."
    thinking_started: bool = False
    thinking_content: list[str] = field(default_factory=list)
    response_content: list[str] = field(default_factory=list)
    # Prevents double-counting when LangGraph emits both CONTEXT_TRIMMED and LLM_USAGE
    _context_trimmed_this_invocation: bool = False

    def get_status(self) -> str:
        """Build status display for the events thinking box."""
        return self.status_line

    def reset_for_retry(self) -> None:
        """Reset state for retry after rate limit."""
        self.status_line = "Processing..."
        self.thinking_content.clear()
        self.response_content.clear()
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n agenticcli python -m pytest tests/test_dual_thinking_boxes.py::TestEventProcessingStateSimplified -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/agentic_cli/cli/message_processor.py tests/test_dual_thinking_boxes.py
git commit -m "refactor: remove task progress fields from _EventProcessingState"
```

---

### Task 3: Add _task_box and _last_task_content to MessageProcessor

**Files:**
- Modify: `src/agentic_cli/cli/message_processor.py:175-188`
- Test: `tests/test_dual_thinking_boxes.py`

**Step 1: Write failing tests**

Add to `tests/test_dual_thinking_boxes.py`:

```python
class TestMessageProcessorTaskBox:
    """Verify MessageProcessor has task_box instance attribute."""

    def test_task_box_initially_none(self):
        from agentic_cli.cli.message_processor import MessageProcessor

        processor = MessageProcessor()
        assert processor._task_box is None

    def test_last_task_content_initially_none(self):
        from agentic_cli.cli.message_processor import MessageProcessor

        processor = MessageProcessor()
        assert processor._last_task_content is None

    def test_clear_history_resets_task_state(self):
        from agentic_cli.cli.message_processor import MessageProcessor

        processor = MessageProcessor()
        processor._task_box = MagicMock()
        processor._last_task_content = "some content"
        processor._last_task_progress = "cached"

        processor.clear_history()

        assert processor._task_box is None
        assert processor._last_task_content is None
        assert processor._last_task_progress is None
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n agenticcli python -m pytest tests/test_dual_thinking_boxes.py::TestMessageProcessorTaskBox -v`
Expected: FAIL — `_task_box` and `_last_task_content` not defined

**Step 3: Update MessageProcessor.__init__ and clear_history**

In `src/agentic_cli/cli/message_processor.py`, modify `__init__` (lines 175-178):

```python
    def __init__(self) -> None:
        """Initialize the message processor."""
        self._message_history = MessageHistory()
        self._last_task_progress: str | None = None
        self._task_box: "ThinkingContext | None" = None
        self._last_task_content: str | None = None
```

Add `ThinkingContext` to the TYPE_CHECKING imports (line 24):

```python
    from thinking_prompt import ThinkingPromptSession, ThinkingContext
```

Modify `clear_history` (lines 185-188):

```python
    def clear_history(self) -> None:
        """Clear message history."""
        self._message_history.clear()
        self._last_task_progress = None
        if self._task_box is not None:
            self._task_box.finish(add_to_history=False, echo_to_console=False)
            self._task_box = None
        self._last_task_content = None
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n agenticcli python -m pytest tests/test_dual_thinking_boxes.py::TestMessageProcessorTaskBox -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/agentic_cli/cli/message_processor.py tests/test_dual_thinking_boxes.py
git commit -m "feat: add _task_box and _last_task_content to MessageProcessor"
```

---

### Task 4: Rewrite _handle_task_progress to manage task box

**Files:**
- Modify: `src/agentic_cli/cli/message_processor.py:462-492`
- Test: `tests/test_dual_thinking_boxes.py`

**Step 1: Write failing tests**

Add to `tests/test_dual_thinking_boxes.py`:

```python
class TestHandleTaskProgress:
    """Verify _handle_task_progress creates/updates/destroys task box."""

    @pytest.fixture
    def setup(self):
        from agentic_cli.cli.message_processor import (
            MessageProcessor,
            _EventProcessingState,
        )
        from agentic_cli.workflow.events import WorkflowEvent, EventType

        processor = MessageProcessor()
        state = _EventProcessingState()
        ui = MagicMock()
        # start_thinking returns a ThinkingContext mock
        task_ctx = MagicMock()
        ui.start_thinking.return_value = task_ctx
        settings = MagicMock()
        workflow = MagicMock()
        return processor, state, ui, settings, workflow, task_ctx, WorkflowEvent

    async def test_creates_task_box_on_first_progress(self, setup):
        processor, state, ui, settings, workflow, task_ctx, WorkflowEvent = setup
        event = WorkflowEvent(
            type="task_progress",
            content="[▸] Task 1\n[ ] Task 2",
            metadata={"progress": {"total": 2, "completed": 0}},
        )
        await processor._handle_task_progress(event, state, ui, settings, workflow)

        ui.start_thinking.assert_called_once()
        call_kwargs = ui.start_thinking.call_args
        assert call_kwargs.kwargs["order"] == 100
        assert "0/2" in call_kwargs.kwargs["title"]
        assert processor._task_box is task_ctx

    async def test_updates_existing_task_box(self, setup):
        processor, state, ui, settings, workflow, task_ctx, WorkflowEvent = setup
        # Simulate existing task box
        existing_ctx = MagicMock()
        processor._task_box = existing_ctx

        event = WorkflowEvent(
            type="task_progress",
            content="[✓] Task 1\n[▸] Task 2",
            metadata={"progress": {"total": 2, "completed": 1}},
        )
        await processor._handle_task_progress(event, state, ui, settings, workflow)

        # Should NOT create a new box
        ui.start_thinking.assert_not_called()
        # Should update title and content
        existing_ctx.set_title.assert_called_once_with("Tasks: 1/2")
        existing_ctx.clear.assert_called_once()
        existing_ctx.append.assert_called_once()

    async def test_destroys_task_box_when_all_complete(self, setup):
        processor, state, ui, settings, workflow, task_ctx, WorkflowEvent = setup
        existing_ctx = MagicMock()
        processor._task_box = existing_ctx

        event = WorkflowEvent(
            type="task_progress",
            content="[✓] Task 1\n[✓] Task 2",
            metadata={"progress": {"total": 2, "completed": 2}},
        )
        await processor._handle_task_progress(event, state, ui, settings, workflow)

        existing_ctx.finish.assert_called_once_with(
            add_to_history=False, echo_to_console=False
        )
        assert processor._task_box is None

    async def test_destroys_task_box_when_total_zero(self, setup):
        processor, state, ui, settings, workflow, task_ctx, WorkflowEvent = setup
        existing_ctx = MagicMock()
        processor._task_box = existing_ctx

        event = WorkflowEvent(
            type="task_progress",
            content="",
            metadata={"progress": {"total": 0}},
        )
        await processor._handle_task_progress(event, state, ui, settings, workflow)

        existing_ctx.finish.assert_called_once()
        assert processor._task_box is None

    async def test_deduplicates_unchanged_content(self, setup):
        processor, state, ui, settings, workflow, task_ctx, WorkflowEvent = setup
        processor._task_box = MagicMock()
        processor._last_task_content = "[▸] Task 1"

        event = WorkflowEvent(
            type="task_progress",
            content="[▸] Task 1",
            metadata={"progress": {"total": 1, "completed": 0}},
        )
        await processor._handle_task_progress(event, state, ui, settings, workflow)

        # Should not update content (deduped)
        processor._task_box.clear.assert_not_called()

    async def test_updates_events_status_line(self, setup):
        processor, state, ui, settings, workflow, task_ctx, WorkflowEvent = setup
        event = WorkflowEvent(
            type="task_progress",
            content="[▸] Do something",
            metadata={
                "progress": {"total": 1, "completed": 0},
                "current_task_description": "Do something",
            },
        )
        await processor._handle_task_progress(event, state, ui, settings, workflow)
        assert state.status_line == "Working on: Do something"

    async def test_no_task_box_when_no_progress(self, setup):
        """No task box created when progress metadata is missing."""
        processor, state, ui, settings, workflow, task_ctx, WorkflowEvent = setup
        event = WorkflowEvent(
            type="task_progress",
            content="",
            metadata={},
        )
        await processor._handle_task_progress(event, state, ui, settings, workflow)
        assert processor._task_box is None
        ui.start_thinking.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n agenticcli python -m pytest tests/test_dual_thinking_boxes.py::TestHandleTaskProgress -v`
Expected: FAIL — current handler writes to `state.task_progress_display` not `self._task_box`

**Step 3: Rewrite _handle_task_progress**

Replace the handler at lines 462-492 in `src/agentic_cli/cli/message_processor.py`:

```python
    async def _handle_task_progress(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle TASK_PROGRESS events — manage persistent task box."""
        progress = event.metadata.get("progress", {})
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)

        if total > 0 and completed < total:
            # Create task box if it doesn't exist yet
            if self._task_box is None:
                self._task_box = ui.start_thinking(
                    title=f"Tasks: {completed}/{total}",
                    order=100,
                    content_format="ansi",
                )
            else:
                self._task_box.set_title(f"Tasks: {completed}/{total}")

            # Update content (skip if unchanged)
            if event.content and event.content != self._last_task_content:
                self._last_task_content = event.content
                self._task_box.clear()
                self._task_box.append(_richify_task_display(event.content))
        else:
            # All tasks done or no tasks — remove task box
            if self._task_box is not None:
                self._task_box.finish(add_to_history=False, echo_to_console=False)
                self._task_box = None
            self._last_task_content = None

        # Update events box status line
        current_task = event.metadata.get("current_task_description")
        if current_task:
            state.status_line = f"Working on: {current_task}"
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n agenticcli python -m pytest tests/test_dual_thinking_boxes.py::TestHandleTaskProgress -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add src/agentic_cli/cli/message_processor.py tests/test_dual_thinking_boxes.py
git commit -m "feat: rewrite _handle_task_progress to manage persistent task box"
```

---

### Task 5: Refactor process() to use per-box finish and cold start

**Files:**
- Modify: `src/agentic_cli/cli/message_processor.py:190-316`
- Test: `tests/test_dual_thinking_boxes.py`

**Step 1: Write failing tests**

Add to `tests/test_dual_thinking_boxes.py`:

```python
class TestProcessDualBoxes:
    """Verify process() manages events box independently from task box."""

    @pytest.fixture
    def setup(self):
        from agentic_cli.cli.message_processor import MessageProcessor

        processor = MessageProcessor()
        ui = MagicMock()
        events_ctx = MagicMock()
        ui.start_thinking.return_value = events_ctx
        settings = MagicMock()
        settings.log_activity = False
        settings.default_user = "test"
        workflow_ctrl = MagicMock()
        workflow_ctrl.ensure_initialized = AsyncMock(return_value=True)
        workflow_ctrl.workflow.process = AsyncMock(return_value=iter([]))
        workflow_ctrl.workflow.set_input_callback = MagicMock()
        workflow_ctrl.workflow.clear_input_callback = MagicMock()
        return processor, ui, settings, workflow_ctrl, events_ctx

    async def test_events_box_finished_per_cycle(self, setup):
        """Events box is finished at end of process(), task box is not."""
        processor, ui, settings, workflow_ctrl, events_ctx = setup
        # Simulate an active task box
        task_ctx = MagicMock()
        processor._task_box = task_ctx

        await processor.process("hello", workflow_ctrl, ui, settings)

        # Events box was finished
        events_ctx.finish.assert_called()
        # Task box was NOT finished
        task_ctx.finish.assert_not_called()

    async def test_cold_start_recreates_task_box(self, setup):
        """If _last_task_progress cached, task box is recreated on cold start."""
        processor, ui, settings, workflow_ctrl, events_ctx = setup
        processor._last_task_progress = "[▸] Cached task"

        await processor.process("hello", workflow_ctrl, ui, settings)

        # start_thinking called twice: once for task box cold start, once for events box
        assert ui.start_thinking.call_count == 2
        # First call is task box with order=100
        first_call = ui.start_thinking.call_args_list[0]
        assert first_call.kwargs.get("order") == 100

    async def test_no_cold_start_when_task_box_active(self, setup):
        """If task box already active, no cold start recreation."""
        processor, ui, settings, workflow_ctrl, events_ctx = setup
        processor._task_box = MagicMock()  # Already active
        processor._last_task_progress = "[▸] Cached task"

        await processor.process("hello", workflow_ctrl, ui, settings)

        # Only events box created (1 call), no extra for task box
        assert ui.start_thinking.call_count == 1

    async def test_task_progress_cached_on_exit(self, setup):
        """_last_task_progress is set from task box content on process() exit."""
        processor, ui, settings, workflow_ctrl, events_ctx = setup
        task_ctx = MagicMock()
        task_ctx.get_content.return_value = "[✓] Done task"
        processor._task_box = task_ctx

        await processor.process("hello", workflow_ctrl, ui, settings)

        assert processor._last_task_progress == "[✓] Done task"

    async def test_no_session_finish_thinking_called(self, setup):
        """Session-level finish_thinking() should NOT be called (deprecated)."""
        processor, ui, settings, workflow_ctrl, events_ctx = setup

        await processor.process("hello", workflow_ctrl, ui, settings)

        ui.finish_thinking.assert_not_called()
```

Also add `from unittest.mock import AsyncMock` to the imports at the top of the test file.

**Step 2: Run tests to verify they fail**

Run: `conda run -n agenticcli python -m pytest tests/test_dual_thinking_boxes.py::TestProcessDualBoxes -v`
Expected: FAIL — process() still uses `ui.finish_thinking()` and doesn't do cold start

**Step 3: Rewrite process()**

Replace the `process` method in `src/agentic_cli/cli/message_processor.py` (lines 190-316):

```python
    async def process(
        self,
        message: str,
        workflow_controller: "WorkflowController",
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        usage_tracker: "UsageTracker | None" = None,
    ) -> None:
        """Process a user message through the workflow."""
        # Wait for initialization if needed
        if not await workflow_controller.ensure_initialized(ui):
            ui.add_error(
                "Cannot process message - workflow not initialized. "
                "Please check your API keys (GOOGLE_API_KEY or ANTHROPIC_API_KEY)."
            )
            return

        from agentic_cli.workflow import WorkflowEvent

        bind_context(user_id=settings.default_user)
        logger.info("handling_message", message_length=len(message))

        if settings.log_activity:
            self._message_history.add(message, MessageType.USER)

        state = _EventProcessingState(
            usage_tracker=usage_tracker,
            workflow_controller=workflow_controller,
        )
        workflow = workflow_controller.workflow
        dispatch = self._get_event_dispatch()

        # Cold start: recreate task box if tasks were active from previous turn
        if self._task_box is None and self._last_task_progress is not None:
            self._task_box = ui.start_thinking(
                title="Tasks",
                order=100,
                content_format="ansi",
            )
            self._task_box.append(self._last_task_progress)

        # Events box: created per processing cycle
        events_ctx = ui.start_thinking(state.get_status, content_format="ansi")
        state.thinking_started = True

        async def _handle_input(request: "UserInputRequest") -> str:
            nonlocal events_ctx
            if state.thinking_started:
                events_ctx.finish(add_to_history=False)
                state.thinking_started = False

            response = await self._prompt_user_input(request, ui)

            events_ctx = ui.start_thinking(state.get_status, content_format="ansi")
            state.thinking_started = True
            return response

        workflow.set_input_callback(_handle_input)
        try:
            while True:
                try:
                    if not state.thinking_started:
                        events_ctx = ui.start_thinking(
                            state.get_status, content_format="ansi"
                        )
                        state.thinking_started = True

                    async for event in workflow.process(
                        message=message,
                        user_id=settings.default_user,
                    ):
                        handler = dispatch.get(event.type)
                        if handler is not None:
                            await handler(self, event, state, ui, settings, workflow)

                    # Finish events box only (task box persists)
                    if state.thinking_started:
                        events_ctx.finish(add_to_history=False)
                        state.thinking_started = False

                    workflow_controller.update_status_bar(ui)

                    if settings.log_activity:
                        if state.thinking_content:
                            self._message_history.add(
                                "".join(state.thinking_content),
                                MessageType.THINKING,
                            )
                        if state.response_content:
                            self._message_history.add(
                                "".join(state.response_content),
                                MessageType.ASSISTANT,
                            )

                    logger.debug("message_handled_successfully")
                    break

                except Exception as e:
                    if state.thinking_started:
                        events_ctx.finish(add_to_history=False)
                        state.thinking_started = False

                    from agentic_cli.workflow.retry import (
                        is_rate_limit_error,
                        parse_retry_delay,
                    )

                    if is_rate_limit_error(e):
                        delay = parse_retry_delay(e) or 60.0
                        retry = await ui.yes_no_dialog(
                            title="Rate Limited",
                            text=f"API rate limit reached. Retry in {delay:.0f}s?",
                        )
                        if retry:
                            ui.add_warning(f"Waiting {delay:.0f}s before retrying...")
                            await asyncio.sleep(delay)
                            state.reset_for_retry()
                            continue

                    ui.add_error(f"Workflow error: {e}")
                    if settings.log_activity:
                        self._message_history.add(str(e), MessageType.ERROR)
                    break
        finally:
            workflow.clear_input_callback()
            # Cache task box content for cold start on next turn
            self._last_task_progress = (
                self._task_box.get_content() if self._task_box else None
            )
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n agenticcli python -m pytest tests/test_dual_thinking_boxes.py::TestProcessDualBoxes -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/agentic_cli/cli/message_processor.py tests/test_dual_thinking_boxes.py
git commit -m "feat: refactor process() for dual thinking boxes with cold start"
```

---

### Task 6: Fix integration tests and existing tests

**Files:**
- Modify: `tests/integration/test_adk_integration.py` (lines where `ui.finish_thinking` is mocked)
- Modify: `tests/test_context_trimming.py` (if _EventProcessingState changes break it)

**Step 1: Run the full test suite to find breakages**

Run: `conda run -n agenticcli python -m pytest tests/ -v --tb=short 2>&1 | head -100`
Expected: Some failures from removed `task_progress_display` field and `finish_thinking` calls

**Step 2: Fix test_context_trimming.py**

The `_EventProcessingState` fixture in `TestHandleContextTrimmed` and `TestADKHeuristicFallback` should still work since they don't use `task_progress_display`. Verify no breakage.

**Step 3: Fix integration tests**

In `tests/integration/test_adk_integration.py`, the mock UI sets up `ui.start_thinking = MagicMock()` and `ui.finish_thinking = MagicMock()`. With the new code:

- `ui.start_thinking` now returns a `ThinkingContext` mock that needs a `finish()` method
- `ui.finish_thinking` should NOT be called anymore

Update the mock UI setup in each test that uses it:

```python
ui = MagicMock()
# start_thinking returns a context with finish()
ctx_mock = MagicMock()
ui.start_thinking.return_value = ctx_mock
```

**Step 4: Run full test suite**

Run: `conda run -n agenticcli python -m pytest tests/ -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add tests/
git commit -m "fix: update tests for dual thinking box API changes"
```

---

### Task 7: Update unreleased changelog

**Files:**
- Modify: `changes/unreleased.md`

**Step 1: Update changelog**

```markdown
# Unreleased

## Added
- Persistent task progress thinking box that stays visible across turns while tasks are active
- Ephemeral LLM events thinking box for tool calls and thinking status

## Changed
- Task progress display moved from composite status line to dedicated thinking box (order=100)
- Events box uses per-box `ctx.finish()` instead of session-level `finish_thinking()`
- Bumped thinking-prompt dependency to >=0.2.6 for multi-box support

## Fixed

## Removed
- `task_progress_display` and `_last_task_display_content` fields from `_EventProcessingState`
```

**Step 2: Commit**

```bash
git add changes/unreleased.md
git commit -m "docs: update unreleased changelog for dual thinking boxes"
```

---

### Task 8: End-to-end manual verification

**Step 1: Run the full test suite one final time**

Run: `conda run -n agenticcli python -m pytest tests/ -v`
Expected: All tests pass, no regressions

**Step 2: Verify with a consumer app (if available)**

If there's a consumer app that uses `agentic_cli`, test interactively:
1. Start the app
2. Send a message that triggers task creation
3. Verify: task box appears near prompt with task checklist
4. Verify: events box appears above with tool call status
5. Verify: when processing ends, events box disappears, task box stays
6. Send another message
7. Verify: events box reappears, task box persists with current state
8. Wait for all tasks to complete
9. Verify: task box disappears
