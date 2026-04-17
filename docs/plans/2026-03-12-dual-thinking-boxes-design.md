# Dual Thinking Boxes: Persistent Task Progress + Ephemeral LLM Events

**Date:** 2026-03-12
**Status:** Approved

## Problem

The current thinking box serves dual purposes — showing both task progress and LLM event status (tool calls, thinking). It disappears when the user needs to respond (HITL prompts, turn boundaries), losing task progress visibility. Users lose context about which tasks are pending/complete between turns.

## Solution

Split into two independent thinking boxes using thinking-prompt's multi-box API:

| Box | Purpose | Order | Lifecycle |
|-----|---------|-------|-----------|
| **Task Progress** | Persistent task checklist | `100` (near prompt) | First task → all tasks complete |
| **Events** | Tool calls, thinking status | `0` (above tasks) | Per processing cycle |

## Design

### Task Progress Box

- **Created** when first `TASK_PROGRESS` event arrives with `total > 0` and `self._task_box is None`
- **Updated** on each `TASK_PROGRESS` event: title set to `"Tasks: M/N"`, content replaced with richified checklist
- **Persists** across `process()` calls — NOT destroyed at end of turn
- **Destroyed** when `TASK_PROGRESS` event shows `total == 0` or `completed == total`, or on `clear_history()`
- **Cold start**: at start of `process()`, if `_task_box is None` and `_last_task_progress` cache exists, recreate the box
- **HITL flow**: task box stays visible when events box is destroyed for user input dialogs
- **API**: `ui.start_thinking(title="Tasks: 0/N", order=100, content_format="ansi")` → `ThinkingContext`

### Events Box

- Same behavior as the current single thinking box, minus the task progress content
- `ui.start_thinking(state.get_status, content_format="ansi")` — uses default order=0
- Created at start of `process()`, destroyed at end
- Destroyed/recreated around HITL prompts (same as today)
- `get_status()` returns only `status_line` (no task progress composite)

### Changes to `MessageProcessor`

1. **New instance attribute**: `_task_box: ThinkingContext | None = None`
2. **`_EventProcessingState`**: remove `task_progress_display` and `_last_task_display_content` fields; `get_status()` returns only `status_line`
3. **`_handle_task_progress()`**: creates/updates/destroys `self._task_box` directly on the MessageProcessor instance
4. **`process()`**: cold-start check at beginning; events box lifecycle unchanged; task box NOT touched by `finish_thinking`
5. **`clear_history()`**: also destroys `_task_box` if active
6. **Events box**: use `ctx.finish()` (per-box finish) instead of `session.finish_thinking()`

### Changes to `_handle_task_progress`

```python
async def _handle_task_progress(self, event, state, ui, settings, workflow):
    progress = event.metadata.get("progress", {})
    total = progress.get("total", 0)
    completed = progress.get("completed", 0)

    if total > 0 and completed < total:
        if self._task_box is None:
            self._task_box = ui.start_thinking(
                title=f"Tasks: {completed}/{total}",
                order=100,
                content_format="ansi",
            )
        else:
            self._task_box.set_title(f"Tasks: {completed}/{total}")

        if event.content:
            self._task_box.clear()
            self._task_box.append(_richify_task_display(event.content))
    else:
        # All tasks done or no tasks — remove task box
        if self._task_box is not None:
            self._task_box.finish(add_to_history=False, echo_to_console=False)
            self._task_box = None

    # Update events box status line
    current_task = event.metadata.get("current_task_description")
    if current_task:
        state.status_line = f"Working on: {current_task}"
```

### Changes to `process()` Flow

```python
async def process(self, message, workflow_controller, ui, settings, usage_tracker=None):
    # ... initialization ...

    state = _EventProcessingState(...)

    # Cold start: recreate task box if tasks were active from previous turn
    if self._task_box is None and self._last_task_progress is not None:
        self._task_box = ui.start_thinking(
            title="Tasks",
            order=100,
            content_format="ansi",
        )
        self._task_box.append(self._last_task_progress)

    # Events box: created per-cycle
    events_ctx = ui.start_thinking(state.get_status, content_format="ansi")

    async def _handle_input(request):
        events_ctx.finish(add_to_history=False)  # Only events box
        response = await self._prompt_user_input(request, ui)
        nonlocal events_ctx
        events_ctx = ui.start_thinking(state.get_status, content_format="ansi")
        return response

    # ... event processing loop (unchanged) ...

    events_ctx.finish(add_to_history=False)  # Only events box

    # Persist task progress for cold start
    self._last_task_progress = (
        self._task_box.get_content() if self._task_box else None
    )
```

### Dependency

Bump `thinking-prompt` to `>=0.2.6` (multi-box support, currently unreleased — release first).

### What Stays the Same

- `_richify_task_display()` — same formatting
- All event handlers except `_handle_task_progress` — unchanged
- `WorkflowEvent`, `EventType` — unchanged
- Task progress event generation in workflow managers — unchanged
- `MessageHistory` — unchanged

## Testing

- Unit test: task box created on first TASK_PROGRESS, destroyed when all complete
- Unit test: events box created/destroyed per process() cycle, task box persists
- Unit test: cold start recreates task box from cached content
- Unit test: HITL flow keeps task box, only events box cycles
- Unit test: clear_history() destroys task box
- Integration test: visual verification with demo app
