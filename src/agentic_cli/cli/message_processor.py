"""Message processing for agentic CLI applications.

This module provides:
- MessageProcessor: Handles user message processing through workflow
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from agentic_cli.logging import Loggers, bind_context

if TYPE_CHECKING:
    from agentic_cli.cli.usage_tracker import UsageTracker
    from agentic_cli.config import BaseSettings
    from agentic_cli.cli.workflow_controller import WorkflowController
    from agentic_cli.workflow import WorkflowEvent
    from agentic_cli.workflow.events import UserInputRequest
    from thinking_prompt import ThinkingContext, ThinkingPromptSession


logger = Loggers.cli()


def _richify_task_display(plain: str) -> str:
    """Convert plain task checklist to rich ANSI-colored display.

    Task box uses full-color styling (not dim grey like thinking/tool boxes):
    - Completed: green ✓, dim text (done = de-emphasised)
    - In-progress: yellow ▸, white text (active = prominent)
    - Pending: white ○, white text (upcoming = visible)
    - Cancelled: red ✗, dim text (irrelevant = de-emphasised)
    - Section headers: bold white
    """
    from thinking_prompt import rich_to_ansi

    def _esc(text: str) -> str:
        """Escape Rich markup brackets in task text."""
        return text.replace("[", "\\[")

    lines = []
    for line in plain.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("[✓]"):
            lines.append(f"{indent}[green]✓[/green] [dim]{_esc(stripped[4:])}[/dim]")
        elif stripped.startswith("[▸]"):
            lines.append(
                f"{indent}[yellow]▸[/yellow] [white]{_esc(stripped[4:])}[/white]"
            )
        elif stripped.startswith("[ ]"):
            lines.append(
                f"{indent}[white]○[/white] [white]{_esc(stripped[4:])}[/white]"
            )
        elif stripped.startswith("[-]"):
            lines.append(
                f"{indent}[red]✗[/red] [dim]{_esc(stripped[4:])}[/dim]"
            )
        else:
            lines.append(f"[bold white]{_esc(line)}[/bold white]")
    return rich_to_ansi("\n".join(lines))


# === Event Processing State ===


@dataclass
class _EventProcessingState:
    """Mutable state shared across event handlers during process()."""

    usage_tracker: "UsageTracker | None" = field(default=None, repr=False)
    workflow_controller: "WorkflowController | None" = field(default=None, repr=False)
    status_line: str = "Processing..."
    thinking_started: bool = False
    # True while a HITL dialog is on screen: during that window there are no
    # active thinking boxes, so the Ctrl+C watcher must not mistake the absence
    # of boxes for a cancel request.
    in_hitl: bool = False
    thinking_content: list[str] = field(default_factory=list)
    response_content: list[str] = field(default_factory=list)
    # Prevents double-counting when LangGraph emits both CONTEXT_TRIMMED and LLM_USAGE
    _context_trimmed_this_invocation: bool = False

    def get_status(self) -> str:
        """Return the current status line for the events thinking box."""
        return self.status_line

    def reset_for_retry(self) -> None:
        """Reset state for retry after rate limit."""
        self.status_line = "Processing..."
        self.thinking_content.clear()
        self.response_content.clear()


# === Message Processor ===


class MessageProcessor:
    """Processes user messages through the workflow.

    Handles:
    - Event stream processing from workflow
    - Thinking box state management
    - Tool result display
    - User input prompting during workflow
    - Message history tracking

    Example:
        processor = MessageProcessor()

        await processor.process(
            message="Hello",
            workflow_controller=controller,
            ui=session,
            settings=settings,
        )

        # Access history
        for msg in processor.history.get_all():
            print(f"{msg.message_type}: {msg.content}")
    """

    def __init__(self) -> None:
        """Initialize the message processor."""
        self._last_task_progress: str | None = None
        self._task_box: "ThinkingContext | None" = None
        self._last_task_content: str | None = None

    def clear_task_state(self) -> None:
        """Clear task box state."""
        self._last_task_progress = None
        if self._task_box is not None:
            self._task_box.finish(add_to_history=False, echo_to_console=False)
            self._task_box = None
        self._last_task_content = None

    async def process(
        self,
        message: str,
        workflow_controller: "WorkflowController",
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        usage_tracker: "UsageTracker | None" = None,
    ) -> None:
        """Process a user message through the workflow.

        Args:
            message: User message to process
            workflow_controller: Controller managing workflow lifecycle
            ui: UI session for output
            settings: Application settings
            usage_tracker: Optional tracker for accumulating LLM token usage
        """
        # Wait for initialization if needed
        if not await workflow_controller.ensure_initialized(ui):
            ui.add_error(
                "Cannot process message - workflow not initialized. "
                "Please check your API keys (GOOGLE_API_KEY or ANTHROPIC_API_KEY)."
            )
            return

        bind_context(user_id=settings.default_user)
        logger.info("handling_message", message_length=len(message))

        def _source(workflow):
            return workflow.process(message=message, user_id=settings.default_user)

        await self._run_turn(_source, workflow_controller, ui, settings, usage_tracker)

    async def process_resume(
        self,
        record,
        workflow_controller: "WorkflowController",
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        usage_tracker: "UsageTracker | None" = None,
    ) -> None:
        """Resume the agent with a finished long-running job's result.

        Streams ``workflow.resume_with_job_result(record)`` through the exact
        same rendering path as a user turn (events box, tool results, token
        accounting, Ctrl+C). A no-op if the backend can't resume.

        Args:
            record: The terminal JobRecord to resume from.
            workflow_controller: Controller managing workflow lifecycle.
            ui: UI session for output.
            settings: Application settings.
            usage_tracker: Optional tracker for accumulating LLM token usage.
        """
        if not await workflow_controller.ensure_initialized(ui):
            return

        workflow = workflow_controller.workflow
        bind_context(user_id=settings.default_user)
        icon = "✓" if record.state.value == "succeeded" else "✗"

        # Resume only if the backend supports it AND the originating conversation
        # is still available (after a restart the in-memory ADK session is gone).
        # Otherwise surface a notice — the result stays reachable by job id.
        resumable = hasattr(
            workflow, "resume_with_job_result"
        ) and await workflow.can_resume(record)
        if not resumable:
            logger.info(
                "job_resume_not_resumable",
                job_id=record.job_id,
                backend=getattr(workflow, "backend_type", "?"),
            )
            ui.add_message(
                "system",
                f"{icon} Background job '{record.name}' finished "
                f"({record.state.value}) while its conversation was unavailable "
                f"— fetch the result with /jobs {record.job_id}.",
            )
            return

        logger.info("resuming_job", job_id=record.job_id, state=record.state.value)
        ui.add_message(
            "system",
            f"↻ Background job '{record.name}' finished ({icon} {record.state.value}) "
            "— resuming.",
        )

        def _source(wf):
            return wf.resume_with_job_result(record)

        await self._run_turn(_source, workflow_controller, ui, settings, usage_tracker)

    async def _run_turn(
        self,
        source_factory,
        workflow_controller: "WorkflowController",
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        usage_tracker: "UsageTracker | None" = None,
    ) -> None:
        """Drive one turn from an event-source factory through the UI.

        Shared by ``process`` (user message) and ``process_resume`` (job
        result). ``source_factory(workflow)`` returns the WorkflowEvent async
        generator to consume; everything else (events box, HITL callback,
        Ctrl+C cancel, rate-limit retry, token accounting) is identical.
        """
        state = _EventProcessingState(
            usage_tracker=usage_tracker,
            workflow_controller=workflow_controller,
        )
        workflow = workflow_controller.workflow
        dispatch = self._get_event_dispatch()

        # Cold start: recreate task box from cached content if it was alive
        # last turn but got cleaned up between turns
        if self._task_box is None and self._last_task_progress is not None:
            self._task_box = ui.start_thinking(
                title="Tasks",
                order=100,
                content_format="ansi",
            )
            self._task_box.append(self._last_task_progress)

        # Events thinking box context — tracks the per-invocation events box.
        # This is a callback-driven box (no append/clear), only the callback
        # (state.get_status) drives its display.
        events_ctx: "ThinkingContext | None" = None

        # Set up direct callback so HITL tools can prompt the user without
        # deadlocking the workflow runner.
        async def _handle_input(request: "UserInputRequest") -> str:
            nonlocal events_ctx
            # Mark the HITL window before tearing down the events box so the
            # cancel watcher doesn't read "no active boxes" as a Ctrl+C.
            state.in_hitl = True
            try:
                if state.thinking_started and events_ctx is not None:
                    events_ctx.finish(add_to_history=False)
                    state.thinking_started = False

                response = await self._prompt_user_input(request, ui)
            finally:
                events_ctx = ui.start_thinking(
                    state.get_status, content_format="ansi"
                )
                state.thinking_started = True
                state.in_hitl = False
            return response

        workflow.set_input_callback(_handle_input)
        try:
            while True:
                try:
                    events_ctx = ui.start_thinking(
                        state.get_status, content_format="ansi"
                    )
                    state.thinking_started = True

                    # Consume the event stream in a cancellable task so Ctrl+C
                    # can abort an in-flight run. thinking_prompt's Ctrl+C
                    # binding finishes all thinking boxes (so ui.is_thinking
                    # flips False) but never cancels our coroutine, so we watch
                    # for that and cancel the task ourselves.
                    async def _consume() -> None:
                        async for event in source_factory(workflow):
                            handler = dispatch.get(event.type)
                            if handler is not None:
                                await handler(
                                    self, event, state, ui, settings, workflow
                                )

                    proc_task = asyncio.create_task(_consume())
                    if await self._watch_for_cancel(proc_task, ui, state):
                        # Ctrl+C already finished every active box; just drop
                        # our now-dead references so the next turn starts clean.
                        state.thinking_started = False
                        self._task_box = None
                        self._last_task_content = None
                        ui.add_warning("Cancelled.")
                        workflow_controller.update_status_bar(ui)
                        logger.info("message_cancelled_by_user")
                        break

                    # Finish events box only (don't add status to history)
                    if state.thinking_started and events_ctx is not None:
                        events_ctx.finish(add_to_history=False)

                    # Ensure final token counts are reflected in status bar
                    workflow_controller.update_status_bar(ui)

                    logger.debug("message_handled_successfully")
                    break  # Success — exit retry loop

                except Exception as e:
                    if state.thinking_started and events_ctx is not None:
                        events_ctx.finish(add_to_history=False)
                        state.thinking_started = False

                    # Check for 429 rate limit errors — prompt user to wait and retry
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
                            continue  # Retry the loop

                    # Non-429 or user chose cancel
                    ui.add_error(f"Workflow error: {e}")
                    break
        finally:
            workflow.clear_input_callback()
            # Cache plain-text task content for cold start on next turn
            # (not get_content() which returns already-richified ANSI)
            self._last_task_progress = (
                self._last_task_content if self._task_box else None
            )

    async def _watch_for_cancel(
        self,
        proc_task: "asyncio.Task[None]",
        ui: "ThinkingPromptSession",
        state: _EventProcessingState,
    ) -> bool:
        """Watch a running workflow task and cancel it on Ctrl+C.

        thinking_prompt's Ctrl+C binding finishes all active thinking boxes
        (so ``ui.is_thinking`` becomes False) but does not cancel the coroutine
        driving the workflow. We poll for that transition — ignoring the HITL
        window, when boxes are legitimately absent — and cancel the task when
        it happens.

        Args:
            proc_task: Task consuming the workflow event stream.
            ui: UI session (provides the public ``is_thinking`` box state).
            state: Shared processing state (``in_hitl`` guards the dialog window).

        Returns:
            True if the task was cancelled by the user. Otherwise False, after
            re-raising any exception the task raised.
        """
        while True:
            done, _ = await asyncio.wait({proc_task}, timeout=0.1)
            if proc_task in done:
                break
            if not state.in_hitl and not ui.is_thinking:
                proc_task.cancel()
                with suppress(asyncio.CancelledError):
                    await proc_task
                return True
        # Task completed on its own — surface its result/exception.
        proc_task.result()
        return False

    async def _prompt_user_input(
        self,
        request: "UserInputRequest",
        ui: "ThinkingPromptSession",
    ) -> str:
        """Prompt user for input requested by a tool.

        Args:
            request: The user input request from the tool.
            ui: UI session for prompting.

        Returns:
            User's response string.
        """
        from agentic_cli.workflow.events import InputType

        tool_name = request.tool_name or "Tool"

        if request.input_type == InputType.CHOICE and request.choices:
            result = await ui.dropdown_dialog(
                title=f"{tool_name} - Input Required",
                text=request.prompt,
                options=request.choices,
                default=request.default,
            )
            return result or (request.default or "")

        elif request.input_type == InputType.CONFIRM:
            result = await ui.yes_no_dialog(
                title=f"{tool_name} - Confirmation",
                text=request.prompt,
            )
            return "yes" if result else "no"

        else:
            result = await ui.input_dialog(
                title=f"{tool_name} - Input Required",
                text=request.prompt,
                default=request.default or "",
            )
            return result or ""

    # === Event handler methods ===

    async def _handle_text(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle TEXT events — stream response to console."""
        ui.add_response(event.content, markdown=True)
        state.response_content.append(event.content)

    async def _handle_thinking(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle THINKING events — update status, optionally display."""
        state.status_line = "Thinking..."
        if settings.verbose_thinking:
            ui.add_message("system", event.content)
        state.thinking_content.append(event.content)

    async def _handle_tool_call(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle TOOL_CALL events — update status line."""
        tool_name = event.metadata.get("tool_name", "unknown")
        state.status_line = f"Calling: {tool_name}"

    async def _handle_tool_result(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle TOOL_RESULT events — display result summary."""
        tool_name = event.metadata.get("tool_name", "unknown")
        success = event.metadata.get("success", True)
        duration = event.metadata.get("duration_ms")
        icon = "+" if success else "x"
        duration_str = f" ({duration}ms)" if duration else ""
        lines = event.content.split("\n")
        first_line = lines[0]
        state.status_line = f"{icon} {tool_name}: {first_line}{duration_str}"
        style = "green" if success else "red"
        display = f"[{style}]{icon}[/{style}] {tool_name}: {first_line}{duration_str}"
        if len(lines) > 1:
            display += "\n" + "\n".join(lines[1:])
        ui.add_rich(display)

    async def _handle_code_execution(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle CODE_EXECUTION events — update status with result preview."""
        result_preview = (
            event.content[:40] + "..."
            if len(event.content) > 40
            else event.content
        )
        state.status_line = f"Result: {result_preview}"

    async def _handle_executable_code(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle EXECUTABLE_CODE events — update status with language."""
        lang = event.metadata.get("language", "python")
        state.status_line = f"Running {lang} code..."

    async def _handle_file_data(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle FILE_DATA events — update status with filename."""
        state.status_line = f"File: {event.content}"

    async def _handle_task_progress(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle TASK_PROGRESS events — manage a separate task thinking box."""
        progress = event.metadata.get("progress", {})
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)

        # Destroy box when tasks are cleared or all complete
        if total == 0 or completed >= total:
            if self._task_box is not None:
                self._task_box.finish(add_to_history=False, echo_to_console=False)
                self._task_box = None
                self._last_task_content = None
            # Update events box status line
            current_task = event.metadata.get("current_task_description")
            if current_task:
                state.status_line = f"Working on: {current_task}"
            return

        # Create task box on first progress event with work remaining
        if self._task_box is None:
            self._task_box = ui.start_thinking(
                title=f"Tasks: {completed}/{total}",
                order=100,
                content_format="ansi",
            )
        else:
            # Update title on existing box
            self._task_box.set_title(f"Tasks: {completed}/{total}")

        # Deduplicate: skip content update if unchanged
        if event.content and event.content != self._last_task_content:
            self._last_task_content = event.content
            self._task_box.clear()
            self._task_box.append(_richify_task_display(event.content))

        # Update events box status line
        current_task = event.metadata.get("current_task_description")
        if current_task:
            state.status_line = f"Working on: {current_task}"

    async def _handle_context_trimmed(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle CONTEXT_TRIMMED events — increment counter and warn user."""
        if state.usage_tracker is not None:
            state.usage_tracker.context_trimmed_count += 1
        state._context_trimmed_this_invocation = True

        removed = event.metadata.get("messages_removed")
        source = event.metadata.get("source", "unknown")
        if removed is not None:
            ui.add_warning(
                f"Context trimmed: {removed} messages removed ({source})"
            )
        else:
            ui.add_warning(f"Context window trimmed ({source})")

    async def _handle_llm_usage(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle LLM_USAGE events — accumulate token counts and refresh status bar."""
        if state.usage_tracker is not None:
            tracker = state.usage_tracker
            prev_prompt = tracker.last_prompt_tokens
            heuristic_trimmed = tracker.record(
                event.metadata,
                context_trimmed_already=state._context_trimmed_this_invocation,
            )

            if heuristic_trimmed:
                from agentic_cli.cli.usage_tracker import format_tokens

                ui.add_warning(
                    f"Context window trimmed: {format_tokens(prev_prompt)}"
                    f" → {format_tokens(tracker.last_prompt_tokens)} tokens"
                    " (token_drop_heuristic)"
                )

            # Reset per-invocation flag for next LLM call
            state._context_trimmed_this_invocation = False

        if state.workflow_controller is not None:
            state.workflow_controller.update_status_bar(ui)

    # === Dispatch table ===

    _EVENT_DISPATCH: ClassVar[dict | None] = None

    @classmethod
    def _get_event_dispatch(cls) -> dict:
        """Get or build the EventType → handler dispatch table."""
        if cls._EVENT_DISPATCH is None:
            from agentic_cli.workflow import EventType

            cls._EVENT_DISPATCH = {
                EventType.TEXT: cls._handle_text,
                EventType.THINKING: cls._handle_thinking,
                EventType.TOOL_CALL: cls._handle_tool_call,
                EventType.TOOL_RESULT: cls._handle_tool_result,
                EventType.CODE_EXECUTION: cls._handle_code_execution,
                EventType.EXECUTABLE_CODE: cls._handle_executable_code,
                EventType.FILE_DATA: cls._handle_file_data,
                EventType.TASK_PROGRESS: cls._handle_task_progress,
                EventType.CONTEXT_TRIMMED: cls._handle_context_trimmed,
                EventType.LLM_USAGE: cls._handle_llm_usage,
            }
        return cls._EVENT_DISPATCH
