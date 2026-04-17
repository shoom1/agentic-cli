"""Message processing for agentic CLI applications.

This module provides:
- MessageProcessor: Handles user message processing through workflow
"""

from __future__ import annotations

import asyncio
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

        # Import WorkflowEvent here (workflow module is now loaded)
        from agentic_cli.workflow import WorkflowEvent

        bind_context(user_id=settings.default_user)
        logger.info("handling_message", message_length=len(message))

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
            if state.thinking_started and events_ctx is not None:
                events_ctx.finish(add_to_history=False)
                state.thinking_started = False

            response = await self._prompt_user_input(request, ui)

            events_ctx = ui.start_thinking(
                state.get_status, content_format="ansi"
            )
            state.thinking_started = True
            return response

        workflow.set_input_callback(_handle_input)
        try:
            while True:
                try:
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
