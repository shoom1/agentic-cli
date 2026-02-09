"""Message processing for agentic CLI applications.

This module provides:
- MessageHistory: Tracks conversation history for persistence
- MessageProcessor: Handles user message processing through workflow
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from agentic_cli.logging import Loggers, bind_context

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.cli.workflow_controller import WorkflowController
    from agentic_cli.workflow import WorkflowEvent
    from agentic_cli.workflow.events import UserInputRequest
    from thinking_prompt import ThinkingPromptSession


logger = Loggers.cli()


# === Message History for Persistence ===


class MessageType(Enum):
    """Types of messages in history."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ERROR = "error"
    WARNING = "warning"
    SUCCESS = "success"
    THINKING = "thinking"


@dataclass
class Message:
    """A message stored in history for persistence."""

    content: str
    message_type: MessageType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class MessageHistory:
    """Simple message history for persistence."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(
        self,
        content: str,
        message_type: MessageType | str,
        timestamp: datetime | None = None,
        **metadata: object,
    ) -> None:
        """Add a message to history."""
        if isinstance(message_type, str):
            message_type = MessageType(message_type)
        self._messages.append(
            Message(
                content=content,
                message_type=message_type,
                timestamp=timestamp or datetime.now(),
                metadata=dict(metadata),
            )
        )

    def get_all(self) -> list[Message]:
        """Get all messages."""
        return list(self._messages)

    def get_by_type(self, message_type: MessageType) -> list[Message]:
        """Get messages of a specific type."""
        return [m for m in self._messages if m.message_type == message_type]

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


# === Event Processing State ===


@dataclass
class _EventProcessingState:
    """Mutable state shared across event handlers during process()."""

    status_line: str = "Processing..."
    task_progress_display: str | None = None
    thinking_started: bool = False
    thinking_content: list[str] = field(default_factory=list)
    response_content: list[str] = field(default_factory=list)

    def get_status(self) -> str:
        """Build status display with current action and task progress."""
        lines = [self.status_line]
        if self.task_progress_display:
            lines.append(self.task_progress_display)
        return "\n".join(lines)

    def reset_for_retry(self) -> None:
        """Reset state for retry after rate limit."""
        self.status_line = "Processing..."
        self.task_progress_display = None
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
        self._message_history = MessageHistory()

    @property
    def history(self) -> MessageHistory:
        """Get the message history."""
        return self._message_history

    def clear_history(self) -> None:
        """Clear message history."""
        self._message_history.clear()

    async def process(
        self,
        message: str,
        workflow_controller: "WorkflowController",
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
    ) -> None:
        """Process a user message through the workflow.

        Args:
            message: User message to process
            workflow_controller: Controller managing workflow lifecycle
            ui: UI session for output
            settings: Application settings
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

        # Track message in history (if logging enabled)
        if settings.log_activity:
            self._message_history.add(message, MessageType.USER)

        state = _EventProcessingState()
        workflow = workflow_controller.workflow
        dispatch = self._get_event_dispatch()

        # Set up direct callback so HITL tools can prompt the user without
        # deadlocking (the Future pattern blocks the ADK runner, preventing
        # the USER_INPUT_REQUIRED event from ever being yielded).
        async def _handle_input(request: "UserInputRequest") -> str:
            if state.thinking_started:
                ui.finish_thinking(add_to_history=False)
                state.thinking_started = False

            wf_event = WorkflowEvent.user_input_required(
                request_id=request.request_id,
                tool_name=request.tool_name,
                prompt=request.prompt,
                input_type=request.input_type,
                choices=request.choices,
                default=request.default,
            )
            response = await self._prompt_user_input(wf_event, workflow, ui)

            ui.start_thinking(state.get_status)
            state.thinking_started = True
            return response

        workflow._user_input_callback = _handle_input
        try:
            while True:
                try:
                    ui.start_thinking(state.get_status)
                    state.thinking_started = True

                    async for event in workflow.process(
                        message=message,
                        user_id=settings.default_user,
                    ):
                        handler = dispatch.get(event.type)
                        if handler is not None:
                            await handler(self, event, state, ui, settings, workflow)

                    # Finish thinking box (don't add status to history)
                    if state.thinking_started:
                        ui.finish_thinking(add_to_history=False)

                    # Add accumulated content to message history (if logging enabled)
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
                    break  # Success — exit retry loop

                except Exception as e:
                    if state.thinking_started:
                        ui.finish_thinking(add_to_history=False)
                        state.thinking_started = False

                    # Check for 429 rate limit errors — prompt user to wait and retry
                    from agentic_cli.workflow.adk.event_processor import (
                        _is_rate_limit_error,
                        _parse_retry_delay,
                    )

                    if _is_rate_limit_error(e):
                        delay = _parse_retry_delay(e) or 60.0
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
                    if settings.log_activity:
                        self._message_history.add(str(e), MessageType.ERROR)
                    break
        finally:
            workflow._user_input_callback = None

    async def _prompt_user_input(
        self,
        event: "WorkflowEvent",
        workflow: object,
        ui: "ThinkingPromptSession",
    ) -> str:
        """Prompt user for input requested by a tool.

        Args:
            event: USER_INPUT_REQUIRED event with prompt details
            workflow: The workflow manager (unused but kept for potential future use)
            ui: UI session for prompting

        Returns:
            User's response string
        """
        from agentic_cli.workflow.events import InputType

        input_type = event.metadata.get("input_type", InputType.TEXT.value)
        tool_name = event.metadata.get("tool_name", "Tool")
        choices = event.metadata.get("choices")
        default = event.metadata.get("default")

        if input_type == InputType.CHOICE.value and choices:
            # Use choice dialog for multiple options
            result = await ui.dropdown_dialog(
                title=f"{tool_name} - Input Required",
                text=event.content,
                options=choices,
                default=default,
            )
            return result or (default or "")

        elif input_type == InputType.CONFIRM.value:
            # Use yes/no dialog for confirmation
            result = await ui.yes_no_dialog(
                title=f"{tool_name} - Confirmation",
                text=event.content,
            )
            return "yes" if result else "no"

        else:
            # Use text input dialog for free-form input
            result = await ui.input_dialog(
                title=f"{tool_name} - Input Required",
                text=event.content,
                default=default or "",
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

    async def _handle_user_input_required(
        self,
        event: "WorkflowEvent",
        state: _EventProcessingState,
        ui: "ThinkingPromptSession",
        settings: "BaseSettings",
        workflow: object,
    ) -> None:
        """Handle USER_INPUT_REQUIRED events — legacy ADK event stream path."""
        if state.thinking_started:
            ui.finish_thinking(add_to_history=False)
            state.thinking_started = False

        response = await self._prompt_user_input(event, workflow, ui)
        workflow.provide_user_input(
            event.metadata["request_id"],
            response,
        )

        # Resume thinking box
        ui.start_thinking(state.get_status)
        state.thinking_started = True

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
        """Handle TASK_PROGRESS events — update progress display."""
        progress = event.metadata.get("progress", {})
        if progress.get("total", 0) > 0:
            completed = progress.get("completed", 0)
            total = progress["total"]
            state.task_progress_display = f"--- Tasks: {completed}/{total} ---"
            if event.content:
                state.task_progress_display += f"\n{event.content}"

        current_task = event.metadata.get("current_task_description")
        if current_task:
            state.status_line = f"Working on: {current_task}"

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
                EventType.USER_INPUT_REQUIRED: cls._handle_user_input_required,
                EventType.CODE_EXECUTION: cls._handle_code_execution,
                EventType.EXECUTABLE_CODE: cls._handle_executable_code,
                EventType.FILE_DATA: cls._handle_file_data,
                EventType.TASK_PROGRESS: cls._handle_task_progress,
            }
        return cls._EVENT_DISPATCH
