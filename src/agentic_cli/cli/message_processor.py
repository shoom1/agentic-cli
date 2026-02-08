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
from typing import TYPE_CHECKING

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

        # Import EventType and WorkflowEvent here (workflow module is now loaded)
        from agentic_cli.workflow import EventType, WorkflowEvent

        bind_context(user_id=settings.default_user)
        logger.info("handling_message", message_length=len(message))

        # Track message in history (if logging enabled)
        if settings.log_activity:
            self._message_history.add(message, MessageType.USER)

        # Status line for thinking box (multi-line with task progress)
        status_line = "Processing..."
        task_progress_display: str | None = None
        thinking_started = False

        # Accumulate content for history
        thinking_content: list[str] = []
        response_content: list[str] = []

        def get_status() -> str:
            """Build status display with current action and task progress."""
            lines = [status_line]

            # Add task progress if available (from TASK_PROGRESS events)
            if task_progress_display:
                lines.append(task_progress_display)

            return "\n".join(lines)

        workflow = workflow_controller.workflow

        # Set up direct callback so HITL tools can prompt the user without
        # deadlocking (the Future pattern blocks the ADK runner, preventing
        # the USER_INPUT_REQUIRED event from ever being yielded).
        async def _handle_input(request: "UserInputRequest") -> str:
            nonlocal thinking_started
            if thinking_started:
                ui.finish_thinking(add_to_history=False)
                thinking_started = False

            wf_event = WorkflowEvent.user_input_required(
                request_id=request.request_id,
                tool_name=request.tool_name,
                prompt=request.prompt,
                input_type=request.input_type,
                choices=request.choices,
                default=request.default,
            )
            response = await self._prompt_user_input(wf_event, workflow, ui)

            ui.start_thinking(get_status)
            thinking_started = True
            return response

        workflow._user_input_callback = _handle_input
        try:
            while True:
                try:
                    ui.start_thinking(get_status)
                    thinking_started = True

                    async for event in workflow.process(
                        message=message,
                        user_id=settings.default_user,
                    ):
                        if event.type == EventType.TEXT:
                            # Stream response directly to console
                            ui.add_response(event.content, markdown=True)
                            response_content.append(event.content)

                        elif event.type == EventType.THINKING:
                            # Stream thinking to console only if verbose_thinking is enabled
                            status_line = "Thinking..."
                            if settings.verbose_thinking:
                                ui.add_message("system", event.content)
                            thinking_content.append(event.content)

                        elif event.type == EventType.TOOL_CALL:
                            # Update status line in thinking box
                            tool_name = event.metadata.get("tool_name", "unknown")
                            status_line = f"Calling: {tool_name}"

                        elif event.type == EventType.TOOL_RESULT:
                            # Display tool result summary
                            tool_name = event.metadata.get("tool_name", "unknown")
                            success = event.metadata.get("success", True)
                            duration = event.metadata.get("duration_ms")
                            icon = "+" if success else "x"
                            duration_str = f" ({duration}ms)" if duration else ""
                            status_line = f"{icon} {tool_name}: {event.content}{duration_str}"
                            # Also show in message area for visibility
                            style = "green" if success else "red"
                            ui.add_rich(
                                f"[{style}]{icon}[/{style}] {tool_name}: {event.content}{duration_str}"
                            )

                        elif event.type == EventType.USER_INPUT_REQUIRED:
                            # Legacy path: handle USER_INPUT_REQUIRED events
                            # that may still arrive from the ADK event stream.
                            if thinking_started:
                                ui.finish_thinking(add_to_history=False)
                                thinking_started = False

                            response = await self._prompt_user_input(event, workflow, ui)
                            workflow.provide_user_input(
                                event.metadata["request_id"],
                                response,
                            )

                            # Resume thinking box
                            ui.start_thinking(get_status)
                            thinking_started = True

                        elif event.type == EventType.CODE_EXECUTION:
                            # Update status with execution result
                            result_preview = (
                                event.content[:40] + "..."
                                if len(event.content) > 40
                                else event.content
                            )
                            status_line = f"Result: {result_preview}"

                        elif event.type == EventType.EXECUTABLE_CODE:
                            # Update status when executing code
                            lang = event.metadata.get("language", "python")
                            status_line = f"Running {lang} code..."

                        elif event.type == EventType.FILE_DATA:
                            # Update status with file info
                            status_line = f"File: {event.content}"

                        elif event.type == EventType.TASK_PROGRESS:
                            # Task progress update - event.content has compact display,
                            # event.metadata has progress stats
                            progress = event.metadata.get("progress", {})
                            if progress.get("total", 0) > 0:
                                completed = progress.get("completed", 0)
                                total = progress["total"]
                                # Build display: separator + task list
                                task_progress_display = f"--- Tasks: {completed}/{total} ---"
                                if event.content:
                                    task_progress_display += f"\n{event.content}"

                            # Update status line with current task if available
                            current_task = event.metadata.get("current_task_description")
                            if current_task:
                                status_line = f"Working on: {current_task}"

                    # Finish thinking box (don't add status to history)
                    if thinking_started:
                        ui.finish_thinking(add_to_history=False)

                    # Add accumulated content to message history (if logging enabled)
                    if settings.log_activity:
                        if thinking_content:
                            self._message_history.add(
                                "".join(thinking_content), MessageType.THINKING
                            )
                        if response_content:
                            self._message_history.add(
                                "".join(response_content), MessageType.ASSISTANT
                            )

                    logger.debug("message_handled_successfully")
                    break  # Success — exit retry loop

                except Exception as e:
                    if thinking_started:
                        ui.finish_thinking(add_to_history=False)
                        thinking_started = False

                    # Check for 429 rate limit errors — prompt user to wait and retry
                    from agentic_cli.workflow.adk_manager import (
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
                            # Reset state for retry
                            status_line = "Processing..."
                            task_progress_display = None
                            thinking_content.clear()
                            response_content.clear()
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
