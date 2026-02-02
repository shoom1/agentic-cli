"""Event types for workflow communication.

This module defines the event protocol for workflow → consumer communication,
enabling UI-agnostic workflow processing.

Event Types:
    - TEXT: Final text response content
    - THINKING: Model's reasoning/thinking content
    - TOOL_CALL: Tool invocation (before execution)
    - TOOL_RESULT: Tool execution result (after execution)
    - CODE_EXECUTION: Code execution result
    - EXECUTABLE_CODE: Code to be executed
    - FILE_DATA: File attachment/data
    - ERROR: Error message

Example:
    async for event in workflow.process(message, user_id):
        if event.type == EventType.TOOL_CALL:
            print(f"Calling {event.metadata['tool_name']}")
            print(f"Args: {event.metadata.get('tool_args', {})}")
        elif event.type == EventType.TOOL_RESULT:
            print(f"Result: {event.metadata['result']}")
            print(f"Duration: {event.metadata.get('duration_ms')}ms")
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(Enum):
    """Types of events yielded by the workflow."""

    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CODE_EXECUTION = "code_execution_result"
    EXECUTABLE_CODE = "executable_code"
    FILE_DATA = "file_data"
    ERROR = "error"
    USER_INPUT_REQUIRED = "user_input_required"
    TASK_PROGRESS = "task_progress"

    # LLM debugging events (raw traffic logging)
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_USAGE = "llm_usage"


class InputType(str, Enum):
    """Types of user input that can be requested."""

    TEXT = "text"
    CHOICE = "choice"
    CONFIRM = "confirm"


@dataclass
class UserInputRequest:
    """Request for user input from a tool.

    Attributes:
        request_id: Unique ID to correlate response
        tool_name: Which tool is asking for input
        prompt: Question/prompt to show user
        input_type: Type of input expected (TEXT, CHOICE, CONFIRM)
        choices: Available choices for CHOICE input type
        default: Default value if user provides no input
    """

    request_id: str
    tool_name: str
    prompt: str
    input_type: InputType = InputType.TEXT
    choices: list[str] | None = None
    default: str | None = None


@dataclass
class WorkflowEvent:
    """Event yielded by the workflow manager.

    Attributes:
        type: The type of event
        content: Primary content (text, code, etc.)
        metadata: Optional additional data specific to event type
        timestamp: When the event occurred
    """

    type: EventType
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def text(cls, content: str, session_id: str | None = None) -> "WorkflowEvent":
        """Create a text response event."""
        metadata = {"session_id": session_id} if session_id else {}
        return cls(type=EventType.TEXT, content=content, metadata=metadata)

    @classmethod
    def thinking(cls, content: str, session_id: str | None = None) -> "WorkflowEvent":
        """Create a thinking/reasoning event."""
        metadata = {"session_id": session_id} if session_id else {}
        return cls(type=EventType.THINKING, content=content, metadata=metadata)

    @classmethod
    def tool_call(
        cls,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
    ) -> "WorkflowEvent":
        """Create a tool call event.

        Args:
            tool_name: Name of the tool being called
            tool_args: Optional arguments passed to the tool
        """
        metadata = {
            "tool_name": tool_name,
            "tool_args": tool_args or {},
        }
        return cls(
            type=EventType.TOOL_CALL,
            content=f"Calling: {tool_name}",
            metadata=metadata,
        )

    @classmethod
    def tool_result(
        cls,
        tool_name: str,
        result: Any,
        success: bool = True,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> "WorkflowEvent":
        """Create a tool result event.

        Args:
            tool_name: Name of the tool that was called
            result: The result from the tool
            success: Whether the tool call succeeded
            duration_ms: How long the tool took to execute (milliseconds)
            error: Error message if success is False
        """
        content = cls._format_result_content(result)

        metadata: dict[str, Any] = {
            "tool_name": tool_name,
            "result": result,
            "success": success,
        }
        if duration_ms is not None:
            metadata["duration_ms"] = duration_ms
        if error is not None:
            metadata["error"] = error

        return cls(
            type=EventType.TOOL_RESULT,
            content=content,
            metadata=metadata,
        )

    @staticmethod
    def _format_result_content(result: Any) -> str:
        """Format result for display, truncating large values."""
        if isinstance(result, str):
            return result[:200] + "..." if len(result) > 200 else result
        if isinstance(result, (dict, list)):
            return f"Result: {len(result)} items"
        return str(result)[:200]

    @classmethod
    def code_execution(cls, outcome: str) -> "WorkflowEvent":
        """Create a code execution result event."""
        return cls(
            type=EventType.CODE_EXECUTION,
            content=outcome,
            metadata={"outcome": outcome},
        )

    @classmethod
    def executable_code(cls, code: str, language: str) -> "WorkflowEvent":
        """Create an executable code event."""
        return cls(
            type=EventType.EXECUTABLE_CODE,
            content=code,
            metadata={"language": language},
        )

    @classmethod
    def file_data(cls, display_name: str) -> "WorkflowEvent":
        """Create a file data event."""
        return cls(
            type=EventType.FILE_DATA,
            content=display_name,
            metadata={"display_name": display_name},
        )

    @classmethod
    def error(
        cls,
        message: str,
        error_code: str | None = None,
        recoverable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> "WorkflowEvent":
        """Create an error event.

        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            recoverable: Whether the error can be recovered from
            details: Additional error details
        """
        metadata: dict[str, Any] = {
            "recoverable": recoverable,
        }
        if error_code:
            metadata["error_code"] = error_code
        if details:
            metadata["details"] = details

        return cls(type=EventType.ERROR, content=message, metadata=metadata)

    @classmethod
    def user_input_required(
        cls,
        request_id: str,
        tool_name: str,
        prompt: str,
        input_type: InputType | str = InputType.TEXT,
        choices: list[str] | None = None,
        default: str | None = None,
    ) -> "WorkflowEvent":
        """Create a user input required event.

        This event signals that a tool needs user input to proceed.
        The CLI should prompt the user and call workflow.provide_user_input()
        with the response.

        Args:
            request_id: Unique ID to correlate the response
            tool_name: Name of the tool requesting input
            prompt: Question/prompt to show the user
            input_type: Type of input (TEXT, CHOICE, CONFIRM)
            choices: Available choices for CHOICE input type
            default: Default value if user provides no input
        """
        # Normalize input_type to enum value string for metadata
        if isinstance(input_type, InputType):
            input_type_value = input_type.value
        else:
            input_type_value = input_type

        metadata: dict[str, Any] = {
            "request_id": request_id,
            "tool_name": tool_name,
            "input_type": input_type_value,
        }
        if choices:
            metadata["choices"] = choices
        if default is not None:
            metadata["default"] = default

        return cls(
            type=EventType.USER_INPUT_REQUIRED,
            content=prompt,
            metadata=metadata,
        )

    @classmethod
    def task_progress(
        cls,
        display: str,
        progress: dict[str, int],
        current_task_id: str | None = None,
        current_task_description: str | None = None,
    ) -> "WorkflowEvent":
        """Create a task progress event for updating the thinking box.

        This event signals that the task graph has been updated and the UI
        should refresh its display of task progress.

        Args:
            display: Formatted string representation of the task graph
            progress: Progress statistics (total, pending, completed, etc.)
            current_task_id: ID of the task currently being worked on
            current_task_description: Description of the current task
        """
        metadata: dict[str, Any] = {
            "progress": progress,
        }
        if current_task_id is not None:
            metadata["current_task_id"] = current_task_id
        if current_task_description is not None:
            metadata["current_task_description"] = current_task_description

        return cls(
            type=EventType.TASK_PROGRESS,
            content=display,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    # LLM debugging events (raw traffic logging)
    # -------------------------------------------------------------------------

    @classmethod
    def llm_request(
        cls,
        model: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[str] | None = None,
        system_instruction: str | None = None,
        config: dict[str, Any] | None = None,
        invocation_id: str | None = None,
    ) -> "WorkflowEvent":
        """Create an LLM request event for debugging.

        Captures raw request data sent to the LLM before inference.

        Args:
            model: Model identifier being called
            messages: Conversation history/messages sent
            tools: List of tool names available to the model
            system_instruction: System prompt/instruction
            config: Generation config (temperature, max_tokens, etc.)
            invocation_id: ADK invocation ID for correlation
        """
        metadata: dict[str, Any] = {
            "model": model,
        }
        if messages is not None:
            metadata["messages"] = messages
        if tools is not None:
            metadata["tools"] = tools
        if system_instruction is not None:
            metadata["system_instruction"] = system_instruction
        if config is not None:
            metadata["config"] = config
        if invocation_id is not None:
            metadata["invocation_id"] = invocation_id

        return cls(
            type=EventType.LLM_REQUEST,
            content=f"LLM Request: {model}",
            metadata=metadata,
        )

    @classmethod
    def llm_response(
        cls,
        model: str,
        content: str | None = None,
        finish_reason: str | None = None,
        model_version: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        invocation_id: str | None = None,
        author: str | None = None,
        raw_parts: list[dict[str, Any]] | None = None,
    ) -> "WorkflowEvent":
        """Create an LLM response event for debugging.

        Captures raw response data from the LLM after inference.

        Args:
            model: Model identifier that responded
            content: Text content of response (may be truncated for display)
            finish_reason: Why generation stopped (STOP, MAX_TOKENS, etc.)
            model_version: Specific model version used
            error_code: Error code if request failed
            error_message: Error message if request failed
            invocation_id: ADK invocation ID for correlation
            author: Agent name that generated this response
            raw_parts: Raw response parts for detailed inspection
        """
        metadata: dict[str, Any] = {
            "model": model,
        }
        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason
        if model_version is not None:
            metadata["model_version"] = model_version
        if error_code is not None:
            metadata["error_code"] = error_code
        if error_message is not None:
            metadata["error_message"] = error_message
        if invocation_id is not None:
            metadata["invocation_id"] = invocation_id
        if author is not None:
            metadata["author"] = author
        if raw_parts is not None:
            metadata["raw_parts"] = raw_parts

        display_content = content[:200] + "..." if content and len(content) > 200 else (content or "")
        return cls(
            type=EventType.LLM_RESPONSE,
            content=f"LLM Response: {display_content}",
            metadata=metadata,
        )

    @classmethod
    def llm_usage(
        cls,
        model: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        thinking_tokens: int | None = None,
        cached_tokens: int | None = None,
        invocation_id: str | None = None,
        latency_ms: float | None = None,
    ) -> "WorkflowEvent":
        """Create an LLM usage event for debugging and metrics.

        Captures token usage and performance metrics from an LLM call.

        Args:
            model: Model identifier
            prompt_tokens: Input token count
            completion_tokens: Output token count
            total_tokens: Total token count
            thinking_tokens: Tokens used for thinking/reasoning
            cached_tokens: Tokens served from cache
            invocation_id: ADK invocation ID for correlation
            latency_ms: Response latency in milliseconds
        """
        metadata: dict[str, Any] = {
            "model": model,
        }
        if prompt_tokens is not None:
            metadata["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            metadata["completion_tokens"] = completion_tokens
        if total_tokens is not None:
            metadata["total_tokens"] = total_tokens
        if thinking_tokens is not None:
            metadata["thinking_tokens"] = thinking_tokens
        if cached_tokens is not None:
            metadata["cached_tokens"] = cached_tokens
        if invocation_id is not None:
            metadata["invocation_id"] = invocation_id
        if latency_ms is not None:
            metadata["latency_ms"] = latency_ms

        # Build summary content
        parts = []
        if prompt_tokens is not None:
            parts.append(f"in={prompt_tokens}")
        if completion_tokens is not None:
            parts.append(f"out={completion_tokens}")
        if total_tokens is not None:
            parts.append(f"total={total_tokens}")
        if latency_ms is not None:
            parts.append(f"{latency_ms:.0f}ms")

        summary = ", ".join(parts) if parts else "No usage data"
        return cls(
            type=EventType.LLM_USAGE,
            content=f"LLM Usage ({model}): {summary}",
            metadata=metadata,
        )
