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
