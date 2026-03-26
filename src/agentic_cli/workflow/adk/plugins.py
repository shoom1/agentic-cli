"""ADK Plugins for agentic-cli.

Provides framework-level cross-cutting concerns as ADK Plugins:
- ConfirmationPlugin: HITL confirmation for DANGEROUS tools
- LLMLoggingPlugin: Raw LLM traffic logging for debugging
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin

from agentic_cli.tools.registry import get_registry, PermissionLevel
from agentic_cli.workflow.context import get_context_workflow
from agentic_cli.workflow.events import WorkflowEvent, UserInputRequest, InputType
from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models import LlmRequest, LlmResponse
    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext

logger = Loggers.workflow()

_APPROVED_RESPONSES = ("yes", "y", "approve", "true")


def is_dangerous(tool_name: str) -> bool:
    """Check if a tool is registered as DANGEROUS permission level.

    Uses the ToolRegistry's O(1) lookup directly — no caching needed.
    """
    defn = get_registry().get(tool_name)
    if defn is None:
        return False
    return defn.permission_level == PermissionLevel.DANGEROUS


async def request_tool_confirmation(
    tool_name: str, tool_args: dict[str, Any]
) -> bool | None:
    """Prompt user for confirmation of a dangerous tool call.

    Shared by ConfirmationPlugin (ADK) and _wrap_for_confirmation (LangGraph).

    Returns:
        True if approved, False if denied, None if no workflow/callback available.
    """
    workflow = get_context_workflow()
    if workflow is None:
        return None

    arg_summary = ", ".join(
        f"{k}={repr(v)[:50]}" for k, v in list(tool_args.items())[:3]
    )
    request = UserInputRequest(
        request_id=str(uuid.uuid4())[:8],
        tool_name=tool_name,
        prompt=(
            f"Tool requires approval: {tool_name}({arg_summary})\n\n"
            f"Allow this operation? (yes/no)"
        ),
        input_type=InputType.CONFIRM,
        default="no",
    )

    try:
        response = await workflow.request_user_input(request)
    except RuntimeError:
        return None

    return response.strip().lower() in _APPROVED_RESPONSES


class ConfirmationPlugin(BasePlugin):
    """ADK Plugin that requires user confirmation for DANGEROUS tools.

    Uses the workflow manager's request_user_input callback to prompt
    the user before executing any tool with PermissionLevel.DANGEROUS.

    Replaces the old _wrap_dangerous decorator pattern with a single
    framework-level hook that applies to all agents globally.
    """

    def __init__(self) -> None:
        super().__init__(name="confirmation")

    async def before_tool_callback(
        self,
        *,
        tool: "BaseTool",
        tool_args: dict[str, Any],
        tool_context: "ToolContext",
    ) -> dict | None:
        """Intercept DANGEROUS tool calls and request user confirmation."""
        if not is_dangerous(tool.name):
            return None

        approved = await request_tool_confirmation(tool.name, tool_args)

        if approved is None:
            logger.warning("confirmation_plugin.no_workflow_or_callback", tool=tool.name)
            return None

        if approved:
            logger.debug("confirmation_plugin.approved", tool=tool.name)
            return None

        logger.info("confirmation_plugin.denied", tool=tool.name)
        return {
            "success": False,
            "error": f"User denied approval for {tool.name}",
        }


class LLMLoggingPlugin(BasePlugin):
    """ADK Plugin that captures raw LLM request/response traffic for debugging.

    Replaces the old per-agent LLMEventLogger with a single global plugin
    that applies to all agents via the Runner. Events are written to a
    JSON Lines file and buffered in memory for programmatic access.

    Attributes:
        model_name: Default model name (used if not in request)
        app_name: Application name for log directory (./.{app_name}/logs/)
        max_events: Maximum events to buffer in memory (oldest are dropped)
        include_messages: Whether to include full message history
        include_raw_parts: Whether to include raw response parts
    """

    def __init__(
        self,
        model_name: str = "unknown",
        app_name: str = "agentic_cli",
        max_events: int = 1000,
        include_messages: bool = True,
        include_raw_parts: bool = True,
    ) -> None:
        super().__init__(name="llm_logging")
        self.model_name = model_name
        self.app_name = app_name
        self.max_events = max_events
        self.include_messages = include_messages
        self.include_raw_parts = include_raw_parts

        self._events: deque[WorkflowEvent] = deque(maxlen=max_events)
        self._request_timestamps: dict[str, float] = {}

        # Initialize log file
        log_dir = Path.cwd() / f".{self.app_name}" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file: Path = log_dir / "llm_events.jsonl"

    # ------------------------------------------------------------------
    # ADK Plugin callbacks
    # ------------------------------------------------------------------

    async def before_model_callback(
        self,
        *,
        callback_context: "CallbackContext",
        llm_request: "LlmRequest",
    ) -> None:
        """Capture LLM request before it's sent to the model.

        Returns None to allow the request to proceed unchanged.
        """
        invocation_id = getattr(callback_context, "invocation_id", None)

        # Record timestamp for latency calculation
        if invocation_id:
            self._request_timestamps[invocation_id] = time.perf_counter()

        # Extract model name
        model = llm_request.model or self.model_name

        # Extract messages if enabled
        messages = None
        if self.include_messages and llm_request.contents:
            messages = self._serialize_contents(llm_request.contents)

        # Extract tool names
        tools = None
        if llm_request.tools_dict:
            tools = list(llm_request.tools_dict.keys())

        # Extract system instruction
        system_instruction = None
        if llm_request.config and llm_request.config.system_instruction:
            si = llm_request.config.system_instruction
            if isinstance(si, str):
                system_instruction = si
            elif hasattr(si, "parts"):
                system_instruction = self._extract_text_from_parts(si.parts)

        # Extract generation config
        config = None
        if llm_request.config:
            config = self._serialize_config(llm_request.config)

        event = WorkflowEvent.llm_request(
            model=model,
            messages=messages,
            tools=tools,
            system_instruction=system_instruction,
            config=config,
            invocation_id=invocation_id,
        )

        self._events.append(event)
        self._write_to_log(event)

        return None

    async def after_model_callback(
        self,
        *,
        callback_context: "CallbackContext",
        llm_response: "LlmResponse",
    ) -> None:
        """Capture LLM response after it's received from the model.

        Returns None to allow the response to proceed unchanged.
        """
        invocation_id = getattr(callback_context, "invocation_id", None)

        # Calculate latency
        latency_ms = None
        if invocation_id and invocation_id in self._request_timestamps:
            start_time = self._request_timestamps.pop(invocation_id)
            latency_ms = (time.perf_counter() - start_time) * 1000

        model = self.model_name

        # Extract text content
        content = None
        raw_parts = None
        if llm_response.content and llm_response.content.parts:
            content = self._extract_text_from_parts(llm_response.content.parts)
            if self.include_raw_parts:
                raw_parts = self._serialize_parts(llm_response.content.parts)

        # Extract finish reason
        finish_reason = None
        if llm_response.finish_reason:
            finish_reason = str(llm_response.finish_reason)

        # Extract model version
        model_version = llm_response.model_version

        # Extract error info
        error_code = llm_response.error_code
        error_message = llm_response.error_message

        # Get author from context if available
        author = getattr(callback_context, "agent_name", None)

        # Create response event
        response_event = WorkflowEvent.llm_response(
            model=model,
            content=content,
            finish_reason=finish_reason,
            model_version=model_version,
            error_code=error_code,
            error_message=error_message,
            invocation_id=invocation_id,
            author=author,
            raw_parts=raw_parts,
        )
        self._events.append(response_event)
        self._write_to_log(response_event)

        # Create usage event if usage_metadata is available
        if llm_response.usage_metadata:
            usage = llm_response.usage_metadata
            usage_event = WorkflowEvent.llm_usage(
                model=model,
                prompt_tokens=getattr(usage, "prompt_token_count", None),
                completion_tokens=getattr(usage, "candidates_token_count", None),
                total_tokens=getattr(usage, "total_token_count", None),
                thinking_tokens=getattr(usage, "thoughts_token_count", None),
                cached_tokens=getattr(usage, "cached_content_token_count", None),
                invocation_id=invocation_id,
                latency_ms=latency_ms,
            )
            self._events.append(usage_event)
            self._write_to_log(usage_event)

        return None

    # ------------------------------------------------------------------
    # Event buffer management
    # ------------------------------------------------------------------

    def get_events(self) -> list[WorkflowEvent]:
        """Get all captured events without clearing the buffer."""
        return list(self._events)

    def drain_events(self) -> list[WorkflowEvent]:
        """Get and clear all captured events."""
        events = list(self._events)
        self._events.clear()
        return events

    def clear(self) -> None:
        """Clear all captured events and pending timestamps."""
        self._events.clear()
        self._request_timestamps.clear()

    def get_log_file_path(self) -> Path:
        """Get the path to the log file."""
        return self._log_file

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_to_log(self, event: WorkflowEvent) -> None:
        """Write event to JSON Lines log file."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event.type.value,
            "content": event.content,
            "metadata": event.metadata,
        }

        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass

    def _serialize_contents(self, contents: list[Any]) -> list[dict[str, Any]]:
        """Serialize Content objects to dictionaries."""
        result = []
        for content in contents:
            item: dict[str, Any] = {}
            if hasattr(content, "role"):
                item["role"] = content.role
            if hasattr(content, "parts") and content.parts:
                item["parts"] = self._serialize_parts(content.parts)
            result.append(item)
        return result

    def _serialize_parts(self, parts: list[Any]) -> list[dict[str, Any]]:
        """Serialize Part objects to dictionaries."""
        result = []
        for part in parts:
            item: dict[str, Any] = {}

            if hasattr(part, "text") and part.text:
                item["text"] = part.text

            if hasattr(part, "thought") and part.thought:
                item["thought"] = True

            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                item["function_call"] = {
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                }

            if hasattr(part, "function_response") and part.function_response:
                fr = part.function_response
                item["function_response"] = {
                    "name": fr.name,
                    "response": fr.response,
                }

            if hasattr(part, "executable_code") and part.executable_code:
                ec = part.executable_code
                item["executable_code"] = {
                    "code": ec.code,
                    "language": str(ec.language) if ec.language else None,
                }

            if hasattr(part, "code_execution_result") and part.code_execution_result:
                cer = part.code_execution_result
                item["code_execution_result"] = {
                    "outcome": str(cer.outcome) if cer.outcome else None,
                }

            if item:
                result.append(item)

        return result

    def _extract_text_from_parts(self, parts: list[Any]) -> str:
        """Extract text content from parts."""
        texts = []
        for part in parts:
            if hasattr(part, "text") and part.text:
                texts.append(part.text)
        return "\n".join(texts)

    def _serialize_config(self, config: Any) -> dict[str, Any]:
        """Serialize GenerateContentConfig to dictionary."""
        result: dict[str, Any] = {}

        if hasattr(config, "temperature") and config.temperature is not None:
            result["temperature"] = config.temperature
        if hasattr(config, "top_p") and config.top_p is not None:
            result["top_p"] = config.top_p
        if hasattr(config, "top_k") and config.top_k is not None:
            result["top_k"] = config.top_k
        if hasattr(config, "max_output_tokens") and config.max_output_tokens is not None:
            result["max_output_tokens"] = config.max_output_tokens
        if hasattr(config, "stop_sequences") and config.stop_sequences:
            result["stop_sequences"] = config.stop_sequences

        return result
