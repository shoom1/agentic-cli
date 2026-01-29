"""LLM Event Logger for capturing raw LLM traffic.

This module provides callbacks for Google ADK's before_model_callback and
after_model_callback hooks to capture raw LLM request/response data for
debugging and understanding model-specific behaviors.

Usage:
    logger = LLMEventLogger(model_name="gemini-2.0-flash")

    agent = LlmAgent(
        name="my_agent",
        before_model_callback=logger.before_model_callback,
        after_model_callback=logger.after_model_callback,
    )

    # After processing, retrieve events:
    for event in logger.get_events():
        print(event.type, event.metadata)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from google.adk.models import LlmRequest, LlmResponse
    from google.adk.agents.callback_context import CallbackContext

logger = Loggers.workflow()


@dataclass
class LLMEventLogger:
    """Logger for capturing raw LLM request/response traffic.

    This class provides ADK callbacks that capture LLM interactions for
    debugging purposes. Events are stored in a buffer and can be retrieved
    via get_events() or consumed via drain_events().

    Attributes:
        model_name: Default model name (used if not in request)
        max_events: Maximum events to buffer (oldest are dropped)
        include_messages: Whether to include full message history
        include_raw_parts: Whether to include raw response parts
    """

    model_name: str = "unknown"
    max_events: int = 1000
    include_messages: bool = True
    include_raw_parts: bool = True

    _events: list[WorkflowEvent] = field(default_factory=list)
    _request_timestamps: dict[str, float] = field(default_factory=dict)

    def before_model_callback(
        self,
        callback_context: "CallbackContext",
        llm_request: "LlmRequest",
    ) -> None:
        """Capture LLM request before it's sent to the model.

        This callback captures the raw request data including messages,
        tools, system instruction, and configuration.

        Args:
            callback_context: ADK callback context with session info
            llm_request: The request being sent to the LLM

        Returns:
            None (allows request to proceed unchanged)
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
                # Content object with parts
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

        self._add_event(event)
        logger.debug(
            "llm_request_captured",
            model=model,
            invocation_id=invocation_id,
            message_count=len(messages) if messages else 0,
            tool_count=len(tools) if tools else 0,
        )

        return None  # Allow request to proceed

    def after_model_callback(
        self,
        callback_context: "CallbackContext",
        llm_response: "LlmResponse",
    ) -> None:
        """Capture LLM response after it's received from the model.

        This callback captures the raw response data including content,
        finish reason, usage metadata, and any errors.

        Args:
            callback_context: ADK callback context with session info
            llm_response: The response from the LLM

        Returns:
            None (allows response to proceed unchanged)
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
        self._add_event(response_event)

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
            self._add_event(usage_event)

            logger.debug(
                "llm_response_captured",
                model=model,
                invocation_id=invocation_id,
                finish_reason=finish_reason,
                prompt_tokens=getattr(usage, "prompt_token_count", None),
                completion_tokens=getattr(usage, "candidates_token_count", None),
                latency_ms=latency_ms,
            )
        else:
            logger.debug(
                "llm_response_captured",
                model=model,
                invocation_id=invocation_id,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
            )

        return None  # Allow response to proceed

    def get_events(self) -> list[WorkflowEvent]:
        """Get all captured events without clearing the buffer.

        Returns:
            List of captured WorkflowEvent objects
        """
        return list(self._events)

    def drain_events(self) -> list[WorkflowEvent]:
        """Get and clear all captured events.

        Returns:
            List of captured WorkflowEvent objects (buffer is cleared)
        """
        events = self._events
        self._events = []
        return events

    def clear(self) -> None:
        """Clear all captured events and pending timestamps."""
        self._events = []
        self._request_timestamps = {}

    def _add_event(self, event: WorkflowEvent) -> None:
        """Add event to buffer, enforcing max_events limit."""
        self._events.append(event)
        if len(self._events) > self.max_events:
            # Drop oldest events
            self._events = self._events[-self.max_events :]

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

            # Text content
            if hasattr(part, "text") and part.text:
                item["text"] = part.text

            # Thinking indicator
            if hasattr(part, "thought") and part.thought:
                item["thought"] = True

            # Function call
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                item["function_call"] = {
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                }

            # Function response
            if hasattr(part, "function_response") and part.function_response:
                fr = part.function_response
                item["function_response"] = {
                    "name": fr.name,
                    "response": fr.response,
                }

            # Executable code
            if hasattr(part, "executable_code") and part.executable_code:
                ec = part.executable_code
                item["executable_code"] = {
                    "code": ec.code,
                    "language": str(ec.language) if ec.language else None,
                }

            # Code execution result
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
