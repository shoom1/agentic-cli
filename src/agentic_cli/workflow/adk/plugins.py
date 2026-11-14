"""ADK Plugins for agentic-cli.

Provides framework-level cross-cutting concerns as ADK Plugins:
- LLMLoggingPlugin: Raw LLM traffic logging for debugging
- SentRequestLoggingPlugin: records a request as it is sent, after every
  plugin that edits it (the fit guard), for ``LLMLoggingPlugin``
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin

from agentic_cli.workflow.adk.call_clock import CallClock
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.llm_log import LLMLogWriter
from agentic_cli.workflow.sessions import get_active_run
from agentic_cli.workflow.usage import adk_usage_event
from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.models import LlmRequest, LlmResponse

logger = Loggers.workflow()


class LLMLoggingPlugin(BasePlugin):
    """ADK Plugin that captures raw LLM request/response traffic for debugging.

    Replaces the old per-agent LLMEventLogger with a single global plugin
    that applies to all agents via the Runner. Events are written to a
    JSON Lines file (through the shared ``workflow.llm_log.LLMLogWriter``)
    and buffered in memory for programmatic access, but for each call's
    usage record, which stays in the file: the event processor streams each
    call's ``LLM_USAGE``, credited to its agent, and a second copy would count
    every call twice (as on Deep Agents, where logging adds no usage).

    Each record names the model of the agent that made the call (``models``),
    else the request's own model or ``model_name``; that agent
    (``metadata["agent"]``); and, for a sub-agent's call, its run
    (``WorkflowEvent.run``, the active run its ``MainAgentPlugin`` set,
    which the log writes as ``run``), as Deep Agents' records do. A usage
    record counts as the stream's ``LLM_USAGE`` does
    (``workflow/usage.py``'s ``adk_usage_event``, by the agent's model's
    provider kind, ``kinds``: the thinking in the output, a cache write in
    ``cache_creation_tokens``).

    A call's latency runs from its request to its response, timed per model
    call by a clock of its own (``workflow/adk/call_clock.py``'s
    ``CallClock``, the one every app's calls are timed by for their usage):
    under the identity of the call's ``EventActions``, which ADK hands to
    all three of its callbacks. A failed call keeps its clock
    (``on_model_error_callback``): a plugin after this one may answer the
    error (the overflow handler), and that answer reaches
    ``after_model_callback`` like any response, timed. Each run lets go of
    what another invocation left.

    Attributes:
        model_name: Default model name, for an agent ``models`` does not hold
        models: Each agent's model reference, by agent name
        kinds: Each agent's model's provider kind, by agent name; an agent
            it does not hold is an unknown kind's
        app_name: Application name for log directory (~/.{app_name}/logs/)
        max_events: Maximum events to buffer in memory (oldest are dropped)
        include_messages: Whether to include full message history
        include_raw_parts: Whether to include raw response parts
        log_requests: Whether this plugin records requests itself; False when
            a ``SentRequestLoggingPlugin`` records them as sent
    """

    def __init__(
        self,
        model_name: str = "unknown",
        app_name: str = "agentic_cli",
        max_events: int = 1000,
        include_messages: bool = True,
        include_raw_parts: bool = True,
        log_requests: bool = True,
        models: Mapping[str, str] | None = None,
        kinds: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(name="llm_logging")
        self.model_name = model_name
        self.models: Mapping[str, str] = models or {}
        self.kinds: Mapping[str, str] = kinds or {}
        self.app_name = app_name
        self.max_events = max_events
        self.include_messages = include_messages
        self.include_raw_parts = include_raw_parts
        self.log_requests = log_requests

        self._events: deque[WorkflowEvent] = deque(maxlen=max_events)
        #: Each model call's clock, from its request to its response.
        self.clock = CallClock()

        # The log lives under the user's home — NOT the (often git-tracked)
        # project dir — since it holds full conversations.
        self._log = LLMLogWriter(self.app_name)
        self._log_file: Path = self._log.path

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
        if self.log_requests:
            self.record_request(callback_context, llm_request)
        return None

    def record_request(
        self, callback_context: "CallbackContext", llm_request: "LlmRequest"
    ) -> None:
        """Record ``llm_request`` (and start its call's latency clock)."""
        invocation_id = getattr(callback_context, "invocation_id", None)
        agent = getattr(callback_context, "agent_name", None)

        self.clock.start(callback_context)

        # The calling agent's model, else the one the request names
        model = self.models.get(agent) or llm_request.model or self.model_name

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
        _attributed(event, agent)

        self._events.append(event)
        self._write_to_log(event)

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
        latency_ms = self.clock.elapsed(
            callback_context, done=getattr(llm_response, "partial", None) is not True
        )

        # Get author from context if available
        author = getattr(callback_context, "agent_name", None)
        model = self.models.get(author) or self.model_name

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
        _attributed(response_event, author)
        self._events.append(response_event)
        self._write_to_log(response_event)

        # The usage record, for the log only: the stream's usage is the event
        # processor's, credited to the agent that made the call.
        if llm_response.usage_metadata:
            usage_event = adk_usage_event(
                model,
                llm_response.usage_metadata,
                self.kinds.get(author),
                invocation_id=invocation_id,
                latency_ms=latency_ms,
            )
            self._write_to_log(_attributed(usage_event, author))

        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: "CallbackContext",
        llm_request: "LlmRequest",
        error: Exception,
    ) -> None:
        """Keep a failed call's clock: the error is the next plugin's to answer.

        The overflow handler answers a call refused as too long, and ADK
        hands its answer to every plugin's ``after_model_callback`` with the
        call's ``EventActions``, so the answer is timed and lets the clock
        go. An error nobody answers leaves its clock to the next run's
        ``before_run_callback``.
        """
        return None

    async def before_run_callback(self, *, invocation_context: "InvocationContext") -> None:
        # What another invocation left (turns run one at a time): a call
        # whose error nobody answered, and one cancelled mid-request, which
        # before google-adk 2.14 fired no callback.
        self.clock.keep_only(invocation_context.invocation_id)
        return None

    # ------------------------------------------------------------------
    # Event buffer management
    # ------------------------------------------------------------------

    def drain_events(self) -> list[WorkflowEvent]:
        """Get and clear all captured events."""
        events = list(self._events)
        self._events.clear()
        return events

    def clear(self) -> None:
        """Clear all captured events and pending timestamps."""
        self._events.clear()
        self.clock.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_to_log(self, event: WorkflowEvent) -> None:
        """Write event to JSON Lines log file."""
        self._log.write(event)

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


def _attributed(event: WorkflowEvent, agent: str | None) -> WorkflowEvent:
    """``event`` naming the agent whose model call it records, and that
    call's sub-agent run (``get_active_run``; none for the root's own)."""
    if agent:
        event.metadata["agent"] = agent
    event.run = get_active_run()
    return event


class SentRequestLoggingPlugin(BasePlugin):
    """Records each request to an ``LLMLoggingPlugin`` as it is sent.

    Registered after every plugin that edits a request (the fit guard leaves
    turns out of one over its budget), so the log holds what went out. The
    ``LLMLoggingPlugin`` itself stays ahead, so its ``after_model_callback``
    sees a response before a plugin could replace it.
    """

    def __init__(self, log: LLMLoggingPlugin) -> None:
        super().__init__(name="llm_logging_sent")
        self._log = log

    async def before_model_callback(
        self, *, callback_context: "CallbackContext", llm_request: "LlmRequest"
    ) -> None:
        self._log.record_request(callback_context, llm_request)
        return None
