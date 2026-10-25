"""ADK plugin: the calls a stopped turn leaves open get their result.

A cancelled turn runs with an ``abort_signal`` that the manager sets
(``GoogleADKWorkflowManager._run_and_stream``), so ADK seals the invocation
as it stops: every call it issued without a result is answered with an
error, and an invocation with no call open gets one content-less event,
both marked ``INVOCATION_ABORTED`` and stored through ``on_event_callback``.
This plugin rewrites them as ``workflow/interrupts.py`` says: a call that
finished keeps its result (ADK stores parallel calls' results together,
after the last, so the finished one would be lost), a call at its
permission prompt was not run, a call whose job was submitted runs on, and
any other was interrupted by the user; the content-less event becomes the
main agent's reply ``MARKER``.

``left_open`` answers, on the next turn, the calls nothing sealed: a crash,
or a cancel from before this plugin. ADK would keep a long-running one
without its result in every later request, which Claude and OpenAI refuse,
and drop any other one, logging "Dropping function calls" on every request.

ADK stops at the first plugin whose ``after_tool_callback`` returns a value,
so this one is registered before ``ToolResultCapPlugin``; a finished result
it answers with is cut as that plugin would have cut it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from google.adk.events import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from agentic_cli.workflow.interrupts import INTERRUPTED, MARKER, OpenCalls

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext

    from agentic_cli.workflow.adk.tool_result_cap_plugin import ToolResultCapPlugin

#: ADK's ``error_code`` on the events that seal an aborted invocation
#: (``google/adk/events/_abort_events.py``, private; contract-tested).
ABORTED = "INVOCATION_ABORTED"


@dataclass(frozen=True, slots=True)
class _Finished:
    """A call that returned before the cancel: what the cap needs to cut it."""

    tool: "BaseTool"
    tool_args: dict[str, Any]
    result: Any


class InterruptedCallsPlugin(BasePlugin):
    """Answer the calls a cancelled turn leaves open (``workflow/interrupts.py``).

    Args:
        open_calls: The manager's record of its calls.
        cap: The tool result cap, for a finished result answered here.
    """

    def __init__(self, open_calls: OpenCalls, cap: "ToolResultCapPlugin | None" = None) -> None:
        super().__init__(name="interrupted_calls")
        self._open = open_calls
        self._cap = cap

    async def after_tool_callback(
        self,
        *,
        tool: "BaseTool",
        tool_args: dict[str, Any],
        tool_context: "ToolContext | None",
        result: Any,
    ) -> None:
        if tool_context is not None:
            self._open.finished(tool_context.function_call_id, _Finished(tool, tool_args, result))
        return None

    async def on_event_callback(
        self, *, invocation_context: "InvocationContext", event: Event
    ) -> Event | None:
        if event.error_code != ABORTED:
            return None
        content = event.content
        if content is None or not content.parts:
            return event.model_copy(update={"content": types.Content(
                role="model", parts=[types.Part(text=MARKER)],
            )})
        parts = [await self._answered(part) for part in content.parts]
        return event.model_copy(update={"content": types.Content(role=content.role, parts=parts)})

    async def _answered(self, part: types.Part) -> types.Part:
        call = part.function_response
        if call is None:
            return part
        answer = self._open.answer(call.id, by_user=True)
        if isinstance(answer, _Finished):
            answer = await self._capped(answer)
        return types.Part(function_response=types.FunctionResponse(
            id=call.id, name=call.name, response=_as_response(answer),
        ))

    async def _capped(self, finished: _Finished) -> Any:
        if self._cap is None:
            return finished.result
        capped = await self._cap.after_tool_callback(
            tool=finished.tool, tool_args=finished.tool_args, tool_context=None,
            result=finished.result,
        )
        return finished.result if capped is None else capped


def _result(answer: Any) -> Any:
    return answer.result if isinstance(answer, _Finished) else answer


def _as_response(result: Any) -> dict[str, Any]:
    """A result as ADK stores it: a dict as it is, anything else under ``result``."""
    return result if isinstance(result, dict) else {"result": result}


def left_open(events: list[Event], open_calls: OpenCalls) -> list[Event]:
    """The events that answer the calls nothing answered: a crash's, or an old cancel's.

    Each is answered ``INTERRUPTED`` (or as ``OpenCalls.answer`` says: a
    job's state), in the author, branch, scope and invocation of the event
    that made it, so it pairs with its call in that agent's view; calls that
    share those are answered in one event, marked as ADK marks a seal.
    """
    answered = {
        response.id for event in events for response in event.get_function_responses()
        if response.id
    }
    grouped: dict[tuple, list[types.FunctionCall]] = {}
    for event in events:
        for call in event.get_function_calls():
            if call.id and call.id not in answered:
                key = (event.invocation_id, event.author, event.branch, event.isolation_scope)
                grouped.setdefault(key, []).append(call)
    return [
        Event(
            invocation_id=invocation_id,
            author=author,
            branch=branch,
            isolation_scope=scope,
            content=types.Content(role="user", parts=[
                types.Part(function_response=types.FunctionResponse(
                    id=call.id, name=call.name,
                    response=_as_response(_result(open_calls.answer(call.id, by_user=False))),
                ))
                for call in calls
            ]),
            error_code=ABORTED,
            error_message=INTERRUPTED,
        )
        for (invocation_id, author, branch, scope), calls in grouped.items()
    ]
