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
a cancel from before this plugin, or a call ADK took for answered (its seals
pair a result with a call by id alone, so a sub-agent's result answers its
parent's delegation of the same id, which a provider that numbers its call
ids per agent gives it). ADK would keep a long-running one without its
result in every later request, which Claude and OpenAI refuse, and drop any
other one, logging "Dropping function calls" on every request.

Each call is recorded and answered as the run's that made it, the run its
branch names (``workflow/adk/branches.py``; the root's own calls are no
run), so a sub-agent's call never stands for its parent's of the same id
(``OpenCalls``).

ADK stops at the first plugin whose ``after_tool_callback`` returns a value,
so this one is registered before ``ToolResultCapPlugin``, which tells it each
cut it makes (``ToolResultCapPlugin.on_cut``): a finished result it answers
with is the cap's own cut, whose full output was saved once.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from google.adk.events import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from agentic_cli.workflow.adk.branches import branch_run
from agentic_cli.workflow.adk.tool_error_plugin import is_unknown_tool
from agentic_cli.workflow.agents.runs import AgentRuns
from agentic_cli.workflow.interrupts import INTERRUPTED, MARKER, OpenCalls, unrecorded
from agentic_cli.workflow.sessions import (
    ToolCallRef,
    get_active_turn,
    reset_active_tool_call,
    set_active_tool_call,
)

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext

    from agentic_cli.workflow.adk.tool_result_cap_plugin import ToolResultCapPlugin
    from agentic_cli.workflow.sessions import RunRef

#: ADK's ``error_code`` on the events that seal an aborted invocation
#: (``google/adk/events/_abort_events.py``, private; contract-tested).
ABORTED = "INVOCATION_ABORTED"


@dataclass(frozen=True, slots=True)
class _Finished:
    """A call that returned before the cancel: its result, the cap's cut of
    it (None when the cap left it as it was), and the state it changed (ADK
    stores a call's state change with its result, which a seal replaces:
    save_tasks would say it saved, and not)."""

    result: Any
    state_delta: dict[str, Any]
    artifact_delta: dict[str, Any]
    cut: Any = None


class InterruptedCallsPlugin(BasePlugin):
    """Answer the calls a cancelled turn leaves open (``workflow/interrupts.py``).

    Args:
        open_calls: The manager's record of its calls.
        cap: The tool result cap, registered after this plugin: it tells
            this plugin each cut it makes, which a finished call's seal
            sends.
        runs: The manager's runs, which name the run a seal's branch is
            (the names the active run has, ``MainAgentPlugin``'s); a
            registry of its own when None.
    """

    def __init__(
        self,
        open_calls: OpenCalls,
        cap: "ToolResultCapPlugin | None" = None,
        *,
        runs: AgentRuns | None = None,
    ) -> None:
        super().__init__(name="interrupted_calls")
        self._open = open_calls
        self._cap = cap
        self._runs = runs if runs is not None else AgentRuns()
        if cap is not None:
            cap.on_cut = self._keep_cut
        # Each running call's active-tool-call token, by call id.
        self._active: dict[str, Any] = {}

    async def before_tool_callback(
        self, *, tool: "BaseTool", tool_args: dict[str, Any], tool_context: "ToolContext | None"
    ) -> None:
        """Mark the call running (``workflow.sessions.get_active_tool_call``):
        a question it asks then knows which call waits on it.

        ADK's stand-in for an unknown tool name runs nothing, and ADK skips
        the after-tool callbacks for it, which end the mark: it gets none.
        """
        call_id = getattr(tool_context, "function_call_id", None)
        if call_id and not is_unknown_tool(tool):
            self._active[call_id] = set_active_tool_call(ToolCallRef(call_id, tool.name))
        return None

    async def after_tool_callback(
        self,
        *,
        tool: "BaseTool",
        tool_args: dict[str, Any],
        tool_context: "ToolContext | None",
        result: Any,
    ) -> None:
        if tool_context is None:
            return None
        self._done(tool_context.function_call_id)
        if result is None and getattr(tool, "is_long_running", False):
            # ADK gives a long-running call that returned nothing no result:
            # its job answers it (OpenCalls.answer), never null.
            return None
        actions = tool_context.actions
        self._open.finished(tool_context.function_call_id, _Finished(
            result, dict(actions.state_delta or {}), dict(actions.artifact_delta or {}),
        ))
        return None

    def _keep_cut(self, call_id: str, cut: Any) -> None:
        """Keep the cap's cut of a finished call's result on its record.

        The cap runs right after ``after_tool_callback`` recorded the call,
        in the same turn: nothing awaits between them.
        """
        finished = self._open.recorded(call_id)
        if isinstance(finished, _Finished):
            self._open.finished(call_id, replace(finished, cut=cut))

    def _done(self, call_id: str | None) -> None:
        token = self._active.pop(call_id, None) if call_id else None
        if token is not None:
            try:
                reset_active_tool_call(token)
            except ValueError:  # set in another context: it ends with that context
                pass

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
        actions = event.actions.model_copy(deep=True)
        ref = self._branch_run(event.branch)
        run = ref.run if ref is not None else None
        parts = [await self._answered(part, actions, run) for part in content.parts]
        return event.model_copy(update={
            "content": types.Content(role=content.role, parts=parts), "actions": actions,
        })

    def _branch_run(self, branch: str | None) -> "RunRef | None":
        """The run ``branch`` is (None for the root's own calls).

        A seal is written outside the run whose call it answers (in the
        runner's own context), so its run is its branch's, not the active
        one. Asked while that run is open: ADK seals every branch of a
        stopped turn before any seal reaches the event processor that ends
        its run.
        """
        return branch_run(self._runs, get_active_turn(), branch)

    async def _answered(self, part: types.Part, actions: Any, run: str | None) -> types.Part:
        call = part.function_response
        if call is None:
            return part
        answer = self._open.answer(call.id, by_user=True, run=run)
        if isinstance(answer, _Finished):
            # Its state change is stored with the seal, as with its result.
            actions.state_delta.update(answer.state_delta)
            actions.artifact_delta.update(answer.artifact_delta)
            answer = await self._capped(answer)
        return types.Part(function_response=types.FunctionResponse(
            id=call.id, name=call.name, response=_as_response(answer),
        ))

    async def _capped(self, finished: _Finished) -> Any:
        """What the model would have read for ``finished``: the cap's cut, or
        the result as it was. Never cut again, which would save its full
        output a second time."""
        return finished.result if finished.cut is None else finished.cut


def _as_response(result: Any) -> dict[str, Any]:
    """A result as ADK stores it: a dict as it is, anything else under ``result``."""
    return result if isinstance(result, dict) else {"result": result}


def left_open(events: list[Event], open_calls: OpenCalls) -> list[Event]:
    """The events that answer the calls nothing answered: a crash's, or a cancel's ADK missed.

    A result answers only a call on its own branch: a sub-agent's result
    never answers its parent's delegation of the same id. Each call left
    open is answered as no record holds it (``interrupts.unrecorded``: a
    job's state, else ``INTERRUPTED_BY_USER`` when the user stopped the
    conversation's last turn, ``OpenCalls.take_cancel``, which this asks
    once, else ``INTERRUPTED``): every turn clears its conversation's record
    as it ends (``OpenCalls.clear``), and a new process has none. The answer
    is in the author, branch, scope and invocation of the event that made
    the call, so it pairs with its call in that agent's view; calls that
    share those are answered in one event, marked as ADK marks a seal.
    """
    by_user = open_calls.take_cancel(get_active_turn())
    answered = {
        (event.branch, response.id) for event in events
        for response in event.get_function_responses() if response.id
    }
    grouped: dict[tuple, list[types.FunctionCall]] = {}
    for event in events:
        for call in event.get_function_calls():
            if call.id and (event.branch, call.id) not in answered:
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
                    response=_as_response(unrecorded(call.id, by_user=by_user)),
                ))
                for call in calls
            ]),
            error_code=ABORTED,
            error_message=INTERRUPTED,
        )
        for (invocation_id, author, branch, scope), calls in grouped.items()
    ]
