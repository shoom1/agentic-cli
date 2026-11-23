"""ADK plugins: a sub-agent run's control at each of its model calls (``workflow/agents/control.py``).

In each sub-agent type's plugin chain (``GoogleADKWorkflowManager._agent_plugin_chain``),
for the run in progress (``get_active_run``: the run's own task,
``agents/delegate.py``), whose control the manager's ``Delegations`` holds.

``RunToolsPlugin``, first in the chain's head: before each model call, a run
narrowed to some of its type's tools (``RunControl.tools``) has the others
left out of the request and of its ``tools_dict``, which ADK resolves a call
from, so a call of one is answered as one of a tool the agent does not have
(``narrow``). An ADK built-in it was not given (``google_search``,
``url_context``: it runs inside the model, so what it adds to the request is
no function declaration and nothing in ``tools_dict``) is left out too, its
entries found by having it process a request of its own
(``built_in_entries``); one the run keeps is sent as ADK made it. It runs
before the permission gate, whose mark for the built-ins a request carries
then follows what is sent, and before the fit guard, which measures the
narrowed request; a skill toolset's instruction text stays in the system
prompt, only its calls are refused.

``RunControlPlugin``, after the raw log:

- before each model call, the messages steered to the run since its last
  (``RunControl.take_steering``) go after its last message, each a user
  event appended to the run's child session (through the callback
  context's session service: the invocation's own session, which a session
  loaded apart would fail ADK's revision check on) and a user content added
  to the request, so they stay in its transcript, each the run's stored one
  (``RunControl.stored``) once its append returns. ADK stamps the call's
  answer before this callback runs, so each event of the run from then on
  that is not later than the steered message is stamped just after it
  (``on_event_callback``): a durable store reads a session's events back by
  time, and would put the message after the answer it prompted;
- the call is counted (``RunControl.model_call``), and the run's last
  allowed one gets ``STEP_LIMIT_NOTE`` after its last message, in the
  request only: as a part of the last content when that is the user's (a
  tool result, the task message, a steered message), so no provider sees
  two user messages in a row. Its tools stay declared: Claude refuses a
  request whose messages hold tool calls or results when it defines no
  tools, and no call that answer makes is run (below);
- after each, what it used (``workflow/usage.py::adk_tokens``, by its
  model's provider kind) goes to ``RunControl.used``, which checks the
  run's total (with what its side calls used), with whether the answer
  makes function calls: one past the token limit that makes none is the
  run's result, whole. A call that reports no usage counts none of its
  own;
- on the run's last call (``RunControl.final_call``: its step limit's, or
  one past its token limit that makes calls) the answer's function calls
  are dropped, so ADK
  runs none and the run ends with that answer; an answer that was calls
  alone keeps an empty text, which ADK takes as the run's last answer
  (``MODEL_RETURNED_NO_CONTENT``, no failure for a run).

Each changes the request, the answer and an event's time in place and
returns None, as ``ForeignReasoningPlugin`` does, so every plugin after it
still runs; what ``RunControlPlugin`` keeps is kept by invocation, and let
go of as the run ends (two runs of one type share their runner, and these
plugins). It sits before ``SubAgentOutcomePlugin``, which then reads an
answer whose calls were dropped as the run's final one. The root's calls (no
active run) and a run without a control are left alone.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional

from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from agentic_cli.workflow.agents.control import STEP_LIMIT_NOTE
from agentic_cli.workflow.usage import UsageTotals, adk_tokens

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.models.llm_response import LlmResponse

    from agentic_cli.workflow.agents.control import RunControl

#: How much later than a steered message the run's next events are stamped,
#: at least: past what a durable store keeps of a time (``DatabaseSessionService``
#: keeps microseconds), so it reads them back in the order they came.
_STAMP_STEP = 2e-6


class RunToolsPlugin(BasePlugin):
    """A narrowed sub-agent run's tools at its model calls (see the module docstring).

    Args:
        control: The control of the run in progress (None: no run, or none
            to control), asked at each call.
    """

    def __init__(self, control: Callable[[], RunControl | None]) -> None:
        super().__init__(name="run_tools")
        self._control = control

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        control = self._control()
        if control is None or control.tools is None:
            return None
        built_ins = await built_in_entries(callback_context, llm_request, control.keeps)
        narrow(llm_request, control.keeps, built_ins=built_ins)
        return None


class RunControlPlugin(BasePlugin):
    """A sub-agent run's limits at its model calls (see the module docstring).

    Args:
        control: The control of the run in progress (None: no run, or none
            to control), asked at each callback.
        kind: The provider kind of the type's model (``ProviderKind``'s
            value), which its usage is read by; None when unknown.
    """

    def __init__(self, control: Callable[[], RunControl | None], kind: str | None) -> None:
        super().__init__(name="run_control")
        self._control = control
        self._kind = kind
        # Each invocation's last steered message's time, or the latest of
        # its events stamped since: its next events come after it.
        self._stamped: dict[str, float] = {}

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        control = self._control()
        if control is None:
            return None
        steered = control.take_steering()
        if steered:
            self._stamped[callback_context.invocation_id] = await _steer(
                callback_context, llm_request, steered, control
            )
        if control.model_call():
            llm_request.contents = with_note(list(llm_request.contents or []), STEP_LIMIT_NOTE)
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        if llm_response.partial:
            return None
        control = self._control()
        if control is None:
            return None
        usage = llm_response.usage_metadata
        # A call that reports no usage counts none of its own; the run's
        # side calls still count.
        tokens = adk_tokens(usage, self._kind) if usage is not None else UsageTotals()
        control.used(tokens.input, tokens.output, calls=_calls(llm_response))
        if control.final_call:
            drop_calls(llm_response)
        return None

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        # An event ADK stamped before a message was steered (the answer to
        # the call it was steered at) is stamped after it.
        last = self._stamped.get(event.invocation_id)
        if last is None:
            return None
        if event.timestamp < last + _STAMP_STEP:
            event.timestamp = last + _STAMP_STEP
        self._stamped[event.invocation_id] = event.timestamp
        return None

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        self._stamped.pop(invocation_context.invocation_id, None)
        return None

    async def on_run_error_callback(
        self, *, invocation_context: InvocationContext, error: Exception
    ) -> None:
        self._stamped.pop(invocation_context.invocation_id, None)
        return None


async def _steer(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    messages: list[str],
    control: RunControl,
) -> float:
    """``messages``, as the user's, in the run's child session and after the
    request's last content; the time the last one is stamped with.

    Each one is the run's stored one (``RunControl.stored``) as its append
    returns: a run stopped while it appends keeps the rest unstored, for the
    instance's next run."""
    service = callback_context.get_invocation_context().session_service
    session = callback_context.session
    stamped = 0.0
    for message in messages:
        content = types.Content(role="user", parts=[types.Part.from_text(text=message)])
        event = Event(
            invocation_id=callback_context.invocation_id, author="user", content=content,
        )
        await service.append_event(session, event)
        control.stored(1)
        llm_request.contents.append(content)
        stamped = event.timestamp
    return stamped


async def built_in_entries(
    callback_context: CallbackContext | None,
    llm_request: LlmRequest,
    keeps: Callable[[str], bool],
) -> dict[str, list[types.Tool]]:
    """What each of the running agent's built-ins that ``keeps`` refuses added to
    ``llm_request``, by the tool's name.

    A built-in (``google_search``, ``url_context``, a Vertex AI Search) is an
    ADK tool of the agent's own (``agent.tools``, not a toolset's) that ADK did
    not put in ``tools_dict``: it runs inside the model, and what it adds to the
    request is a tool entry that is no function declaration. Which entry is
    whose the request does not say, so each such tool processes a request of
    its own, for the same model and with a tool context of the same
    invocation, as ADK had it process this one (``process_llm_request``), at
    each model call of a narrowed run; of what it adds there, the entries that
    are more than function declarations are its. No callback context (no
    agent to ask): none.
    """
    get = getattr(callback_context, "get_invocation_context", None)
    invocation = get() if get is not None else None
    agent = getattr(invocation, "agent", None)
    found: dict[str, list[types.Tool]] = {}
    for tool in getattr(agent, "tools", None) or ():
        if (
            not isinstance(tool, BaseTool)
            or keeps(tool.name)
            or tool.name in llm_request.tools_dict
        ):
            continue
        alone = LlmRequest(model=llm_request.model)
        await tool.process_llm_request(
            tool_context=ToolContext(invocation), llm_request=alone
        )
        found[tool.name] = [
            entry for entry in alone.config.tools or ()
            if entry.model_dump(exclude={"function_declarations"}, exclude_none=True)
        ]
    return found


def narrow(
    llm_request: LlmRequest,
    keeps: Callable[[str], bool],
    *,
    built_ins: Mapping[str, Sequence[types.Tool]] | None = None,
) -> None:
    """Leave the tools ``keeps`` refuses out of ``llm_request``, in place.

    Out of ``tools_dict`` (what ADK resolves a call from: a call of a name
    left out gets ADK's answer for a tool that is not there) and out of the
    declarations the model is sent. ``built_ins`` are the entries each
    built-in added, by its name (``built_in_entries``): one ``keeps`` refuses
    has each of its entries left out, once; one it keeps stays as ADK made
    it. A tool left with no declaration and nothing else goes; an entry no
    built-in made (a code executor's) stays as it is.
    """
    for name in [name for name in llm_request.tools_dict if not keeps(name)]:
        del llm_request.tools_dict[name]
    dropped = [
        entry for name, entries in (built_ins or {}).items() if not keeps(name)
        for entry in entries
    ]
    kept: list[types.Tool] = []
    for tool in llm_request.config.tools or ():
        if tool in dropped:
            dropped.remove(tool)
            continue
        declarations = tool.function_declarations
        if declarations is None:
            kept.append(tool)
            continue
        left = [d for d in declarations if d.name is None or keeps(d.name)]
        if len(left) == len(declarations):
            kept.append(tool)
        elif left or tool.model_dump(
            exclude={"function_declarations"}, exclude_none=True
        ):
            kept.append(tool.model_copy(update={"function_declarations": left or None}))
    llm_request.config.tools = kept or None


def with_note(contents: list[types.Content], note: str) -> list[types.Content]:
    """``contents`` with ``note`` after their last message, as the user's.

    A part of the last content when that is the user's (a tool result, the
    task message, a steered message); else a user content of its own.
    ``contents`` are not changed in place.
    """
    text = types.Part.from_text(text=note)
    if contents and contents[-1].role == "user":
        last = contents[-1]
        return [*contents[:-1], last.model_copy(update={"parts": [*(last.parts or []), text]})]
    return [*contents, types.Content(role="user", parts=[text])]


def _calls(llm_response: Any) -> bool:
    """Whether ``llm_response`` makes function calls: the run would go on past it."""
    content = llm_response.content
    return content is not None and any(part.function_call for part in content.parts or ())


def drop_calls(llm_response: Any) -> None:
    """Drop ``llm_response``'s function calls, in place: it is the run's last answer.

    Its text and thoughts are kept; an answer that was calls alone keeps an
    empty text, so ADK takes it as an answer (none is not one: ADK would
    call the model again).
    """
    content = llm_response.content
    if content is None or not any(part.function_call for part in content.parts or ()):
        return
    kept = [part for part in content.parts or () if not part.function_call]
    if not any(part.text is not None and not part.thought for part in kept):
        kept.append(types.Part.from_text(text=""))
    llm_response.content = types.Content(role=content.role or "model", parts=kept)
