"""ADK's ``task`` tool: the framework's delegation, as an ADK tool.

An agent built from configs that can have sub-agents (its own, or types it
may start at run time: ``agents/spawn.py::can_have_children``) gets one
``TaskTool``, whatever its sub-agents (they are no ADK ``sub_agents``: each
runs on a runner of its own, ``GoogleADKWorkflowManager._run_agent``). Its
declaration is the framework's, made for each request (ADK asks the tool
for it per model call, in the run's own context): what the run that asks
may start (``Delegations.task_offer``, ``agents/brief.py``'s
``task_parameters`` and ``task_description``), as Deep Agents' ``task``'s is,
and none at all for a run that may start nothing: ADK adds a tool to a
request only for a declaration, so the model is offered no ``task``, and a
call of the name gets ADK's answer for a tool it does not have. A call
hands its arguments, as the model wrote them, to the manager's
``Delegations`` (``agents/delegate.py``): it reads the brief, runs the sub-agent and returns the result envelope.

The call's result is the envelope as JSON text, which ADK stores as
``{"result": <text>}``: every model adapter sends that text whole (ADK's
OpenAI Responses adapter sends only a dict's ``result``, and would lose the
rest of an envelope given as a dict), as Deep Agents' ``ToolMessage`` holds
it. The text is written unescaped (``ensure_ascii=False``: an escaped
non-Latin answer is six times as long), and a long one is cut by the tool
result cap as the dict it holds (``workflow/tool_output.py``), so the parent
reads JSON still.

A run's plan and task list are its own: it starts with none, and what it
saves stays with it (``RunAnswer.plan``, ``RunAnswer.tasks``), so the
conversation's are the delegating agent's alone. The one shared key is the
outside-content mark (``SHARED_KEYS``): a run starts from the caller's
(``tool_context.state``) and, when it changes it, writes it back there, so it
rides on the call's function response into the conversation's state. An
inheriting type reads the conversation (``inherited_events``).

The framework builds the tool, so it binds it to the ``task`` declaration
(``tools/_core/delegation.py``) as it is made: the permission gate knows it
by identity, and lets it through as the exempt declaration says. Only an
exact ``TaskTool`` is bound: a subclass may run something else, and the gate
denies it, as it denies any tool the framework did not issue. What it
relies on of ADK's ``BaseTool`` (its constructor's ``name`` and
``description``, ``_get_declaration``, ``run_async(args=, tool_context=)``)
is contract-tested.

With ``SubAgentPolicy.follow_ups`` a delegating agent also gets a
``FollowUpTool`` for each of the parent's other tools (``message_agent``,
``stop_agent``, ``check_agents``): the same kind of tool, bound to its own
declaration, whose call hands its arguments to ``call_follow_up_tool``
(``agents/follow_ups.py``) and returns the result as JSON text as ``task``
does.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from google.adk.events import Event, EventActions
from google.adk.tools import BaseTool
from google.genai import types

from agentic_cli.tools._core.delegation import FOLLOW_UP_TOOLS, TASK_TOOL
from agentic_cli.tools.registry import bind_tool_identity, get_registry
from agentic_cli.workflow.agents.brief import (
    follow_up_description,
    follow_up_parameters,
)
from agentic_cli.workflow.agents.delegate import DelegationRequest
from agentic_cli.workflow.agents.follow_ups import call_follow_up_tool
from agentic_cli.workflow.outside_content import OUTSIDE_CONTENT_KEY
from agentic_cli.workflow.sessions import get_active_run

if TYPE_CHECKING:
    from google.adk.tools.tool_context import ToolContext

    from agentic_cli.workflow.agents.delegate import Delegations

#: The state a run shares with its caller: the outside-content mark alone (a
#: security mark: memory writes in a marked conversation are approved per
#: call). The plan and the tasks (the state tools', ``tools/adk/state_tools.py``)
#: are each run's own.
SHARED_KEYS = (OUTSIDE_CONTENT_KEY,)


class TaskTool(BaseTool):
    """An agent's ``task``, which starts a run of a type the calling run may start.

    Its ``description`` is the offer at the agent's first depth (the root's
    0, any other agent's 1); the declaration each request carries is the
    offer for the run that asks.

    Args:
        agent: The agent's name.
        delegations: The manager's delegations, asked for at each call.
        root: Whether the agent is the root.
    """

    def __init__(
        self,
        agent: str,
        delegations: Callable[[], Delegations],
        *,
        root: bool = True,
    ) -> None:
        offer = delegations().task_offer(agent, 0 if root else 1, warn=True)
        super().__init__(
            name=TASK_TOOL, description=offer.description if offer is not None else ""
        )
        self._agent = agent
        self._delegations = delegations
        if type(self) is TaskTool:
            bind_tool_identity(self, get_registry().get(TASK_TOOL))

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        run = get_active_run()
        offer = self._delegations().task_offer(self._agent, run.depth if run is not None else 0)
        if offer is None:
            return None
        return types.FunctionDeclaration(
            name=self.name,
            description=offer.description,
            parameters_json_schema=offer.parameters,
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: "ToolContext") -> str:
        state = tool_context.state
        session = tool_context.session
        delegated = await self._delegations().delegate(DelegationRequest(
            caller=get_active_run(),
            caller_agent=self._agent,
            call_id=tool_context.function_call_id or "",
            arguments=args,
            state={key: state[key] for key in SHARED_KEYS if key in state},
            history=lambda: inherited_events(session.events),
        ))
        for key, value in delegated.state.items():
            state[key] = value
        return json.dumps(delegated.result, ensure_ascii=False)


class FollowUpTool(BaseTool):
    """One of a delegating agent's tools for the sub-agents it started: ``message_agent``,
    ``stop_agent`` or ``check_agents``.

    Args:
        name: Which tool (one of ``FOLLOW_UP_TOOLS``).
        delegations: The manager's delegations, asked for at each call.

    Raises:
        ValueError: ``name`` is not one of the parent's tools.
    """

    def __init__(self, name: str, delegations: Callable[[], Delegations]) -> None:
        if name not in FOLLOW_UP_TOOLS:
            raise ValueError(f"{name!r} is not one of the parent's tools.")
        super().__init__(name=name, description=follow_up_description(name))
        self._parameters = follow_up_parameters(name)
        self._delegations = delegations
        if type(self) is FollowUpTool:
            bind_tool_identity(self, get_registry().get(name))

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self._parameters,
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: "ToolContext") -> str:
        state = tool_context.state
        delegated = await call_follow_up_tool(
            self._delegations(), self.name, args,
            call_id=tool_context.function_call_id,
            state={key: state[key] for key in SHARED_KEYS if key in state},
        )
        for key, value in delegated.state.items():
            state[key] = value
        return json.dumps(delegated.result, ensure_ascii=False)


def follow_up_tools(delegations: Callable[[], Delegations]) -> list[FollowUpTool]:
    """The parent's tools, in ``FOLLOW_UP_TOOLS`` order."""
    return [FollowUpTool(name, delegations) for name in FOLLOW_UP_TOOLS]


def inherited_events(events: Sequence[Event]) -> list[Event]:
    """The conversation as an inheriting run reads it: copies of its root-visible events.

    Each event the conversation's own agents and user wrote (no branch) that
    shows anything (content, or a summary: a compaction, which keeps its
    range), with its author, invocation and time, so ADK shows the run the
    conversation as it shows any other agent's: the user's turns, the
    other agents' turns quoted as context, without their thoughts, a summary
    in place of what it covers. The copies change no state.
    """
    copied: list[Event] = []
    for event in events:
        compaction = event.actions.compaction if event.actions else None
        if event.branch or (event.content is None and compaction is None):
            continue
        copied.append(Event(
            invocation_id=event.invocation_id,
            author=event.author,
            timestamp=event.timestamp,
            content=event.content.model_copy(deep=True) if event.content is not None else None,
            actions=EventActions(
                compaction=compaction.model_copy(deep=True) if compaction is not None else None
            ),
        ))
    return copied
