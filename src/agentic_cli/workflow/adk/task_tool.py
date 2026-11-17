"""ADK's ``task`` tool: the framework's delegation, as an ADK tool.

A delegating agent built from configs gets one ``TaskTool``, whatever its
sub-agents (its sub-agents are no ADK ``sub_agents``: each runs on a runner
of its own, ``GoogleADKWorkflowManager._run_agent``). Its declaration is the
framework's (``agents/brief.py``: ``task_parameters``, ``task_description``),
as Deep Agents' ``task``'s is, and a call hands its arguments, as the model
wrote them, to the manager's ``Delegations`` (``agents/delegate.py``): it
reads the brief, runs the sub-agent and returns the result envelope.

The call's result is the envelope as JSON text, which ADK stores as
``{"result": <text>}``: every model adapter sends that text whole (ADK's
OpenAI Responses adapter sends only a dict's ``result``, and would lose the
rest of an envelope given as a dict), as Deep Agents' ``ToolMessage`` holds
it. The text is written unescaped (``ensure_ascii=False``: an escaped
non-Latin answer is six times as long), and a long one is cut by the tool
result cap as the dict it holds (``workflow/tool_output.py``), so the parent
reads JSON still.

The shared state is the conversation's plan, tasks and outside-content mark
(``SHARED_KEYS``): a run starts from the caller's (``tool_context.state``),
and the keys it changes are written back there, so they ride on the call's
function response into the conversation's state. An inheriting type reads
the conversation (``inherited_events``).

The framework builds the tool, so it binds it to the ``task`` declaration
(``tools/_core/delegation.py``) as it is made: the permission gate knows it
by identity, and lets it through as the exempt declaration says. Only an
exact ``TaskTool`` is bound: a subclass may run something else, and the gate
denies it, as it denies any tool the framework did not issue. What it
relies on of ADK's ``BaseTool`` (its constructor's ``name`` and
``description``, ``_get_declaration``, ``run_async(args=, tool_context=)``)
is contract-tested.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from google.adk.events import Event, EventActions
from google.adk.tools import BaseTool
from google.genai import types

from agentic_cli.tools._core.delegation import TASK_TOOL
from agentic_cli.tools.registry import bind_tool_identity, get_registry
from agentic_cli.workflow.agents.brief import task_description, task_parameters
from agentic_cli.workflow.agents.delegate import DelegationRequest
from agentic_cli.workflow.outside_content import OUTSIDE_CONTENT_KEY
from agentic_cli.workflow.sessions import get_active_run

if TYPE_CHECKING:
    from google.adk.tools.tool_context import ToolContext

    from agentic_cli.workflow.agents.delegate import Delegations
    from agentic_cli.workflow.config import AgentConfig

#: The state a run shares with its caller: the plan and the tasks (the state
#: tools', ``tools/adk/state_tools.py``) and the outside-content mark.
SHARED_KEYS = ("plan", "tasks", OUTSIDE_CONTENT_KEY)


class TaskTool(BaseTool):
    """A delegating agent's ``task``, which starts a run of one of its sub-agents.

    Args:
        agent: The delegating agent's name.
        types: Its sub-agents, by name: what ``agent_type`` may name.
        delegations: The manager's delegations, asked for at each call.
    """

    def __init__(
        self,
        agent: str,
        types: Mapping[str, AgentConfig],
        delegations: Callable[[], Delegations],
    ) -> None:
        super().__init__(name=TASK_TOOL, description=task_description(types, delegations().policy))
        self._agent = agent
        self._parameters = task_parameters(types)
        self._delegations = delegations
        if type(self) is TaskTool:
            bind_tool_identity(self, get_registry().get(TASK_TOOL))

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self._parameters,
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
