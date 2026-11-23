"""Deep Agents' ``task`` tool: the framework's delegation, as a LangChain tool.

``build_task_tool`` makes the ``task`` of an agent that can have sub-agents
(its own, or types it may start at run time: ``agents/spawn.py``). Its
schema and its description are the framework's, made for the types the run
that calls it may start (``Delegations.task_offer``: ``agents/brief.py``'s
``task_parameters`` and ``task_description``), as ADK's ``TaskTool``'s are:
the root's are fixed when it is built (depth 0), a sub-agent's are rewritten
for each of its model calls (``RunControlMiddleware``), and a call from a run
that may start nothing gets the answer for a tool it does not have. A call
hands its arguments, as the model wrote them, to the manager's
``Delegations`` (``agents/delegate.py``): it reads the brief, runs the sub-agent
(``DeepAgentsWorkflowManager._run_agent``, apart from this agent's stream)
and returns the result envelope. The call is answered with the envelope as
JSON (unescaped, as ADK's), one ``ToolMessage``, and the shared state the run
changed in the same ``Command``.

A run's plan and task list are its own: it starts with none, and what it
saves stays with it (``RunAnswer.plan``, ``RunAnswer.tasks``), so the
conversation's are the delegating agent's alone. The one shared key is the
outside-content mark (``SHARED_KEYS``): a run starts from its caller's, and
what it changes comes back. An inheriting type sees its caller's effective
conversation (the root's): its messages with its summary applied, without
the message that called ``task`` (``inherited_messages``).

The framework built the tool, so it binds it to the ``task`` declaration
(``tools/_core/delegation.py``): the permission gate knows it by identity.

With ``SubAgentPolicy.follow_ups`` a delegating agent also gets the parent's
other tools (``build_follow_up_tools``: ``message_agent``, ``stop_agent``,
``check_agents``), each bound to its own declaration and answering as
``task`` does: its result as JSON, one ``ToolMessage``, the shared state a
follow-up run changed in the same ``Command``.

A ``load_skill`` call of a skill that names an agent type starts that run here
too (``start_skill_run``, from ``tools/skills/langchain_tools.py``), answered
as ``task`` is.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from deepagents.middleware.summarization import (
    SUMMARIZATION_EVENT_KEY,
    SummarizationMiddleware,
)
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from agentic_cli.logging import Loggers
from agentic_cli.tools._core.delegation import FOLLOW_UP_TOOLS, LOAD_SKILL_TOOL, TASK_TOOL
from agentic_cli.tools.registry import bind_tool_identity, get_registry
from agentic_cli.workflow.agents.brief import (
    follow_up_description,
    follow_up_parameters,
    task_parameters,
)
from agentic_cli.workflow.agents.delegate import DelegationRequest
from agentic_cli.workflow.agents.follow_ups import call_follow_up_tool
from agentic_cli.workflow.deepagents.middleware import error_tool_message
from agentic_cli.workflow.sessions import get_active_run
from agentic_cli.workflow.tool_errors import unknown_tool_result

if TYPE_CHECKING:
    from agentic_cli.tools.skills.agents import SkillAgents
    from agentic_cli.workflow.agents.delegate import Delegations

logger = Loggers.workflow()

#: The state a run shares with its caller (``state.AgentCliState``): the
#: outside-content mark alone, which starts from the caller's and comes back
#: with the run's result when it changed. The plan and the tasks are each
#: run's own.
SHARED_KEYS = ("outside_content",)


def build_task_tool(
    agent: str, delegations: Callable[[], Delegations], *, root: bool = True, warn: bool = True
) -> StructuredTool:
    """``agent``'s ``task`` tool, which starts a run of a type the calling run may start.

    Its schema and description are the offer at the agent's first depth (the
    root's 0, any other agent's 1), made once with the warnings; a
    sub-agent's are rewritten for each model call.

    Args:
        agent: The delegating agent's name.
        delegations: The manager's delegations, asked for at each call.
        root: Whether the agent is the root.
        warn: Whether the offer warns of what it lacks (a type offered with no
            description): False for a model choice's graph of its type, whose
            own build warned.
    """
    offer = delegations().task_offer(agent, 0 if root else 1, warn=warn)

    async def task(runtime: ToolRuntime, **arguments: Any) -> Command:
        state = runtime.state
        caller = get_active_run()
        if not delegations().offered(agent, caller.depth if caller is not None else 0):
            # A run that may start nothing is offered no task: a call of the
            # name gets the answer for a tool it does not have, as on ADK.
            logger.info("tool_not_found", tool=TASK_TOOL, agent=agent)
            others = [tool.name for tool in runtime.tools or () if tool.name != TASK_TOOL]
            call = {"name": TASK_TOOL, "args": arguments, "id": runtime.tool_call_id}
            answer = error_tool_message(unknown_tool_result(TASK_TOOL, others), call)
            return Command(update={"messages": [answer]})
        delegated = await delegations().delegate(DelegationRequest(
            caller=caller,
            caller_agent=agent,
            call_id=runtime.tool_call_id,
            arguments=arguments,
            state={key: state[key] for key in SHARED_KEYS if key in state},
            history=lambda: inherited_messages(state),
        ))
        answer = ToolMessage(
            json.dumps(delegated.result, ensure_ascii=False),
            tool_call_id=runtime.tool_call_id,
            name=TASK_TOOL,
        )
        return Command(update={**delegated.state, "messages": [answer]})

    tool = StructuredTool.from_function(
        coroutine=task,
        name=TASK_TOOL,
        description=offer.description if offer is not None else "",
        args_schema=offer.parameters if offer is not None else task_parameters({}),
        infer_schema=False,
    )
    bind_tool_identity(tool, get_registry().get(TASK_TOOL))
    return tool


async def start_skill_run(agents: SkillAgents, skill: Any, runtime: ToolRuntime) -> Command:
    """A ``load_skill`` of ``skill``, which names an agent type: the run it starts.

    Answered as ``task`` is: one ``ToolMessage`` with the envelope (or the
    refusal) as JSON, and the shared state the run changed in the same
    ``Command``.

    Args:
        agents: The loading agent's ``SkillAgents``.
        skill: The skill, an ADK ``Skill``.
        runtime: The call's runtime, which the tool node injects.
    """
    state = runtime.state
    delegated = await agents.start(
        skill,
        call_id=runtime.tool_call_id,
        state={key: state[key] for key in SHARED_KEYS if key in state},
        history=lambda: inherited_messages(state),
    )
    answer = ToolMessage(
        json.dumps(delegated.result, ensure_ascii=False),
        tool_call_id=runtime.tool_call_id,
        name=LOAD_SKILL_TOOL,
    )
    return Command(update={**delegated.state, "messages": [answer]})


def build_follow_up_tools(delegations: Callable[[], Delegations]) -> list[StructuredTool]:
    """The parent's tools (``message_agent``, ``stop_agent``, ``check_agents``), in
    ``FOLLOW_UP_TOOLS`` order.

    Args:
        delegations: The manager's delegations, asked for at each call.
    """
    return [_follow_up_tool(name, delegations) for name in FOLLOW_UP_TOOLS]


def _follow_up_tool(name: str, delegations: Callable[[], Delegations]) -> StructuredTool:
    """One of the parent's tools, bound to its declaration."""

    async def call(runtime: ToolRuntime, **arguments: Any) -> Command:
        state = runtime.state
        delegated = await call_follow_up_tool(
            delegations(), name, arguments,
            call_id=runtime.tool_call_id,
            state={key: state[key] for key in SHARED_KEYS if key in state},
        )
        answer = ToolMessage(
            json.dumps(delegated.result, ensure_ascii=False),
            tool_call_id=runtime.tool_call_id,
            name=name,
        )
        return Command(update={**delegated.state, "messages": [answer]})

    tool = StructuredTool.from_function(
        coroutine=call,
        name=name,
        description=follow_up_description(name),
        args_schema=follow_up_parameters(name),
        infer_schema=False,
    )
    bind_tool_identity(tool, get_registry().get(name))
    return tool


def inherited_messages(state: Mapping[str, Any]) -> list[AnyMessage]:
    """The conversation an inheriting run reads: ``state``'s messages as its caller's model sees them.

    The caller's summary applied (Deep Agents'
    ``SummarizationMiddleware._apply_event_to_messages``, private: a contract
    test pins it), without the caller's last message when it is the one
    that called ``task`` (an AI message with tool calls), as a Deep Agents
    fork's history is.
    """
    messages = list(state.get("messages") or ())
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        messages.pop()
    return SummarizationMiddleware._apply_event_to_messages(
        messages, state.get(SUMMARIZATION_EVENT_KEY)
    )
