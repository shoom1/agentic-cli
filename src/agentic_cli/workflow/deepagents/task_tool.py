"""Deep Agents' ``task`` tool: the framework's delegation, as a LangChain tool.

``build_task_tool`` makes a delegating agent's ``task``. Its schema and its
description are the framework's (``agents/brief.py``: ``task_parameters``,
``task_description``), as ADK's ``TaskTool``'s are, and a call hands its
arguments, as the model wrote them, to the manager's ``Delegations``
(``agents/delegate.py``): it reads the brief, runs the sub-agent
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

from agentic_cli.tools._core.delegation import FOLLOW_UP_TOOLS, TASK_TOOL
from agentic_cli.tools.registry import bind_tool_identity, get_registry
from agentic_cli.workflow.agents.brief import (
    follow_up_description,
    follow_up_parameters,
    task_description,
    task_parameters,
)
from agentic_cli.workflow.agents.delegate import DelegationRequest
from agentic_cli.workflow.agents.follow_ups import call_follow_up_tool
from agentic_cli.workflow.sessions import get_active_run

if TYPE_CHECKING:
    from agentic_cli.workflow.agents.delegate import Delegations
    from agentic_cli.workflow.config import AgentConfig

#: The state a run shares with its caller (``state.AgentCliState``): the
#: outside-content mark alone, which starts from the caller's and comes back
#: with the run's result when it changed. The plan and the tasks are each
#: run's own.
SHARED_KEYS = ("outside_content",)


def build_task_tool(
    agent: str, types: Mapping[str, AgentConfig], delegations: Callable[[], Delegations]
) -> StructuredTool:
    """``agent``'s ``task`` tool, which starts a run of one of ``types``.

    Args:
        agent: The delegating agent's name.
        types: Its sub-agents, by name: what ``agent_type`` may name.
        delegations: The manager's delegations, asked for at each call.
    """

    async def task(runtime: ToolRuntime, **arguments: Any) -> Command:
        state = runtime.state
        delegated = await delegations().delegate(DelegationRequest(
            caller=get_active_run(),
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
        description=task_description(types, delegations().policy),
        args_schema=task_parameters(types),
        infer_schema=False,
    )
    bind_tool_identity(tool, get_registry().get(TASK_TOOL))
    return tool


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
