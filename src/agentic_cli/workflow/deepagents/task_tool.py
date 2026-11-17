"""Deep Agents' ``task`` tool: the framework's delegation, as a LangChain tool.

``build_task_tool`` makes a delegating agent's ``task``. Its schema and its
description are the framework's (``agents/brief.py``: ``task_parameters``,
``task_description``), as ADK's ``TaskTool``'s are, and a call hands its
arguments, as the model wrote them, to the manager's ``Delegations``
(``agents/delegate.py``): it reads the brief, runs the sub-agent
(``DeepAgentsWorkflowManager._run_agent``, apart from this agent's stream)
and returns the result envelope. The call is answered with the envelope as
JSON, one ``ToolMessage``, and the shared state the run changed in the same
``Command``.

The shared state is the conversation's plan, tasks and outside-content mark
(``SHARED_KEYS``): a run starts from its caller's, and what it changes comes
back, as Deep Agents' own ``task`` handed it on. An inheriting type sees its
caller's effective conversation (the root's): its messages with its summary
applied, without the message that called ``task`` (``inherited_messages``).

The framework built the tool, so it binds it to the ``task`` declaration
(``tools/_core/delegation.py``): the permission gate knows it by identity.
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

from agentic_cli.tools._core.delegation import TASK_TOOL
from agentic_cli.tools.registry import bind_tool_identity, get_registry
from agentic_cli.workflow.agents.brief import task_description, task_parameters
from agentic_cli.workflow.agents.delegate import DelegationRequest
from agentic_cli.workflow.sessions import get_active_run

if TYPE_CHECKING:
    from agentic_cli.workflow.agents.delegate import Delegations
    from agentic_cli.workflow.config import AgentConfig

#: The state a run shares with its caller (``state.AgentCliState``): it starts
#: from the caller's, and the keys it changes come back with its result.
SHARED_KEYS = ("plan", "tasks", "outside_content")


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
            json.dumps(delegated.result), tool_call_id=runtime.tool_call_id, name=TASK_TOOL
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
