"""The agent tree an ADK app runs, however it was built.

An app's agents are built from agent configs (``GoogleADKWorkflowManager``
builds them) or by ADK from a native ADK YAML (``adk_config_path``). That is
construction, and the only place the two differ: once the tree is built, the
run reads what it needs from the tree itself:

- **The agents that reply to the user** (``reply_agents``): the root, and
  every chat-mode agent it can transfer to. ``transfer_to_agent`` hands such
  an agent the conversation; it answers the user, and ADK starts the next
  turn at it. Their text is the reply, the conversation they hold is the
  user's, and a conversation their model no longer takes is the turn's
  error; only the root's is summarized, and only the root carries the memory
  index. A single-turn sub-agent is a tool of its parent's (every agent
  built from configs but the root is one): its text is its work, and the
  agents under it run inside its call.
- **The agents with after-tool callbacks of their own**
  (``own_after_tool_agents``, a native ADK YAML app's): ADK runs them only
  when no plugin answered the call, so the result cap leaves these agents'
  results whole.

A native tree is loaded when the manager validates its agents
(``load_native_root``), before any model, service or store, so its agents'
models are resolved and their services built as a config-built app's are.
Two shapes are refused there, with the agent named: a root that is not an
``LlmAgent`` (a workflow or another agent class, which ADK runs on another
path), and a task-mode agent, whose delegation ADK runs outside the tool
pipeline and keeps open across turns until the agent calls ``finish_task``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from google.adk.agents import LlmAgent

from agentic_cli.logging import Loggers
from agentic_cli.workflow.config import AgentGraphError

logger = Loggers.workflow()

#: ADK's agent modes (``google.adk.utils._agent_mode.AgentMode``), as values.
_TASK = "task"
_SINGLE_TURN = "single_turn"


def walk(agent: Any) -> Iterator[Any]:
    """``agent`` and every agent under it, parents before their sub-agents."""
    yield agent
    for sub_agent in getattr(agent, "sub_agents", None) or ():
        yield from walk(sub_agent)


def _mode(agent: Any) -> str | None:
    """An agent's ADK mode as a plain value (``chat``, ``task``, ``single_turn``), or None."""
    mode = getattr(agent, "mode", None)
    return getattr(mode, "value", mode)


def load_native_root(path: str | Path) -> LlmAgent:
    """Build a native ADK YAML app's agents with ADK's ``from_config``, and check them.

    A bare ``claude-*`` model the YAML names is resolved by ADK's model
    registry, which this points at Anthropic's API first
    (``anthropic_llm.ensure_direct_anthropic``).

    Raises:
        AgentGraphError: The root is not an ``LlmAgent``, or an agent is in
            task mode.
    """
    from agentic_cli.workflow.adk.anthropic_llm import ensure_direct_anthropic
    from agentic_cli.workflow.adk_config_bridge import load_adk_agent_native

    ensure_direct_anthropic()
    root = load_adk_agent_native(path)
    if not isinstance(root, LlmAgent):
        raise AgentGraphError(
            f"the root agent {getattr(root, 'name', None)!r} is a {type(root).__name__}, "
            "not an LlmAgent: a native ADK YAML app's root must be an LlmAgent"
        )
    for agent in walk(root):
        if _mode(agent) == _TASK:
            raise AgentGraphError(
                f"agent {agent.name!r} is in task mode, which is not supported; "
                "use single_turn or chat"
            )
    logger.info("adk_native_config_loaded", path=str(path), root=root.name)
    return root


def reply_agents(root: Any) -> frozenset[str]:
    """The agents that reply to the user: ``root`` and every agent it can transfer to.

    An agent ADK runs as a tool of its parent's (single-turn, or task mode)
    is left out, with every agent under it: they run inside that call. Any
    other agent is reached by a transfer (a chat-mode ``LlmAgent``, ADK's
    default for a sub-agent, or an agent that has no mode), and so is
    every agent it runs.
    """
    names = {root.name}
    for sub_agent in getattr(root, "sub_agents", None) or ():
        if _mode(sub_agent) not in (_SINGLE_TURN, _TASK):
            names |= reply_agents(sub_agent)
    return frozenset(names)


def own_after_tool_agents(root: Any) -> frozenset[str]:
    """The agents in ``root``'s tree with after-tool callbacks of their own."""
    return frozenset(
        agent.name
        for agent in walk(root)
        if getattr(agent, "canonical_after_tool_callbacks", None)
    )
