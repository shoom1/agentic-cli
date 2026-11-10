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
Three shapes are refused there, with the agent named: a root that is not an
``LlmAgent`` (a workflow or another agent class, which ADK runs on another
path); a name used twice in the tree (ADK requires unique names only among
siblings, but the run reads every agent by name, and ADK's transfer finds
the first), as a config-built graph refuses one; and a task-mode agent,
whose delegation ADK runs outside the tool pipeline and keeps open across
turns until the agent calls ``finish_task``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import EventsCompactionConfig

from agentic_cli.logging import Loggers
from agentic_cli.workflow.config import AgentGraphError

logger = Loggers.workflow()

#: ADK's agent modes (``google.adk.utils._agent_mode.AgentMode``), as values.
_TASK = "task"
_SINGLE_TURN = "single_turn"


@dataclass(frozen=True)
class ADKBuild:
    """What one initialization built an ADK app from: the tree and what the run reads of it.

    ``GoogleADKWorkflowManager._build_app`` makes it from the root agent
    alone, whoever built the tree, through the manager's hooks
    (``_plugin_chain`` and ``_init_plugins``, ``_compaction_config``,
    ``_context_cache_config``, ``_agent_model_refs``), so a subclass's
    override of any of them applies to every app.

    Attributes:
        root: The root agent, the main agent.
        reply_agents: The agents that reply to the user (``reply_agents``).
        plugins: The app's whole plugin chain, in order.
        compaction: How the main agent's conversation is summarized; None
            when its model has no context budget.
        cache: The app's context cache; None for no caching.
        models: Each agent's model reference, by name: what its usage and
            the raw log name.
    """

    root: LlmAgent
    reply_agents: frozenset[str]
    plugins: tuple[Any, ...]
    compaction: EventsCompactionConfig | None
    cache: ContextCacheConfig | None
    models: Mapping[str, str]


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
        AgentGraphError: The YAML cannot be loaded (ADK's error, a missing
            file, a malformed one, a field ADK does not know, a module it
            names that does not import, as the cause, the file named), the
            root is not an ``LlmAgent``, a name is used twice in the tree, or
            an agent is in task mode.
    """
    from agentic_cli.workflow.adk.anthropic_llm import ensure_direct_anthropic
    from agentic_cli.workflow.adk_config_bridge import load_adk_agent_native

    ensure_direct_anthropic()
    try:
        root = load_adk_agent_native(path)
    except AgentGraphError:
        raise
    except Exception as exc:
        # As a config-built graph's errors are: one type, naming the file.
        raise AgentGraphError(
            f"cannot load the native ADK app {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(root, LlmAgent):
        raise AgentGraphError(
            f"the root agent {getattr(root, 'name', None)!r} is a {type(root).__name__}, "
            "not an LlmAgent: a native ADK YAML app's root must be an LlmAgent"
        )
    names = [agent.name for agent in walk(root)]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise AgentGraphError(
            f"Duplicate agent name(s): {', '.join(duplicates)}. Agent names must be "
            "unique across a native ADK YAML app's tree."
        )
    for agent in walk(root):
        if _mode(agent) == _TASK:
            # A chat-mode one, ADK's default for a sub-agent: a YAML has no
            # `mode`, and the gate denies a single-turn `code:` sub-agent's
            # tool, which the framework did not build.
            raise AgentGraphError(
                f"agent {agent.name!r} is in task mode, which is not supported; "
                "use a chat-mode sub-agent"
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
