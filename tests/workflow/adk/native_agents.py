"""Agents and callbacks a native ADK YAML app in the tests names by module path.

ADK resolves a ``code:`` sub-agent with ``getattr(module, name)`` and an agent
takes one parent, so each lookup builds a fresh agent (PEP 562 ``__getattr__``):
every app the tests load gets its own.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent

MODEL = "claude-sonnet-4-6"

#: What the agents' own callbacks were given, in order: (callback, agent or tool, size).
SEEN: list[tuple[str, str, int]] = []

_AGENTS = {
    # ADK runs a task-mode sub-agent's delegation itself and keeps its call
    # open until the agent calls finish_task.
    "asker": dict(instruction="You ask.", mode="task", description="Asks the user."),
    "solver": dict(instruction="You solve.", mode="single_turn", description="Solves."),
    "chatter": dict(instruction="You chat.", mode="chat", description="Chats."),
}


def __getattr__(name: str) -> Any:
    if name not in _AGENTS:
        raise AttributeError(name)
    return LlmAgent(name=name, model=MODEL, **_AGENTS[name])


def after_tool(tool, args, tool_context, tool_response):
    """An agent's own after-tool callback: it records the result's size."""
    SEEN.append(("after_tool", tool.name, len(str(tool_response))))
    return None
