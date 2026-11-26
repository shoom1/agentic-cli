"""The graph state every Deep Agents agent runs on."""

from typing import Annotated, NotRequired

from deepagents import DeepAgentState


def _last(old: object, new: object) -> object:
    """Last write wins."""
    return new


def _first(old: object, new: object) -> object:
    """First write wins (the channel starts as "")."""
    return old or new


class AgentCliState(DeepAgentState):
    """Deep Agents' agent state, plus the plan and task state tools' keys.

    ``messages`` is Deep Agents' delta channel: each checkpoint stores the
    messages its step added (a full copy every 50 updates), and LangGraph
    puts the list together from the thread's writes as it reads a
    checkpoint, so a conversation's checkpoints grow with its length, not
    its square (``CheckpointSessions`` reads them so too). ``plan`` and ``tasks``
    are written by ``save_plan`` and ``save_tasks`` (the LangGraph-library
    state tools). A sub-agent has its own, starting empty, and none flows
    back to the coordinator; one agent's parallel ``save_tasks`` calls in
    one step may both change ``tasks``, which a key without a reducer would
    fail with ``InvalidUpdateError``. The last write wins.

    ``outside_content`` is the tool that first brought outside content into
    the conversation (``""``: none yet); a sub-agent's mark flows back
    alone, as the one key a run shares with its caller.
    """

    plan: Annotated[NotRequired[str], _last]
    tasks: Annotated[NotRequired[list[dict]], _last]
    outside_content: Annotated[NotRequired[str], _first]
