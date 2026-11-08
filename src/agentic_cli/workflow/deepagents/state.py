"""The graph state every Deep Agents agent runs on."""

from typing import Annotated, NotRequired

from langchain.agents import AgentState


def _last(old: object, new: object) -> object:
    """Last write wins."""
    return new


def _first(old: object, new: object) -> object:
    """First write wins (the channel starts as "")."""
    return old or new


class AgentCliState(AgentState):
    """LangChain's agent state, plus the plan and task state tools' keys.

    ``messages`` keeps LangChain's ``add_messages``. ``plan`` and ``tasks``
    are written by ``save_plan`` and ``save_tasks`` (the LangGraph-library
    state tools). A sub-agent receives them with its brief and its changes
    flow back to the coordinator; two sub-agents running in parallel may
    both change ``tasks`` in one step, which a key without a reducer would
    fail with ``InvalidUpdateError``. The last write wins.

    ``outside_content`` is the tool that first brought outside content into
    the conversation (``""``: none yet); a sub-agent's mark flows back with
    its other changed keys.
    """

    plan: Annotated[NotRequired[str], _last]
    tasks: Annotated[NotRequired[list[dict]], _last]
    outside_content: Annotated[NotRequired[str], _first]
