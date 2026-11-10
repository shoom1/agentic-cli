"""Task progress is reported when the task list changes, and the list is kept.

On both backends a ``TASK_PROGRESS`` event follows the tool result that
wrote a changed task list (``save_tasks``), within a turn; a tool call that
changes nothing (another tool, ``get_tasks``, a write of the same list)
reports nothing. A list whose tasks are all done stays: ``get_tasks`` answers
it, in that turn and the next, until the model writes another.

The real managers run on Anthropic's API faked offline
(``tests/workflow/test_failed_turn.py``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("deepagents")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.events import EventType, WorkflowEvent  # noqa: E402
from tests.workflow.test_failed_turn import (  # noqa: E402, F401
    BACKENDS,
    CLAUDE,
    Script,
    calls,
    ft_lookup,
    hermetic,
    say,
    serve,
)


def task(task_id: str, description: str, status: str) -> dict:
    """A task as ``save_tasks`` keeps it, every field given, so writing it
    twice writes the same list."""
    return {
        "id": task_id, "description": description, "status": status,
        "priority": "medium", "tags": [], "created_at": "2026-11-10T09:00:00",
        "completed_at": "2026-11-10T09:30:00" if status == "completed" else "",
    }


STARTED = [task("a", "Research", "in_progress"), task("b", "Write", "pending")]
DONE = [task("a", "Research", "completed"), task("b", "Write", "completed")]


def planner(backend: str):
    """A main agent with the plan and task tools and one lookup tool."""
    configs = [AgentConfig(name="solo", prompt="Answer.", tools=[ft_lookup])]
    settings = BaseSettings(
        default_model=CLAUDE, session_store="memory", retry_max_attempts=0,
        prompt_caching_enabled=False,
    )
    if backend == "adk":
        return GoogleADKWorkflowManager(agent_configs=configs, settings=settings)
    return DeepAgentsWorkflowManager(agent_configs=configs, settings=settings)


async def turn(manager, text: str) -> list[WorkflowEvent]:
    return [event async for event in manager.process(text, "u", session_id="s")]


def steps(events: list[WorkflowEvent]) -> list[str]:
    """Each tool result (by tool) and each progress report (by what is done), in order."""
    out: list[str] = []
    for event in events:
        if event.type is EventType.TOOL_RESULT:
            out.append(event.metadata["tool_name"])
        elif event.type is EventType.TASK_PROGRESS:
            progress = event.metadata["progress"]
            out.append(f"progress {progress['completed']}/{progress['total']}")
    return out


def listed(events: list[WorkflowEvent]) -> list[list[str]]:
    """What each ``get_tasks`` answered: its tasks' statuses."""
    return [
        [t["status"] for t in event.metadata["result"]["tasks"]]
        for event in events
        if event.type is EventType.TOOL_RESULT and event.metadata["tool_name"] == "get_tasks"
    ]


@pytest.mark.parametrize("backend", BACKENDS)
async def test_progress_follows_each_change_only_and_the_done_list_stays(
    hermetic, monkeypatch, backend
):
    serve(monkeypatch, Script(
        calls("save_tasks", {"tasks": STARTED}, "toolu_1"),
        calls("ft_lookup", {"topic": "X"}, "toolu_2"),
        calls("save_tasks", {"tasks": STARTED}, "toolu_3"),
        calls("save_tasks", {"tasks": DONE}, "toolu_4"),
        calls("get_tasks", {}, "toolu_5"),
        say("Both done."),
        calls("get_tasks", {}, "toolu_6"),
        say("Still done."),
    ), backend)
    manager = planner(backend)
    try:
        first = await turn(manager, "Research, then write.")
        second = await turn(manager, "Where are we?")
    finally:
        await manager.cleanup()

    assert steps(first) == [
        "save_tasks", "progress 0/2",
        "ft_lookup",
        "save_tasks",  # the same list: nothing changed
        "save_tasks", "progress 2/2",
        "get_tasks",
    ]
    assert listed(first) == [["completed", "completed"]]
    assert steps(second) == ["get_tasks"]
    assert listed(second) == [["completed", "completed"]]


@pytest.mark.parametrize("backend", BACKENDS)
async def test_each_turn_reports_the_first_list_it_writes(hermetic, monkeypatch, backend):
    """A turn reports the task list it first writes, the last turn's included:
    "changed" is within the turn, as Deep Agents' processor (one a turn) has it."""
    serve(monkeypatch, Script(
        calls("save_tasks", {"tasks": STARTED}, "toolu_1"), say("Started."),
        calls("save_tasks", {"tasks": STARTED}, "toolu_2"), say("Still started."),
    ), backend)
    manager = planner(backend)
    try:
        first = await turn(manager, "Start.")
        second = await turn(manager, "Again.")
    finally:
        await manager.cleanup()

    assert steps(first) == ["save_tasks", "progress 0/2"]
    assert steps(second) == ["save_tasks", "progress 0/2"]
