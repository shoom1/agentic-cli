"""The session rows each backend lists, side by side: ADK and Deep Agents.

The same conversations give the same rows: the first prompt as the title,
the turns the user ran, newest first, ``limit`` keeping the newest, and
``session_info`` that conversation's row, or None for one that is not
stored (another id, or another user's).

ADK runs on Claude through ``FakeAPI``, Deep Agents on scripted models; both
keep their sessions in memory.
"""

from __future__ import annotations

import anthropic
import pytest

pytest.importorskip("google.adk")
pytest.importorskip("deepagents")

from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import manager as deepagents_manager  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from tests.conftest import MockContext  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedModels, ai  # noqa: E402
from tests.workflow.test_adk_delegation import (  # noqa: E402,F401
    Team,
    manager_for,
    project,
    route,
    says,
    team,
)

REPLIES = ["One.", "Two.", "Three."]


@pytest.fixture(params=["adk", "deepagents"])
async def backend(request, project, monkeypatch):
    """A manager of each backend whose model answers ``REPLIES`` in turn."""
    if request.param == "adk":
        route(monkeypatch, anthropic, "AsyncAnthropic",
              FakeAPI(Team(coordinator=[says(reply) for reply in REPLIES])))
        manager = manager_for(team())
        try:
            yield manager
        finally:
            await manager.cleanup()
        return
    monkeypatch.setattr(deepagents_manager, "build_chat_model",
                        ScriptedModels(solo=[ai(reply) for reply in REPLIES]))
    with MockContext(google_api_key="test-key", session_store="memory") as context:
        manager = DeepAgentsWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="p")],
            settings=context.settings, model="gemini-2.5-flash",
        )
        try:
            yield manager
        finally:
            await manager.cleanup()


async def say(manager, text: str, session_id: str) -> None:
    async for _ in manager.process(text, "u", session_id=session_id):
        pass


async def test_two_turns_are_the_first_prompt_and_two_turns(backend):
    await say(backend, "  Plan the\ntrip. ", "s1")
    await say(backend, "And the budget?", "s1")

    (row,) = await backend.list_sessions(user_id="u")

    assert (row.session_id, row.title, row.turns) == ("s1", "Plan the trip.", 2)
    assert row.updated.tzinfo is not None
    assert await backend.session_info("s1", user_id="u") == row


async def test_newest_first_and_the_limit_keeps_the_newest(backend):
    await say(backend, "First.", "s1")
    await say(backend, "Second.", "s2")
    await say(backend, "First, again.", "s1")

    listed = await backend.list_sessions(user_id="u")
    newest = await backend.list_sessions(user_id="u", limit=1)

    assert [(r.session_id, r.turns) for r in listed] == [("s1", 2), ("s2", 1)]
    assert newest == listed[:1]


async def test_a_conversation_that_is_not_stored(backend):
    await say(backend, "Hello.", "s1")

    assert await backend.session_info("missing", user_id="u") is None
    assert await backend.session_info("s1", user_id="someone-else") is None
    assert await backend.delete_session("missing", user_id="u") is False
    assert [r.session_id for r in await backend.list_sessions(user_id="someone-else")] == []


async def test_a_job_told_to_the_conversation_is_no_turn(backend):
    """Nobody typed a job's notice: the turns and the title stay the user's."""
    from types import SimpleNamespace

    from agentic_cli.tools.jobs import JobState

    await say(backend, "Build it.", "s1")
    record = SimpleNamespace(
        job_id="j1", tool="build", call_id="c1", call_name="build", session_id="s1",
        user_id="u", state=JobState.SUCCEEDED, exit_code=0, error=None,
    )
    async for _ in backend.resume_with_job_result(record, result="built"):
        pass

    row = await backend.session_info("s1", user_id="u")
    assert (row.title, row.turns) == ("Build it.", 1)
