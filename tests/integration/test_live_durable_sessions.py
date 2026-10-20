"""Live end-to-end: durable sessions survive a restart (fresh manager, same db).

Manager 1 tells the model a project name (persisted to sqlite via ADK's
DatabaseSessionService); a brand-new Manager 2 over the same store resumes the
same session_id and the model answers with it — proving cross-restart durability
with full fidelity, no custom snapshot.

The same check runs on the LangGraph and Deep Agents backends (their
persistent checkpointers).

@pytest.mark.llm (needs GOOGLE_API_KEY); skipped by default:

    conda run -n agenticcli python -m pytest tests/integration/test_live_durable_sessions.py -v -m llm
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentic_cli.config import BaseSettings, set_settings
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import EventType
from agentic_cli.workflow.factory import create_workflow_manager_from_settings

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"), reason="durable sessions test needs GOOGLE_API_KEY"
    ),
]

_AGENT = AgentConfig(
    name="assistant",
    prompt="You are a concise assistant.",
    tools=[],
    description="plain assistant",
)


async def _run(manager, message: str, session_id: str, user_id: str) -> str:
    out: list[str] = []
    async for ev in manager.process(message=message, user_id=user_id, session_id=session_id):
        if ev.type == EventType.TEXT:
            out.append(ev.content)
    return " ".join(out)


# A project fact, not something to "remember": Gemini sometimes answers that it
# cannot remember arbitrary information, which made these tests flaky. The name
# is invented, so the model can only answer it from the stored session.
_TELL = "The project we are discussing is named Quillfeather. Reply OK."
_ASK = "What is the project's name? Reply with the name only."
_NAME = "quillfeather"


async def test_session_survives_a_fresh_manager(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = BaseSettings(
        workspace_dir=tmp_path, session_store="sqlite", permissions_enabled=False
    )
    set_settings(settings)
    sid = "durable-project"
    # The CLI always runs turns as settings.default_user; session queries use the
    # same, so mirror that here.
    user = settings.default_user

    # --- "Process 1": tell it the project name, then tear the manager down. ---
    m1 = create_workflow_manager_from_settings(agent_configs=[_AGENT], settings=settings)
    await _run(m1, _TELL, sid, user)
    await m1.cleanup()

    # --- "Process 2": a brand-new manager over the same sqlite store. ---
    m2 = create_workflow_manager_from_settings(agent_configs=[_AGENT], settings=settings)
    await m2.initialize_services()  # the app resumes only after init (ensure_initialized)
    try:
        # The session is already durable in the store (no inject needed).
        assert await m2.session_exists(sid) is True
        answer = await _run(m2, _ASK, sid, user)
        assert _NAME in answer.lower(), f"model did not recall across restart: {answer!r}"
    finally:
        await m2.cleanup()


async def test_langgraph_session_survives_a_fresh_manager(tmp_path: Path, monkeypatch):
    """Same durability check on the LangGraph backend (persistent checkpointer).

    Forces gemini onto LangGraph (orchestrator=langgraph) — which also exercises
    the gemini-2.5 thinking_budget fix on the LangGraph LLM path.
    """
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_google_genai")
    pytest.importorskip("langgraph.checkpoint.sqlite.aio")

    monkeypatch.chdir(tmp_path)
    settings = BaseSettings(
        workspace_dir=tmp_path,
        session_store="sqlite",
        permissions_enabled=False,
        orchestrator="langgraph",  # force LangGraph for the gemini default
    )
    set_settings(settings)
    sid = "lg-durable-project"
    user = settings.default_user

    m1 = create_workflow_manager_from_settings(agent_configs=[_AGENT], settings=settings)
    assert m1.backend_type == "langgraph"
    await _run(m1, _TELL, sid, user)
    await m1.cleanup()

    m2 = create_workflow_manager_from_settings(agent_configs=[_AGENT], settings=settings)
    await m2.initialize_services()
    try:
        assert await m2.session_exists(sid) is True
        answer = await _run(m2, _ASK, sid, user)
        assert _NAME in answer.lower(), f"langgraph did not recall across restart: {answer!r}"
    finally:
        await m2.cleanup()


async def test_deepagents_session_survives_a_fresh_manager(tmp_path: Path, monkeypatch):
    """The same durability check on the Deep Agents backend.

    Its sqlite checkpointer lives at {workspace}/sessions/checkpoints.db, and
    the thread is the whole (app, user, session) triple.
    """
    pytest.importorskip("deepagents")

    monkeypatch.chdir(tmp_path)
    settings = BaseSettings(
        workspace_dir=tmp_path,
        session_store="sqlite",
        permissions_enabled=False,
        orchestrator="deepagents",
    )
    set_settings(settings)
    sid = "da-durable-project"
    user = settings.default_user

    m1 = create_workflow_manager_from_settings(agent_configs=[_AGENT], settings=settings)
    assert m1.backend_type == "deepagents"
    await _run(m1, _TELL, sid, user)
    await m1.cleanup()
    assert (tmp_path / "sessions" / "checkpoints.db").is_file()

    m2 = create_workflow_manager_from_settings(agent_configs=[_AGENT], settings=settings)
    await m2.initialize_services()
    try:
        assert await m2.session_exists(sid) is True
        answer = await _run(m2, _ASK, sid, user)
        assert _NAME in answer.lower(), f"deepagents did not recall across restart: {answer!r}"
    finally:
        await m2.cleanup()
