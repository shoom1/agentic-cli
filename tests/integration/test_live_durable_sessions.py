"""Live end-to-end: durable sessions survive a restart (fresh manager, same db).

Manager 1 tells the model a codeword (persisted to sqlite via ADK's
DatabaseSessionService); a brand-new Manager 2 over the same store resumes the
same session_id and the model recalls it — proving cross-restart durability with
full fidelity, no custom snapshot.

@pytest.mark.llm, ADK-only (needs GOOGLE_API_KEY); skipped by default:

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
    prompt="You are a concise assistant. When asked to remember something, do so.",
    tools=[],
    description="plain assistant",
)


async def _run(manager, message: str, session_id: str, user_id: str) -> str:
    out: list[str] = []
    async for ev in manager.process(message=message, user_id=user_id, session_id=session_id):
        if ev.type == EventType.TEXT:
            out.append(ev.content)
    return " ".join(out)


async def test_session_survives_a_fresh_manager(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = BaseSettings(
        workspace_dir=tmp_path, session_store="sqlite", permissions_enabled=False
    )
    set_settings(settings)
    sid = "durable-codeword"
    # The CLI always runs turns as settings.default_user; session queries use the
    # same, so mirror that here.
    user = settings.default_user

    # --- "Process 1": store a codeword, then tear the manager down. ---
    m1 = create_workflow_manager_from_settings(agent_configs=[_AGENT], settings=settings)
    await _run(m1, "Remember this codeword exactly: BANANA77. Just acknowledge.", sid, user)
    await m1.cleanup()

    # --- "Process 2": a brand-new manager over the same sqlite store. ---
    m2 = create_workflow_manager_from_settings(agent_configs=[_AGENT], settings=settings)
    await m2.initialize_services()  # the app resumes only after init (ensure_initialized)
    try:
        # The session is already durable in the store (no inject needed).
        assert await m2.session_exists(sid) is True
        answer = await _run(m2, "What codeword did I ask you to remember?", sid, user)
        assert "BANANA77" in answer, f"model did not recall across restart: {answer!r}"
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
    sid = "lg-durable-codeword"
    user = settings.default_user

    m1 = create_workflow_manager_from_settings(agent_configs=[_AGENT], settings=settings)
    assert m1.backend_type == "langgraph"
    await _run(m1, "Remember this codeword exactly: BANANA77. Just acknowledge.", sid, user)
    await m1.cleanup()

    m2 = create_workflow_manager_from_settings(agent_configs=[_AGENT], settings=settings)
    await m2.initialize_services()
    try:
        assert await m2.session_exists(sid) is True
        answer = await _run(m2, "What codeword did I ask you to remember?", sid, user)
        assert "BANANA77" in answer, f"langgraph did not recall across restart: {answer!r}"
    finally:
        await m2.cleanup()
