"""A manager keeps conversations in the session store it was built for.

A CLI run keeps one session store for its whole life: ``session_store`` takes
effect at the next start, so every manager the controller builds in the run
(a retry after a failed start, a new conversation on another backend) is
built on the store the run started with (``create_workflow_manager_from_
settings(session_store=)``, ``BaseWorkflowManager.session_store``), not on
the setting as ``/settings`` left it. A manager built without one keeps its
settings' store.
"""

from __future__ import annotations

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.workflow import factory
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.settings import OrchestratorType

SOLO = [AgentConfig(name="solo", prompt="Help.")]


@pytest.fixture
def settings(tmp_path, monkeypatch) -> BaseSettings:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    return BaseSettings(session_store="sqlite")


def test_a_manager_keeps_its_settings_store_by_default(settings):
    manager = factory.create_workflow_manager_from_settings(SOLO, settings)

    assert manager.session_store == "sqlite"


def test_adk_builds_the_store_it_was_built_for(settings):
    from google.adk.sessions import InMemorySessionService

    manager = factory.create_workflow_manager_from_settings(
        SOLO, settings, session_store="memory"
    )

    assert manager.session_store == "memory"
    assert isinstance(manager._make_session_service(), InMemorySessionService)


async def test_deep_agents_opens_the_store_it_was_built_for(settings, monkeypatch):
    pytest.importorskip("deepagents")
    from agentic_cli.workflow.deepagents import manager as deepagents_manager

    settings.orchestrator = OrchestratorType.DEEPAGENTS
    opened: list[str] = []

    async def open_checkpointer(kind, **where):
        opened.append(kind)
        raise RuntimeError("stop here")

    monkeypatch.setattr(deepagents_manager, "open_checkpointer", open_checkpointer)
    manager = factory.create_workflow_manager_from_settings(
        SOLO, settings, session_store="memory"
    )

    with pytest.raises(RuntimeError, match="stop here"):
        await manager._open_session_store()

    assert opened == ["memory"]


def test_langgraph_is_given_the_store_it_was_built_for(settings):
    assert factory._backend_kwargs(OrchestratorType.LANGGRAPH, settings, "memory") == {
        "checkpointer": "memory"
    }
