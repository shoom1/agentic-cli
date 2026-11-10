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


class AppSettings(BaseSettings):
    """An app's settings overriding ``session_db_url()`` as 0.6.4 documented
    it: no argument."""

    def session_db_url(self) -> str | None:
        return None  # the app keeps its sessions in memory, say


ASKED: list = []


class StoreAwareSettings(BaseSettings):
    """An override that takes the store, as ``BaseSettings`` now does."""

    def session_db_url(self, store: str | None = None) -> str | None:
        ASKED.append(store)
        return None


class TestAnAppsSessionDbUrl:
    """``session_db_url()`` keeps 0.6.4's call: no argument while the manager's
    store is the settings' own; ``store=`` only when they differ (a
    ``/settings`` change of ``session_store`` in the run), which an override
    without the parameter cannot follow."""

    @pytest.fixture
    def app_settings(self, settings) -> AppSettings:
        return AppSettings(session_store="sqlite")

    def test_a_library_user_builds_on_it(self, app_settings):
        from google.adk.sessions import InMemorySessionService

        manager = factory.create_workflow_manager_from_settings(SOLO, app_settings)

        assert isinstance(manager._make_session_service(), InMemorySessionService)

    def test_the_cli_builds_on_it_while_the_store_is_unchanged(self, app_settings):
        from google.adk.sessions import InMemorySessionService

        manager = factory.create_workflow_manager_from_settings(
            SOLO, app_settings, session_store="sqlite"
        )

        assert isinstance(manager._make_session_service(), InMemorySessionService)

    def test_after_a_store_change_it_cannot_rebuild_and_says_so(self, app_settings):
        manager = factory.create_workflow_manager_from_settings(
            SOLO, app_settings, session_store="sqlite"
        )
        app_settings.session_store = "memory"  # /settings, in the run

        with pytest.raises(RuntimeError) as raised:
            manager._make_session_service()

        assert str(raised.value) == (
            "This app's settings.session_db_url() takes no store, so this run "
            "cannot rebuild on its session store after session_store changed: "
            "restart the app."
        )

    def test_after_a_store_change_one_that_takes_it_is_asked_for_the_runs(self, settings):
        ASKED.clear()
        app_settings = StoreAwareSettings(session_store="sqlite")
        manager = factory.create_workflow_manager_from_settings(
            SOLO, app_settings, session_store="sqlite"
        )
        app_settings.session_store = "memory"

        manager._make_session_service()

        assert ASKED == ["sqlite"]

    def test_the_frameworks_own_names_the_runs_store_after_a_change(self, settings):
        from google.adk.sessions import DatabaseSessionService

        manager = factory.create_workflow_manager_from_settings(
            SOLO, settings, session_store="sqlite"
        )
        settings.session_store = "memory"

        assert isinstance(manager._make_session_service(), DatabaseSessionService)
