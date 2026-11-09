"""The conversation the app is in (``cli/conversation.py``).

``ConversationCore`` keeps what makes the conversation the CLI's turns run
in: its id, whether a turn ran in it since it was opened (``turned``), and
the lock this process holds on it, at its ``Place`` (the backend, the store,
the app and the user: the same id in another backend or store is another
conversation). ``BaseCLIApp`` holds one, and its ``_session_id`` and
``_turned`` read and write it; an app built with ``__new__`` (tests, an
application's own construction) has one too.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentic_cli.cli import AppInfo
from agentic_cli.cli import app as app_module
from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.conversation import ConversationCore, Place
from agentic_cli.cli.conversation_lock import ConversationLock
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig


@pytest.fixture(autouse=True)
def quiet_app(monkeypatch):
    """No global logging configuration or warning hook left behind."""
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_install_adk_notice_hook", lambda: None)


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def place(folder: Path, store: str = "sqlite", backend: str = "adk") -> Place:
    return Place(backend=backend, store=store, app_name="app", user="me", locks=folder)


def held_elsewhere(where: Place, session_id: str) -> int | None:
    """Whether another process could open it: the holder's pid, or None (and
    it is let go again)."""
    other = ConversationLock(where.locks)
    holder = other.take(where.key(session_id))
    other.release()
    return holder


def build(session_id: str | None = None) -> BaseCLIApp:
    return BaseCLIApp(
        app_info=AppInfo(name="Core test", version="0"),
        agent_configs=[AgentConfig(name="assistant", prompt="Help.")],
        settings=BaseSettings(session_store="sqlite"),
        session_id=session_id,
    )


class TestThePlace:
    def test_it_is_read_from_the_settings(self, home):
        settings = BaseSettings(session_store="sqlite", orchestrator="deepagents")

        assert Place.of(settings) == Place(
            backend="deepagents", store="sqlite", app_name=settings.app_name,
            user=settings.default_user, locks=settings.sessions_dir / "locks",
        )

    def test_a_conversation_is_keyed_by_its_place_and_id(self, tmp_path):
        assert place(tmp_path).key("abc") == ("adk", "sqlite", "app", "me", "abc")


class TestTheCore:
    def test_a_new_core_is_in_its_conversation_unused_and_holding_nothing(self, tmp_path):
        core = ConversationCore("abc", place(tmp_path))

        assert (core.session_id, core.turned) == ("abc", False)
        assert held_elsewhere(place(tmp_path), "abc") is None

    def test_what_it_holds_another_process_is_refused(self, tmp_path):
        core = ConversationCore("abc", place(tmp_path))

        assert core.hold("abc") is None
        assert held_elsewhere(place(tmp_path), "abc") == os.getpid()

    def test_one_held_elsewhere_is_refused_with_its_holder(self, tmp_path):
        other = ConversationLock(tmp_path)
        other.take(place(tmp_path).key("abc"))

        assert ConversationCore("xyz", place(tmp_path)).hold("abc") == os.getpid()

    def test_a_hold_taken_beside_is_kept_or_let_go(self, tmp_path):
        core = ConversationCore("abc", place(tmp_path))
        core.hold("abc")

        kept, dropped = core.acquire("kept"), core.acquire("dropped")
        assert held_elsewhere(place(tmp_path), "abc") == os.getpid()  # until the commit
        core.commit(kept)
        core.abandon(dropped)

        assert held_elsewhere(place(tmp_path), "abc") is None
        assert held_elsewhere(place(tmp_path), "kept") == os.getpid()
        assert held_elsewhere(place(tmp_path), "dropped") is None

    def test_nothing_is_locked_on_the_memory_store(self, tmp_path):
        core = ConversationCore("abc", place(tmp_path, store="memory"))

        assert core.hold("abc") is None
        assert list(tmp_path.iterdir()) == []  # no lock file made

    def test_release_lets_it_go(self, tmp_path):
        core = ConversationCore("abc", place(tmp_path))
        core.hold("abc")

        core.release()

        assert held_elsewhere(place(tmp_path), "abc") is None


class TestTheAppsConversation:
    def test_the_apps_id_and_turned_are_the_cores(self, home):
        app = build("abc")

        assert app._conversation.session_id == "abc" == app.session_id
        app._turned = True
        app._session_id = "xyz"

        assert (app._conversation.session_id, app._conversation.turned) == ("xyz", True)

    def test_its_conversations_are_keyed_by_the_settings_place(self, home):
        app = build()

        assert app._conversation_key("abc") == Place.of(app.settings).key("abc")

    def test_an_app_built_with_new_has_one_too(self):
        app = BaseCLIApp.__new__(BaseCLIApp)

        assert app._turned is False
        app._session_id = "s"
        app._turned = True

        assert (app._conversation.session_id, app._conversation.turned) == ("s", True)
