"""What writes to the open conversation runs in its core (``cli/conversation.py``).

``ConversationCore`` owns the turn lock, which every write to the
conversation and every change of it takes. ``BaseCLIApp`` reads it through
``_turn_lock``; an app built with ``__new__`` (tests, an application's own
construction) has one too.
"""

from __future__ import annotations

import asyncio

import pytest

from agentic_cli.cli import AppInfo
from agentic_cli.cli import app as app_module
from agentic_cli.cli.app import BaseCLIApp
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


def build() -> BaseCLIApp:
    return BaseCLIApp(
        app_info=AppInfo(name="Turns test", version="0"),
        agent_configs=[AgentConfig(name="assistant", prompt="Help.")],
        settings=BaseSettings(session_store="sqlite"),
    )


class TestTheTurnLock:
    def test_the_core_owns_it_and_the_app_reads_it(self, home):
        app = build()

        assert isinstance(app._conversation.turn_lock, asyncio.Lock)
        assert app._turn_lock is app._conversation.turn_lock

    def test_an_app_built_with_new_has_one(self):
        app = BaseCLIApp.__new__(BaseCLIApp)

        assert app._turn_lock is app._conversation.turn_lock

    async def test_a_change_of_conversation_waits_for_it(self, home):
        """``/new`` takes the core's lock: it waits for a turn holding it."""
        app = build()

        async with app._conversation.turn_lock:
            new = asyncio.create_task(app.start_new_session())
            await asyncio.sleep(0.05)
            assert not new.done()
        await new
