"""The lines around memory: proposals waiting, and each memory write (spec §9.3, §12)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_cli.memory import Candidate, MemoryStore
from tests.cli.test_compact_and_new import (  # noqa: F401
    app_on,
    fake_settings,
    project,
    quiet_app,
    said,
)


@pytest.fixture(autouse=True)
def _in_a_project(project):
    """A hermetic project and HOME for every app these tests build."""


def _two_waiting(tmp_path) -> MemoryStore:
    store = MemoryStore(tmp_path / "user")
    store.propose(Candidate("new", None, "Uses metric", "km.", "user", "user"))
    store.propose(Candidate("new", None, "Likes otters", "Otters.", "user", "user"))
    return store


async def test_waiting_proposals_are_named(tmp_path):
    workflow = SimpleNamespace(memory_store=_two_waiting(tmp_path))
    app = app_on(workflow, fake_settings())
    await app._name_waiting_proposals()
    assert "2 memory proposals: /memory inbox" in said(app.session, "add_message")


async def test_nothing_is_said_when_none_wait(tmp_path):
    app = app_on(SimpleNamespace(memory_store=MemoryStore(tmp_path / "user")), fake_settings())
    await app._name_waiting_proposals()
    assert said(app.session, "add_message") == []


async def test_nothing_is_said_without_a_store(tmp_path):
    app = app_on(SimpleNamespace(memory_store=None), fake_settings())
    await app._name_waiting_proposals()
    assert said(app.session, "add_message") == []


async def test_nothing_is_said_before_the_workflow_is_ready(tmp_path):
    app = app_on(SimpleNamespace(memory_store=_two_waiting(tmp_path), is_initialized=False), fake_settings())
    await app._name_waiting_proposals()
    assert said(app.session, "add_message") == []


async def test_a_store_that_cannot_be_read_says_nothing_and_fails_nothing(tmp_path):
    def unreadable():
        raise OSError("disk gone")

    app = app_on(SimpleNamespace(memory_store=SimpleNamespace(proposals=unreadable)), fake_settings())
    await app._name_waiting_proposals()
    assert said(app.session, "add_message") == []


async def test_startup_names_them_before_the_prompt_is_used(tmp_path):
    app = app_on(SimpleNamespace(memory_store=_two_waiting(tmp_path)), fake_settings())
    app.session.app = SimpleNamespace(is_running=True)
    await app._open_on_startup()
    assert said(app.session, "add_message") == ["2 memory proposals: /memory inbox"]


async def test_startup_names_them_before_it_resumes(tmp_path):
    app = app_on(SimpleNamespace(memory_store=_two_waiting(tmp_path)), fake_settings())
    app.session.app = SimpleNamespace(is_running=True)
    app._pending_resume = "abc123"
    at_resume: list[list[str]] = []

    async def resume(target: str) -> bool:
        at_resume.append(said(app.session, "add_message"))
        return True

    app._resume = resume
    await app._open_on_startup()
    assert at_resume == [["2 memory proposals: /memory inbox"]]


async def test_an_extraction_at_new_names_them(tmp_path):
    store = MemoryStore(tmp_path / "user")

    async def on_session_end(messages=None, *, session=None):
        return [store.propose(Candidate("new", None, "Uses metric", "km.", "user", "user"))]

    workflow = SimpleNamespace(memory_store=store, on_session_end=on_session_end,
                               session_ref=lambda session_id, user_id: None)
    app = app_on(workflow, fake_settings(auto_extract_session_facts=True))
    await app._saving_facts("old")
    assert "1 memory proposal: /memory inbox" in said(app.session, "add_message")


async def test_no_extraction_says_nothing_about_proposals(tmp_path):
    workflow = SimpleNamespace(memory_store=_two_waiting(tmp_path))
    app = app_on(workflow, fake_settings(auto_extract_session_facts=False))
    await app._saving_facts("old")
    assert said(app.session, "add_message") == []


async def test_nothing_is_said_at_exit(tmp_path):
    store = MemoryStore(tmp_path / "user")

    async def on_session_end(messages=None, *, session=None):
        return [store.propose(Candidate("new", None, "Uses metric", "km.", "user", "user"))]

    workflow = SimpleNamespace(memory_store=store, on_session_end=on_session_end,
                               session_ref=lambda session_id, user_id: None)
    app = app_on(workflow, fake_settings(auto_extract_session_facts=True))
    app._turned = True
    await app._extract_session_facts_on_exit()
    assert len(store.proposals()) == 1
    assert said(app.session, "add_message") == []
