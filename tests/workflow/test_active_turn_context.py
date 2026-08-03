"""The active turn is context-local, not manager-local.

``JobManager`` reads ``active_session_id``/``active_user_id`` off the WORKFLOW
service to associate a resume-on-complete job with the conversation that
launched it. ``_workflow_context()`` publishes them for the duration of a turn.

They used to live in manager instance attributes, which cross-wired two turns
running on one manager (possible for framework consumers — only the CLI
serializes turns) and made a nested context erase the outer turn on exit. They
are now a ``ContextVar`` restored from a token.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")

from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.sessions import SessionRef, get_active_turn  # noqa: E402


def _bare_manager() -> GoogleADKWorkflowManager:
    """A manager instance without running __init__ (concrete subclass of base)."""
    mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
    mgr._settings = SimpleNamespace(app_name="test", default_user="default_user")
    mgr._app_name = "test"
    mgr._services = {}
    mgr.session_id = "default_session"
    return mgr


def test_idle_active_ids_are_none():
    mgr = _bare_manager()
    assert mgr.active_session_id is None
    assert mgr.active_user_id is None
    assert mgr.active_turn is None


def test_context_sets_and_clears_active_ids():
    mgr = _bare_manager()
    with mgr._workflow_context(session_id="sess-1", user_id="user-1"):
        assert mgr.active_session_id == "sess-1"
        assert mgr.active_user_id == "user-1"
        assert mgr.active_turn == SessionRef("test", "user-1", "sess-1")
    assert mgr.active_session_id is None
    assert mgr.active_user_id is None


def test_context_clears_on_exception():
    mgr = _bare_manager()
    with pytest.raises(RuntimeError):
        with mgr._workflow_context(session_id="s", user_id="u"):
            assert mgr.active_session_id == "s"
            raise RuntimeError("boom")
    assert mgr.active_session_id is None
    assert mgr.active_user_id is None


def test_nested_context_restores_outer_turn():
    """The inner context must restore the outer turn, not clear it."""
    mgr = _bare_manager()
    with mgr._workflow_context(session_id="outer", user_id="alice"):
        with mgr._workflow_context(session_id="inner", user_id="bob"):
            assert mgr.active_session_id == "inner"
            assert mgr.active_user_id == "bob"
        assert mgr.active_session_id == "outer"
        assert mgr.active_user_id == "alice"
    assert mgr.active_turn is None


async def test_concurrent_turns_do_not_cross_wire():
    """Two turns on one manager must each see their own identity throughout."""
    mgr = _bare_manager()
    observed: dict[str, list[tuple[str | None, str | None]]] = {"a": [], "b": []}
    both_inside = asyncio.Barrier(2)

    async def _turn(tag: str, session_id: str, user_id: str) -> None:
        with mgr._workflow_context(session_id=session_id, user_id=user_id):
            observed[tag].append((mgr.active_session_id, mgr.active_user_id))
            # Force overlap: neither task leaves its context until both entered.
            await both_inside.wait()
            observed[tag].append((mgr.active_session_id, mgr.active_user_id))

    await asyncio.gather(
        _turn("a", "sess-a", "alice"),
        _turn("b", "sess-b", "bob"),
    )

    assert observed["a"] == [("sess-a", "alice"), ("sess-a", "alice")]
    assert observed["b"] == [("sess-b", "bob"), ("sess-b", "bob")]
    assert get_active_turn() is None


async def test_turn_identity_visible_to_spawned_tasks():
    """Tools run in tasks spawned inside the turn; they inherit its context."""
    mgr = _bare_manager()

    async def _tool() -> tuple[str | None, str | None]:
        return mgr.active_session_id, mgr.active_user_id

    with mgr._workflow_context(session_id="sess-1", user_id="alice"):
        result = await asyncio.create_task(_tool())

    assert result == ("sess-1", "alice")
