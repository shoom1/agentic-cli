"""The manager exposes the active session/user during a turn (phase-2 association).

``JobManager`` reads ``active_session_id``/``active_user_id`` off the WORKFLOW
service to associate a resume-on-complete job with the conversation that
launched it. ``_workflow_context()`` sets these for the turn and clears them on
exit (even on error).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")

from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402


def _bare_manager() -> GoogleADKWorkflowManager:
    """A manager instance without running __init__ (concrete subclass of base)."""
    mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
    mgr._settings = SimpleNamespace(app_name="test")
    mgr._services = {}
    mgr._active_session_id = None
    mgr._active_user_id = None
    return mgr


def test_idle_active_ids_are_none():
    mgr = _bare_manager()
    assert mgr.active_session_id is None
    assert mgr.active_user_id is None


def test_context_sets_and_clears_active_ids():
    mgr = _bare_manager()
    with mgr._workflow_context(session_id="sess-1", user_id="user-1"):
        assert mgr.active_session_id == "sess-1"
        assert mgr.active_user_id == "user-1"
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
