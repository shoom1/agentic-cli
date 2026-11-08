"""A capability approved per call: no builtin allow, no grant (spec §9.2)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.workflow.events import UserInputUnavailable
from agentic_cli.workflow.permissions import Capability, PermissionContext, PermissionEngine
from agentic_cli.workflow.permissions.engine import Unstorable, unstorable_reason
from agentic_cli.workflow.permissions.prompt import ALLOW_ONCE_CHOICE, DENY_CHOICE, build_request
from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource
from agentic_cli.workflow.outside_content import outside_content_note

NOTE = outside_content_note("web_fetch")
SAVE = [Capability("memory.write")]
ARGS = {"description": "Prefers tabs", "content": "Line one\nLine two", "scope": "user"}


class _Workflow:
    def __init__(self, answer=ALLOW_ONCE_CHOICE):
        self.answer = answer
        self.asked = []

    async def request_user_input(self, request):
        self.asked.append(request)
        if self.answer is None:
            raise UserInputUnavailable("nobody")
        return self.answer


def _engine(tmp_path, workflow, *, marked=True, user_allow=False):
    settings = BaseSettings(workspace_dir=tmp_path / "w")
    ctx = PermissionContext(workdir=tmp_path, home=tmp_path / "home", app_name="app")
    engine = PermissionEngine(
        settings=settings, workflow=workflow, ctx=ctx,
        per_call=lambda s: {"memory.write": NOTE} if marked else {},
    )
    if user_allow:  # a rule from the user's own settings
        engine._base_rules.append(Rule("memory.write", "*", Effect.ALLOW, RuleSource.USER))
    return engine


async def test_unmarked_the_builtin_allow_applies(tmp_path):
    workflow = _Workflow()
    result = await _engine(tmp_path, workflow, marked=False).check("save_memory", SAVE, ARGS)
    assert result.allowed and workflow.asked == []


async def test_marked_a_write_asks_once_only_with_the_note_and_the_arguments(tmp_path):
    workflow = _Workflow()
    result = await _engine(tmp_path, workflow).check("save_memory", SAVE, ARGS)
    assert result.allowed
    [request] = workflow.asked
    assert request.choices == [ALLOW_ONCE_CHOICE, DENY_CHOICE]
    assert NOTE in request.prompt
    assert "content: Line one\nLine two" in request.prompt
    assert "description: Prefers tabs" in request.prompt


async def test_marked_a_session_grant_does_not_carry_over(tmp_path):
    workflow = _Workflow()
    engine = _engine(tmp_path, workflow)
    engine._session_rules.append(Rule("memory.write", "*", Effect.ALLOW, RuleSource.SESSION))
    await engine.check("save_memory", SAVE, ARGS)
    assert len(workflow.asked) == 1


async def test_marked_a_deny_still_denies_without_asking(tmp_path):
    workflow = _Workflow()
    engine = _engine(tmp_path, workflow)
    engine._session_rules.append(Rule("memory.write", "*", Effect.DENY, RuleSource.SESSION))
    result = await engine.check("save_memory", SAVE, ARGS)
    assert not result.allowed and workflow.asked == []


async def test_marked_the_user_s_own_allow_rule_applies(tmp_path):
    workflow = _Workflow()
    engine = _engine(tmp_path, workflow, user_allow=True)
    assert (await engine.check("save_memory", SAVE, ARGS)).allowed
    assert workflow.asked == []


async def test_marked_with_nobody_to_ask_it_is_denied(tmp_path):
    result = await _engine(tmp_path, _Workflow(answer=None)).check("save_memory", SAVE, ARGS)
    assert not result.allowed


async def test_marked_reads_do_not_ask(tmp_path):
    workflow = _Workflow()
    assert (await _engine(tmp_path, workflow).check("read_memory", [Capability("memory.read")], {})).allowed
    assert workflow.asked == []


def test_the_reason_comes_first():
    cap = SimpleNamespace(name="memory.write", target="*")
    assert unstorable_reason(cap, per_call={"memory.write": NOTE}) is Unstorable.OUTSIDE_CONTENT


def test_the_manager_hook_reads_the_turn_s_mark(tmp_path):
    from agentic_cli.workflow.sessions import SessionRef, reset_active_turn, set_active_turn
    from tests.workflow.test_memory_wiring import _FakeManager

    mgr = _FakeManager(agent_configs=[], settings=BaseSettings(workspace_dir=tmp_path))
    ref = SessionRef(app_name=mgr.app_name, user_id="u", session_id="s")
    token = set_active_turn(ref)
    try:
        assert mgr._per_call_capabilities(mgr.settings) == {}
        mgr._outside_marks.mark(ref, "web_fetch")
        assert mgr._per_call_capabilities(mgr.settings) == {"memory.write": NOTE}
    finally:
        reset_active_turn(token)
