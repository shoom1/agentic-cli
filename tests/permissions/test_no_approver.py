"""A permission prompt that nobody can answer is a denial, not a crash.

When no consumer has installed an input callback (a headless run, a script
driving the manager directly, or a turn whose UI has gone away), the engine's
ASK path used to let ``request_user_input``'s RuntimeError escape
``engine.check`` and abort the whole turn. The engine must fail closed: deny
the call with a reason the model can read, and let the turn continue.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_cli.workflow.events import UserInputRequest, UserInputUnavailable
from agentic_cli.workflow.permissions.capabilities import Capability
from agentic_cli.workflow.permissions.engine import PermissionEngine
from agentic_cli.workflow.permissions.rules import RuleSource
from agentic_cli.workflow.permissions.store import PermissionContext

HTTP = [Capability("http.read", target_arg="url")]
ARGS = {"url": "https://example.com/page"}


def _manager():
    """A real BaseWorkflowManager with no input callback installed."""
    from agentic_cli.workflow.base_manager import BaseWorkflowManager

    with patch.object(BaseWorkflowManager, "__abstractmethods__", set()):
        manager = BaseWorkflowManager.__new__(BaseWorkflowManager)
    manager._user_input_callback = ContextVar("test_no_approver", default=None)
    return manager


def _settings() -> MagicMock:
    s = MagicMock()
    s.permissions_enabled = True
    s.app_name = "agentic"
    return s


@pytest.fixture
def ctx(tmp_path: Path) -> PermissionContext:
    return PermissionContext(workdir=tmp_path, home=tmp_path / "home")


def _engine(workflow, ctx) -> PermissionEngine:
    return PermissionEngine(settings=_settings(), workflow=workflow, ctx=ctx)


class TestRequestUserInput:
    async def test_no_callback_raises_a_typed_error(self):
        with pytest.raises(UserInputUnavailable):
            await _manager().request_user_input(
                UserInputRequest(request_id="r", tool_name="t", prompt="p")
            )

    def test_the_typed_error_is_still_a_runtime_error(self):
        """Callers that caught RuntimeError keep working."""
        assert issubclass(UserInputUnavailable, RuntimeError)


class TestEngineWithoutApprover:
    async def test_ask_without_a_callback_denies(self, ctx):
        result = await _engine(_manager(), ctx).check("web_fetch", HTTP, ARGS)

        assert result.allowed is False
        assert "no interactive approver" in result.reason

    async def test_denial_is_not_remembered(self, ctx):
        """Once a UI attaches, the same call is put to the user."""
        manager = _manager()
        engine = _engine(manager, ctx)
        await engine.check("web_fetch", HTTP, ARGS)
        assert not [r for r in engine.rules if r.source is RuleSource.SESSION]

        asked: list[UserInputRequest] = []

        async def approve(request):
            asked.append(request)
            return "Allow once"

        manager.set_input_callback(approve)
        result = await engine.check("web_fetch", HTTP, ARGS)

        assert result.allowed is True
        assert len(asked) == 1

    async def test_a_failing_prompt_denies(self, ctx):
        manager = _manager()

        async def broken(request):
            raise OSError("terminal went away")

        manager.set_input_callback(broken)
        result = await _engine(manager, ctx).check("web_fetch", HTTP, ARGS)

        assert result.allowed is False
        assert "approval prompt failed" in result.reason

    async def test_cancellation_still_propagates(self, ctx):
        """A cancelled turn is not an answer: it must not become a denial the
        model then reacts to."""
        manager = _manager()

        async def cancelled(request):
            raise asyncio.CancelledError

        manager.set_input_callback(cancelled)
        with pytest.raises(asyncio.CancelledError):
            await _engine(manager, ctx).check("web_fetch", HTTP, ARGS)

    async def test_a_later_call_can_still_ask(self, ctx):
        """The failure must release the ask lock."""
        manager = _manager()
        engine = _engine(manager, ctx)
        await engine.check("web_fetch", HTTP, ARGS)

        async def approve(request):
            return "Allow once"

        manager.set_input_callback(approve)
        result = await asyncio.wait_for(engine.check("web_fetch", HTTP, ARGS), 5)
        assert result.allowed is True


class TestAskClarificationWithoutApprover:
    async def test_returns_an_error_dict(self):
        from agentic_cli.tools.interaction_tools import ask_clarification
        from agentic_cli.workflow.service_registry import (
            WORKFLOW,
            clear_service_registry,
            set_service_registry,
        )

        set_service_registry({WORKFLOW: _manager()})
        try:
            result = await ask_clarification("Which format?", ["pdf", "html"])
        finally:
            clear_service_registry()

        assert result["success"] is False
        assert result["response"] is None
        assert "no interactive" in result["error"].lower()
