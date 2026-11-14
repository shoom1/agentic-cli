"""Tests for input callback public API on BaseWorkflowManager.

The callback is held in a per-manager ContextVar rather than a plain attribute
(see ``set_input_callback``), so these assert the observable behaviour —
whether ``request_user_input`` reaches the callback — rather than the storage.
"""

import asyncio
from contextvars import Context, ContextVar
from unittest.mock import AsyncMock, patch

import pytest

from agentic_cli.workflow.events import UserInputRequest, UserInputUnavailable
from agentic_cli.workflow.sessions import close_turn_scope, open_turn_scope


def _manager():
    from agentic_cli.workflow.base_manager import BaseWorkflowManager

    with patch.object(BaseWorkflowManager, "__abstractmethods__", set()):
        manager = BaseWorkflowManager.__new__(BaseWorkflowManager)
    manager._user_input_callback = ContextVar("test_input_callback", default=None)
    return manager


def _request() -> UserInputRequest:
    return UserInputRequest(request_id="r", tool_name="t", prompt="p")


class TestInputCallbackAPI:
    async def test_set_input_callback(self):
        manager = _manager()
        callback = AsyncMock(return_value="answer")

        manager.set_input_callback(callback)

        assert await manager.request_user_input(_request()) == "answer"
        callback.assert_awaited_once()

    async def test_clear_input_callback(self):
        manager = _manager()
        manager.set_input_callback(AsyncMock(return_value="answer"))

        manager.clear_input_callback()

        with pytest.raises(RuntimeError, match="No user input callback"):
            await manager.request_user_input(_request())

    async def test_clear_with_token_restores_the_previous_callback(self):
        manager = _manager()
        manager.set_input_callback(AsyncMock(return_value="outer"))
        token = manager.set_input_callback(AsyncMock(return_value="inner"))

        assert await manager.request_user_input(_request()) == "inner"

        manager.clear_input_callback(token)
        assert await manager.request_user_input(_request()) == "outer"

    async def test_no_callback_raises(self):
        manager = _manager()
        with pytest.raises(RuntimeError, match="No user input callback"):
            await manager.request_user_input(_request())


class TestOnlyInsideItsTurn:
    """A task a turn started copies the turn's context, the callback with it;
    once the turn has ended, the callback does not answer it (spec 9)."""

    async def test_a_task_that_outlives_its_turn_cannot_ask(self):
        manager = _manager()
        callback = AsyncMock(return_value="answer")
        manager.set_input_callback(callback)
        go = asyncio.Event()

        async def later() -> str:
            await go.wait()
            return await manager.request_user_input(_request())

        span = open_turn_scope()
        task = asyncio.create_task(later())  # copies the turn's context
        close_turn_scope(span)  # the turn ends; the task goes on
        go.set()

        with pytest.raises(UserInputUnavailable, match="has ended"):
            await task
        callback.assert_not_awaited()

    async def test_a_task_of_a_turn_still_running_asks(self):
        manager = _manager()
        manager.set_input_callback(AsyncMock(return_value="answer"))

        span = open_turn_scope()
        try:
            assert await asyncio.create_task(manager.request_user_input(_request())) == "answer"
        finally:
            close_turn_scope(span)

    async def test_a_callback_installed_outside_any_turn_answers_every_turn(self):
        """A script's (and the contract suite's ``answer_with``) pattern: one
        callback installed before any turn, asked in each turn that follows."""
        manager = _manager()
        callback = AsyncMock(return_value="answer")
        manager.set_input_callback(callback)

        for _ in range(2):
            span = open_turn_scope()
            try:
                assert await asyncio.create_task(
                    manager.request_user_input(_request())
                ) == "answer"
            finally:
                close_turn_scope(span)

        assert await manager.request_user_input(_request()) == "answer"  # and between turns
        assert callback.await_count == 3

    async def test_a_span_closed_from_another_context_ends_nothing_here(self):
        """A turn left in another context (a ``process()`` generator the GC
        finalizes while another turn runs) cannot end this turn's span."""
        manager = _manager()
        manager.set_input_callback(AsyncMock(return_value="answer"))
        elsewhere = Context().run(open_turn_scope)

        span = open_turn_scope()
        try:
            with pytest.raises(ValueError):
                close_turn_scope(elsewhere)
            assert await manager.request_user_input(_request()) == "answer"
        finally:
            close_turn_scope(span)
