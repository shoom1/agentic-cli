"""Tests for input callback public API on BaseWorkflowManager.

The callback is held in a per-manager ContextVar rather than a plain attribute
(see ``set_input_callback``), so these assert the observable behaviour —
whether ``request_user_input`` reaches the callback — rather than the storage.
"""

from contextvars import ContextVar
from unittest.mock import AsyncMock, patch

import pytest

from agentic_cli.workflow.events import UserInputRequest


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
