"""Tests for input callback public API on BaseWorkflowManager."""

from unittest.mock import AsyncMock, MagicMock, patch


class TestInputCallbackAPI:
    def test_set_input_callback(self):
        from agentic_cli.workflow.base_manager import BaseWorkflowManager

        with patch.object(BaseWorkflowManager, '__abstractmethods__', set()):
            manager = BaseWorkflowManager.__new__(BaseWorkflowManager)
            manager._user_input_callback = None

            callback = AsyncMock()
            manager.set_input_callback(callback)
            assert manager._user_input_callback is callback

    def test_clear_input_callback(self):
        from agentic_cli.workflow.base_manager import BaseWorkflowManager

        with patch.object(BaseWorkflowManager, '__abstractmethods__', set()):
            manager = BaseWorkflowManager.__new__(BaseWorkflowManager)
            manager._user_input_callback = AsyncMock()

            manager.clear_input_callback()
            assert manager._user_input_callback is None
