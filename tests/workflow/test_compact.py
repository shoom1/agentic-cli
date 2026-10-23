"""The ``compact()`` hook: summarize a conversation now, where a backend can.

``BaseWorkflowManager.compact(session, focus=None)`` raises
``NotImplementedError``; ``supports_compact`` is derived from a backend
overriding it, as ``supports_sessions`` is, so it cannot drift from the code.
ADK and Deep Agents implement it (their own tests: ``test_adk_compaction.py``
and ``deepagents/test_compact.py``); LangGraph, deprecated, does not.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.context import CompactResult
from agentic_cli.workflow.sessions import SessionRef


class _Bare(BaseWorkflowManager):
    """A backend that implements nothing optional."""

    def _get_state_tools(self):
        return []

    @property
    def backend_type(self) -> str:
        return "test"

    async def _do_initialize(self) -> None:
        return None

    async def process(self, message, user_id, session_id=None):
        raise NotImplementedError

    async def reinitialize(self, model=None, preserve_sessions=True):
        return None

    async def cleanup(self):
        return None


class _Compacting(_Bare):
    async def compact(self, session, focus=None):
        return CompactResult(messages_summarized=2, tokens_before=10, tokens_after=4)


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.app_name = "test-app"
    settings.default_user = "default_user"
    return settings


REF = SessionRef("test-app", "u", "s")


class TestTheHook:
    async def test_a_backend_without_it_raises(self):
        manager = _Bare(agent_configs=[], settings=_settings())

        assert manager.supports_compact is False
        with pytest.raises(NotImplementedError, match="supports_compact"):
            await manager.compact(REF, focus="the API decisions")

    async def test_a_backend_that_implements_it_supports_it(self):
        manager = _Compacting(agent_configs=[], settings=_settings())

        assert manager.supports_compact is True
        assert await manager.compact(REF) == CompactResult(2, 10, 4)

    def test_the_result_is_frozen(self):
        result = CompactResult(messages_summarized=2, tokens_before=10, tokens_after=4)

        with pytest.raises(AttributeError):
            result.messages_summarized = 3  # type: ignore[misc]


class TestTheBackends:
    def test_adk_supports_it(self):
        pytest.importorskip("google.adk")
        from agentic_cli.config import BaseSettings
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        manager = GoogleADKWorkflowManager(
            agent_configs=[], settings=BaseSettings(session_store="memory")
        )

        assert manager.supports_compact is True

    def test_langgraph_does_not(self):
        pytest.importorskip("langgraph")
        from agentic_cli.config import BaseSettings
        from agentic_cli.workflow.langgraph.manager import LangGraphWorkflowManager

        manager = LangGraphWorkflowManager(
            agent_configs=[], settings=BaseSettings(session_store="memory")
        )

        assert manager.supports_compact is False
