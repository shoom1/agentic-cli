"""Tests for token caching support.

Tests verify prompt caching configuration for Claude models in LangGraph,
cache-related fields in WorkflowEvent.llm_usage(), and the
prompt_caching_enabled setting.
"""

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import WorkflowEvent, EventType


class TestPromptCachingSetting:
    """Tests for the prompt_caching_enabled setting."""

    def test_prompt_caching_enabled_default(self):
        """Verify setting defaults to True."""
        settings = BaseSettings(google_api_key="test-key")
        assert settings.prompt_caching_enabled is True

    def test_prompt_caching_can_be_disabled(self):
        """Verify setting can be set to False."""
        settings = BaseSettings(
            google_api_key="test-key",
            prompt_caching_enabled=False,
        )
        assert settings.prompt_caching_enabled is False


def _langgraph_manager(agent_config, **values):
    pytest.importorskip("langgraph")
    from agentic_cli.workflow.langgraph import LangGraphWorkflowManager

    return LangGraphWorkflowManager(
        agent_configs=[agent_config],
        settings=BaseSettings(session_store="memory", **values),
        checkpointer="memory",
    )


async def _turn(manager) -> None:
    try:
        async for _ in manager.process("hello", user_id="u", session_id="s"):
            pass
    finally:
        await manager.cleanup()


class TestClaudeCacheControl:
    """LangGraph marks Claude's system prompt for Anthropic's cache.

    The marker follows the model's record (its cache mechanism), not its
    name: Claude on Anthropic's API gets it, a model on another API does not,
    whatever it is called. Each request is read off a fake provider API.
    """

    @pytest.fixture
    def agent_config(self):
        return AgentConfig(name="test_agent", prompt="You are a test agent")

    @pytest.fixture(autouse=True)
    def hermetic(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        for variable in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(variable, raising=False)

    @pytest.fixture
    def claude(self, monkeypatch):
        """Claude's API, answering every request."""
        pytest.importorskip("langgraph")
        import anthropic

        from tests.workflow.adk.fakes import FakeAPI
        from tests.workflow.langchain_fakes import anthropic_reply, route

        monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
        api = FakeAPI(anthropic_reply({"type": "text", "text": "test response"}))
        route(monkeypatch, anthropic, "AsyncClient", api)
        return api

    async def test_claude_system_message_has_cache_control(self, agent_config, claude):
        manager = _langgraph_manager(
            agent_config, default_model="claude-sonnet-4-5", prompt_caching_enabled=True
        )

        await _turn(manager)

        (body,) = claude.bodies
        assert body["system"] == [{
            "type": "text",
            "text": "You are a test agent",
            "cache_control": {"type": "ephemeral"},
        }]

    async def test_cache_control_disabled_when_setting_false(self, agent_config, claude):
        manager = _langgraph_manager(
            agent_config, default_model="claude-sonnet-4-5", prompt_caching_enabled=False
        )

        await _turn(manager)

        (body,) = claude.bodies
        assert body["system"] == "You are a test agent"

    async def test_no_cache_control_for_gemini(self, agent_config, monkeypatch):
        pytest.importorskip("langgraph")
        from tests.workflow.adk.fakes import FakeAPI
        from tests.workflow.langchain_fakes import gemini_reply, route_google

        monkeypatch.setenv("GOOGLE_API_KEY", "g-test-key")
        api = FakeAPI(gemini_reply({"text": "test response"}))
        route_google(monkeypatch, api)
        manager = _langgraph_manager(
            agent_config, default_model="gemini-2.5-pro", prompt_caching_enabled=True
        )

        await _turn(manager)

        (request,) = api.requests
        assert b"cache_control" not in request.content
        assert api.bodies[0]["systemInstruction"]["parts"] == [{"text": "You are a test agent"}]

    async def test_no_cache_control_for_a_claude_named_model_on_another_api(
        self, agent_config, monkeypatch
    ):
        """0.6 marked any model whose id started ``claude-``; a server of the
        user's that serves a model under that name does not take the marker."""
        pytest.importorskip("langgraph")
        import openai

        from tests.workflow.adk.fakes import FakeAPI, chat_completion
        from tests.workflow.langchain_fakes import route

        api = FakeAPI(chat_completion("test response"))
        route(monkeypatch, openai, "AsyncOpenAI", api)
        endpoints = {"proxy": {
            "base_url": "http://localhost:8000/v1",
            "models": {"claude-sonnet-4-5": {}},
        }}
        manager = _langgraph_manager(
            agent_config,
            model_endpoints=endpoints,
            default_model="proxy:claude-sonnet-4-5",
            prompt_caching_enabled=True,
        )

        await _turn(manager)

        (request,) = api.requests
        assert b"cache_control" not in request.content
        assert api.bodies[0]["messages"][0] == {
            "role": "system", "content": "You are a test agent",
        }


class TestLLMUsageEventCacheFields:
    """Tests for cache fields in WorkflowEvent.llm_usage()."""

    def test_llm_usage_event_with_cache_fields(self):
        """Verify llm_usage() includes cached_tokens and cache_creation_tokens
        in metadata and summary content."""
        event = WorkflowEvent.llm_usage(
            model="claude-sonnet-4",
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
            cached_tokens=800,
            cache_creation_tokens=500,
        )

        assert event.type == EventType.LLM_USAGE
        assert event.metadata["model"] == "claude-sonnet-4"
        assert event.metadata["prompt_tokens"] == 1000
        assert event.metadata["completion_tokens"] == 200
        assert event.metadata["total_tokens"] == 1200
        assert event.metadata["cached_tokens"] == 800
        assert event.metadata["cache_creation_tokens"] == 500

        # Check summary content includes cache info
        assert "cached=800" in event.content
        assert "cache_write=500" in event.content

    def test_llm_usage_event_without_cache_fields(self):
        """Verify llm_usage() works without cache fields (backward compat)."""
        event = WorkflowEvent.llm_usage(
            model="gemini-2.5-pro",
            prompt_tokens=500,
            completion_tokens=100,
            total_tokens=600,
        )

        assert "cache_creation_tokens" not in event.metadata
        assert "cached_tokens" not in event.metadata
        assert "cached=" not in event.content
        assert "cache_write=" not in event.content

    def test_llm_usage_event_with_only_cached_tokens(self):
        """Verify llm_usage() works with cached_tokens but no cache_creation_tokens."""
        event = WorkflowEvent.llm_usage(
            model="claude-sonnet-4",
            prompt_tokens=1000,
            cached_tokens=800,
        )

        assert event.metadata["cached_tokens"] == 800
        assert "cache_creation_tokens" not in event.metadata
        assert "cached=800" in event.content
        assert "cache_write=" not in event.content
