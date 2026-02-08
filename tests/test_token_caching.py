"""Tests for token caching support.

Tests verify prompt caching configuration for Claude models in LangGraph,
cache-related fields in WorkflowEvent.llm_usage(), and the
prompt_caching_enabled setting.
"""

import pytest
from unittest.mock import MagicMock

from agentic_cli.config import BaseSettings
from agentic_cli.workflow.events import WorkflowEvent, EventType

# Check if LangGraph and langchain_core are available
try:
    from agentic_cli.workflow.langgraph import LangGraphWorkflowManager
    from agentic_cli.workflow.config import AgentConfig
    from langchain_core.messages import SystemMessage

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


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


@pytest.mark.skipif(
    not LANGGRAPH_AVAILABLE,
    reason="LangGraph dependencies not installed",
)
class TestClaudeCacheControl:
    """Tests for cache_control on Claude SystemMessages in LangGraph."""

    @pytest.fixture
    def agent_config(self):
        return AgentConfig(name="test_agent", prompt="You are a test agent")

    def test_claude_system_message_has_cache_control(self, agent_config):
        """When prompt_caching_enabled=True and model is claude-*, verify
        SystemMessage content is a list with cache_control."""
        settings = BaseSettings(
            anthropic_api_key="test-key",
            prompt_caching_enabled=True,
        )
        manager = LangGraphWorkflowManager(
            agent_configs=[agent_config],
            settings=settings,
            model="claude-sonnet-4",
        )

        # Call _create_agent_node to get the node function, then inspect
        # what SystemMessage it would build. We test the logic by calling
        # the inner function with a mock LLM.
        node_fn = manager._create_agent_node(agent_config)

        # We need to intercept the SystemMessage creation.
        # The simplest way: patch _get_llm_for_model to capture messages.
        captured_messages = []
        original_get_llm = manager._get_llm_for_model

        def mock_get_llm(model_name):
            mock_llm = MagicMock()
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)

            async def mock_ainvoke(messages, **kwargs):
                captured_messages.extend(messages)
                mock_response = MagicMock()
                mock_response.content = "test response"
                mock_response.tool_calls = []
                return mock_response

            mock_llm.ainvoke = mock_ainvoke
            return mock_llm

        manager._get_llm_for_model = mock_get_llm

        # Run the node
        import asyncio

        state = {"messages": [{"role": "user", "content": "hello"}]}
        asyncio.get_event_loop().run_until_complete(node_fn(state))

        # Check the SystemMessage
        sys_msgs = [m for m in captured_messages if isinstance(m, SystemMessage)]
        assert len(sys_msgs) == 1
        sys_msg = sys_msgs[0]

        # Content should be a list with cache_control
        assert isinstance(sys_msg.content, list)
        assert len(sys_msg.content) == 1
        block = sys_msg.content[0]
        assert block["type"] == "text"
        assert block["text"] == "You are a test agent"
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_disabled_when_setting_false(self, agent_config):
        """When prompt_caching_enabled=False, verify SystemMessage uses plain string."""
        settings = BaseSettings(
            anthropic_api_key="test-key",
            prompt_caching_enabled=False,
        )
        manager = LangGraphWorkflowManager(
            agent_configs=[agent_config],
            settings=settings,
            model="claude-sonnet-4",
        )

        captured_messages = []

        def mock_get_llm(model_name):
            mock_llm = MagicMock()
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)

            async def mock_ainvoke(messages, **kwargs):
                captured_messages.extend(messages)
                mock_response = MagicMock()
                mock_response.content = "test response"
                mock_response.tool_calls = []
                return mock_response

            mock_llm.ainvoke = mock_ainvoke
            return mock_llm

        manager._get_llm_for_model = mock_get_llm

        import asyncio

        state = {"messages": [{"role": "user", "content": "hello"}]}
        asyncio.get_event_loop().run_until_complete(
            manager._create_agent_node(agent_config)(state)
        )

        sys_msgs = [m for m in captured_messages if isinstance(m, SystemMessage)]
        assert len(sys_msgs) == 1
        # Plain string content, no cache_control
        assert isinstance(sys_msgs[0].content, str)
        assert sys_msgs[0].content == "You are a test agent"

    def test_no_cache_control_for_gemini(self, agent_config):
        """When model is gemini-*, verify SystemMessage uses plain string
        regardless of prompt_caching_enabled setting."""
        settings = BaseSettings(
            google_api_key="test-key",
            prompt_caching_enabled=True,
        )
        manager = LangGraphWorkflowManager(
            agent_configs=[agent_config],
            settings=settings,
            model="gemini-2.5-pro",
        )

        captured_messages = []

        def mock_get_llm(model_name):
            mock_llm = MagicMock()
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)

            async def mock_ainvoke(messages, **kwargs):
                captured_messages.extend(messages)
                mock_response = MagicMock()
                mock_response.content = "test response"
                mock_response.tool_calls = []
                return mock_response

            mock_llm.ainvoke = mock_ainvoke
            return mock_llm

        manager._get_llm_for_model = mock_get_llm

        import asyncio

        state = {"messages": [{"role": "user", "content": "hello"}]}
        asyncio.get_event_loop().run_until_complete(
            manager._create_agent_node(agent_config)(state)
        )

        sys_msgs = [m for m in captured_messages if isinstance(m, SystemMessage)]
        assert len(sys_msgs) == 1
        # Plain string, no cache_control for Gemini
        assert isinstance(sys_msgs[0].content, str)


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
