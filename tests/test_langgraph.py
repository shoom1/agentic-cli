"""Tests for LangGraph workflow manager and state definitions.

These tests verify the LangGraph orchestration implementation.
Tests are skipped if LangGraph dependencies are not installed.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig


# Check if LangGraph is available
try:
    from agentic_cli.workflow.langgraph_manager import LangGraphWorkflowManager
    from agentic_cli.workflow.langgraph_state import (
        AgentState,
        ResearchState,
        ApprovalState,
        FinanceResearchState,
        CheckpointData,
        add_messages,
    )

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


# Skip all tests in this module if LangGraph is not installed
pytestmark = pytest.mark.skipif(
    not LANGGRAPH_AVAILABLE,
    reason="LangGraph dependencies not installed. Install with: pip install agentic-cli[langgraph]",
)


class TestLangGraphState:
    """Tests for LangGraph state definitions."""

    def test_add_messages_with_list(self):
        """Test add_messages reducer with list input."""
        existing = [{"role": "user", "content": "Hello"}]
        new = [{"role": "assistant", "content": "Hi there"}]
        result = add_messages(existing, new)

        assert len(result) == 2
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "Hi there"

    def test_add_messages_with_single(self):
        """Test add_messages reducer with single message."""
        existing = [{"role": "user", "content": "Hello"}]
        new = {"role": "assistant", "content": "Hi there"}
        result = add_messages(existing, new)

        assert len(result) == 2
        assert result[1]["content"] == "Hi there"

    def test_checkpoint_data_to_dict(self):
        """Test CheckpointData serialization."""
        checkpoint = CheckpointData(
            state={"messages": [], "current_agent": "test"},
            checkpoint_id="cp-123",
            parent_checkpoint_id="cp-122",
            metadata={"step": 1},
        )
        data = checkpoint.to_dict()

        assert data["checkpoint_id"] == "cp-123"
        assert data["parent_checkpoint_id"] == "cp-122"
        assert data["metadata"]["step"] == 1
        assert "timestamp" in data

    def test_checkpoint_data_from_dict(self):
        """Test CheckpointData deserialization."""
        now = datetime.now(timezone.utc)
        data = {
            "state": {"messages": []},
            "timestamp": now.isoformat(),
            "checkpoint_id": "cp-123",
            "parent_checkpoint_id": None,
            "metadata": {},
        }
        checkpoint = CheckpointData.from_dict(data)

        assert checkpoint.checkpoint_id == "cp-123"
        assert checkpoint.state == {"messages": []}

    def test_checkpoint_data_roundtrip(self):
        """Test CheckpointData serialization roundtrip."""
        original = CheckpointData(
            state={"messages": [{"role": "user", "content": "test"}]},
            checkpoint_id="cp-456",
            metadata={"iteration": 3},
        )
        data = original.to_dict()
        restored = CheckpointData.from_dict(data)

        assert restored.checkpoint_id == original.checkpoint_id
        assert restored.state == original.state
        assert restored.metadata == original.metadata


class TestLangGraphWorkflowManagerCreation:
    """Tests for LangGraphWorkflowManager initialization."""

    @pytest.fixture
    def settings(self):
        """Create test settings with mock API key."""
        return BaseSettings(
            google_api_key="test-key",
            orchestrator="langgraph",
            langgraph_checkpointer="memory",
        )

    @pytest.fixture
    def agent_configs(self):
        """Create test agent configs."""
        return [
            AgentConfig(
                name="test_agent",
                prompt="You are a test agent",
            )
        ]

    def test_manager_creation(self, settings, agent_configs):
        """Test basic manager creation."""
        manager = LangGraphWorkflowManager(
            agent_configs=agent_configs,
            settings=settings,
        )

        assert manager.app_name == settings.app_name
        assert not manager.is_initialized

    def test_manager_with_model_override(self, settings, agent_configs):
        """Test manager creation with model override."""
        manager = LangGraphWorkflowManager(
            agent_configs=agent_configs,
            settings=settings,
            model="gemini-2.5-flash",
        )

        assert manager.model == "gemini-2.5-flash"

    def test_manager_model_resolved_from_settings(self, agent_configs):
        """Test model is resolved from settings when not overridden."""
        # Need to actually set the API key in environment for model resolution
        import os
        os.environ["GOOGLE_API_KEY"] = "test-key"
        try:
            # Create fresh settings AFTER setting env var
            fresh_settings = BaseSettings(
                google_api_key="test-key",
                orchestrator="langgraph",
            )
            manager = LangGraphWorkflowManager(
                agent_configs=agent_configs,
                settings=fresh_settings,
            )

            # Model should be resolved lazily to the default Google model
            model = manager.model
            assert model is not None
            assert model.startswith("gemini-")
        finally:
            # Clean up
            if "GOOGLE_API_KEY" in os.environ:
                del os.environ["GOOGLE_API_KEY"]

    def test_manager_settings_property(self, settings, agent_configs):
        """Test settings property returns correct instance."""
        manager = LangGraphWorkflowManager(
            agent_configs=agent_configs,
            settings=settings,
        )

        assert manager.settings is settings

    def test_manager_update_settings(self, agent_configs):
        """Test updating settings."""
        settings1 = BaseSettings(google_api_key="key1")
        settings2 = BaseSettings(google_api_key="key2")

        manager = LangGraphWorkflowManager(
            agent_configs=agent_configs,
            settings=settings1,
        )

        manager.update_settings(settings2)
        assert manager.settings is settings2


class TestLangGraphWorkflowManagerLifecycle:
    """Tests for LangGraphWorkflowManager lifecycle operations."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return BaseSettings(
            google_api_key="test-key",
            orchestrator="langgraph",
        )

    @pytest.fixture
    def agent_configs(self):
        """Create test agent configs."""
        return [
            AgentConfig(
                name="test_agent",
                prompt="Test prompt",
            )
        ]

    @pytest.fixture
    def manager(self, settings, agent_configs):
        """Create test manager."""
        return LangGraphWorkflowManager(
            agent_configs=agent_configs,
            settings=settings,
        )

    async def test_cleanup(self, manager):
        """Test cleanup resets state."""
        # Set some state
        manager._initialized = True
        manager._graph = MagicMock()

        await manager.cleanup()

        assert not manager.is_initialized
        assert manager._graph is None
        assert manager._compiled_graph is None

    async def test_context_manager(self, settings, agent_configs):
        """Test async context manager."""
        manager = LangGraphWorkflowManager(
            agent_configs=agent_configs,
            settings=settings,
        )

        # Mock initialize_services to avoid actual LangGraph setup
        with patch.object(manager, "initialize_services") as mock_init:
            with patch.object(manager, "cleanup") as mock_cleanup:
                async with manager:
                    mock_init.assert_called_once()

                mock_cleanup.assert_called_once()


class TestLangGraphUserInput:
    """Tests for user input handling in LangGraph manager."""

    @pytest.fixture
    def manager(self):
        """Create test manager."""
        settings = BaseSettings(google_api_key="test-key")
        configs = [AgentConfig(name="test", prompt="test")]
        return LangGraphWorkflowManager(
            agent_configs=configs,
            settings=settings,
        )

    def test_has_pending_input_empty(self, manager):
        """Test has_pending_input when no requests."""
        assert not manager.has_pending_input()

    def test_get_pending_input_request_empty(self, manager):
        """Test get_pending_input_request when no requests."""
        assert manager.get_pending_input_request() is None

    def test_provide_user_input_unknown(self, manager):
        """Test provide_user_input with unknown request ID."""
        result = manager.provide_user_input("unknown-id", "response")
        assert result is False


class TestOrchestratorSelection:
    """Tests for orchestrator selection via settings."""

    def test_settings_default_orchestrator(self):
        """Test default orchestrator is ADK."""
        settings = BaseSettings(google_api_key="test-key")
        assert settings.orchestrator == "adk"

    def test_settings_langgraph_orchestrator(self):
        """Test setting LangGraph orchestrator."""
        settings = BaseSettings(
            google_api_key="test-key",
            orchestrator="langgraph",
        )
        assert settings.orchestrator == "langgraph"

    def test_settings_langgraph_checkpointer(self):
        """Test LangGraph checkpointer setting."""
        settings = BaseSettings(
            google_api_key="test-key",
            langgraph_checkpointer="postgres",
        )
        assert settings.langgraph_checkpointer == "postgres"


class TestCreateWorkflowManagerFromSettings:
    """Tests for the factory function."""

    def test_factory_creates_adk_manager(self):
        """Test factory creates ADK manager by default."""
        from agentic_cli.cli.app import create_workflow_manager_from_settings

        settings = BaseSettings(
            google_api_key="test-key",
            orchestrator="adk",
        )
        configs = [AgentConfig(name="test", prompt="test")]

        manager = create_workflow_manager_from_settings(configs, settings)

        # Should be GoogleADKWorkflowManager (ADK)
        from agentic_cli.workflow.adk_manager import GoogleADKWorkflowManager

        assert isinstance(manager, GoogleADKWorkflowManager)

    def test_factory_creates_langgraph_manager(self):
        """Test factory creates LangGraph manager when configured."""
        from agentic_cli.cli.app import create_workflow_manager_from_settings

        settings = BaseSettings(
            google_api_key="test-key",
            orchestrator="langgraph",
        )
        configs = [AgentConfig(name="test", prompt="test")]

        manager = create_workflow_manager_from_settings(configs, settings)

        assert isinstance(manager, LangGraphWorkflowManager)
