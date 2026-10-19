"""Tests for LangGraph workflow manager and state definitions.

These tests verify the LangGraph orchestration implementation.
Tests are skipped if LangGraph dependencies are not installed.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig


# Check if LangGraph is available - use new import paths
try:
    from agentic_cli.workflow.langgraph import LangGraphWorkflowManager
    from agentic_cli.workflow.langgraph.state import (
        AgentState,
        ResearchState,
        ApprovalState,
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

        # LangGraph's add_messages auto-converts dicts to LangChain message objects
        assert len(result) == 2
        assert result[0].content == "Hello"
        assert result[1].content == "Hi there"

    def test_add_messages_with_single(self):
        """Test add_messages reducer with single message."""
        existing = [{"role": "user", "content": "Hello"}]
        new = {"role": "assistant", "content": "Hi there"}
        result = add_messages(existing, new)

        assert len(result) == 2
        assert result[1].content == "Hi there"

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
            session_store="memory",
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

        assert manager.model == "google:gemini-2.5-flash"

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

            # Without a default_model, the newest default-tier Google model.
            assert manager.model.startswith("google:gemini-")
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


def _resolved(agent: str, spec: str, settings: BaseSettings):
    """``spec`` resolved for ``agent`` against the shipped table, as the manager does."""
    from agentic_cli.models import ModelCatalog, RoleMap, resolve_model
    from agentic_cli.workflow import model_services

    endpoints = model_services.endpoints_from_settings(settings)
    return resolve_model(
        agent=agent,
        spec=spec,
        generation=None,
        thinking_effort=settings.thinking_effort,
        roles=RoleMap(default_model=None, model_roles={}),
        catalog=ModelCatalog(endpoints=endpoints),
        endpoints=endpoints,
    )


class TestLangGraphBuilder:
    """Tests for LangGraph graph construction."""

    def test_build_minimal_graph(self):
        """A single-agent graph can be built and compiled."""
        from agentic_cli.workflow.langgraph.graph_builder import LangGraphBuilder

        settings = BaseSettings(google_api_key="test-key")
        builder = LangGraphBuilder(settings)

        graph = builder.build(
            [AgentConfig(name="test_agent", prompt="Test prompt")],
            {"test_agent": _resolved("test_agent", "gemini-2.5-flash", settings)},
        )

        assert graph is not None
        assert graph.compile() is not None

    def test_each_agent_gets_its_own_model(self):
        """One chat model per agent, built when the graph is built."""
        from langchain_anthropic import ChatAnthropic
        from langchain_google_genai import ChatGoogleGenerativeAI

        from agentic_cli.workflow.langgraph.graph_builder import LangGraphBuilder

        settings = BaseSettings(google_api_key="test-key", anthropic_api_key="test-key")
        builder = LangGraphBuilder(settings)

        builder.build(
            [
                AgentConfig(name="lead", prompt="Lead.", sub_agents=["helper"]),
                AgentConfig(name="helper", prompt="Help."),
            ],
            {
                "lead": _resolved("lead", "claude-sonnet-4-5", settings),
                "helper": _resolved("helper", "gemini-2.5-flash", settings),
            },
        )

        assert type(builder.chat_models["lead"]) is ChatAnthropic
        assert type(builder.chat_models["helper"]) is ChatGoogleGenerativeAI

    def test_an_agent_without_a_resolved_model_is_an_error(self):
        from agentic_cli.workflow.langgraph.graph_builder import LangGraphBuilder

        settings = BaseSettings(google_api_key="test-key")

        with pytest.raises(ValueError, match="No resolved model for agent 'test_agent'"):
            LangGraphBuilder(settings).build([AgentConfig(name="test_agent", prompt="p")], {})


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

    def test_settings_session_store(self):
        """The unified session_store drives LangGraph persistence."""
        settings = BaseSettings(
            google_api_key="test-key",
            session_store="postgres",
        )
        assert settings.session_store == "postgres"


class TestCreateWorkflowManagerFromSettings:
    """Tests for the factory function."""

    def test_factory_creates_adk_manager(self):
        """Test factory creates ADK manager by default."""
        from agentic_cli.workflow.factory import create_workflow_manager_from_settings

        settings = BaseSettings(
            google_api_key="test-key",
            orchestrator="adk",
        )
        configs = [AgentConfig(name="test", prompt="test")]

        manager = create_workflow_manager_from_settings(configs, settings)

        # Should be GoogleADKWorkflowManager (ADK)
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        assert isinstance(manager, GoogleADKWorkflowManager)

    def test_factory_creates_langgraph_manager(self):
        """Test factory creates LangGraph manager when configured."""
        from agentic_cli.workflow.factory import create_workflow_manager_from_settings

        settings = BaseSettings(
            google_api_key="test-key",
            orchestrator="langgraph",
        )
        configs = [AgentConfig(name="test", prompt="test")]

        manager = create_workflow_manager_from_settings(configs, settings)

        assert isinstance(manager, LangGraphWorkflowManager)


class TestBackwardCompatibility:
    """Tests for backward-compatible imports."""

    def test_workflow_module_exports(self):
        """Test that workflow module exports LangGraph components."""
        from agentic_cli.workflow import (
            LangGraphWorkflowManager as WorkflowManager,
            AgentState as WorkflowAgentState,
        )

        assert WorkflowManager is LangGraphWorkflowManager
        assert WorkflowAgentState is AgentState


class TestPersistenceLayer:
    """Tests for the persistence layer factories."""

    async def test_create_checkpointer_memory(self):
        """Test creating memory checkpointer (no connection to close)."""
        from agentic_cli.workflow.langgraph.persistence import create_checkpointer

        settings = BaseSettings(google_api_key="test-key")
        checkpointer, cm = await create_checkpointer("memory", settings)

        assert checkpointer is not None
        assert cm is None

    async def test_create_checkpointer_none(self):
        """Test disabling checkpointer."""
        from agentic_cli.workflow.langgraph.persistence import create_checkpointer

        settings = BaseSettings(google_api_key="test-key")
        checkpointer, cm = await create_checkpointer(None, settings)

        assert checkpointer is None
        assert cm is None

    async def test_create_checkpointer_sqlite_returns_real_saver(self, tmp_path):
        """sqlite checkpointer returns a usable AsyncSqliteSaver, not the
        context manager (the old bug), and provides a cm to close."""
        pytest.importorskip("langgraph.checkpoint.sqlite.aio")
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from langgraph.checkpoint.base import BaseCheckpointSaver
        from agentic_cli.workflow.langgraph.persistence import create_checkpointer

        settings = BaseSettings(workspace_dir=tmp_path, google_api_key="test-key")
        checkpointer, cm = await create_checkpointer("sqlite", settings)
        try:
            # A real saver — previously this was an _AsyncGeneratorContextManager.
            assert isinstance(checkpointer, AsyncSqliteSaver)
            assert isinstance(checkpointer, BaseCheckpointSaver)
            assert cm is not None  # owns the connection; must be closed
            # setup() ran, so the db file exists at the default workspace path.
            assert (tmp_path / "checkpoints.db").exists()
        finally:
            await cm.__aexit__(None, None, None)

    def test_create_store_memory(self):
        """Test creating memory store."""
        from agentic_cli.workflow.langgraph.persistence import create_store

        settings = BaseSettings(google_api_key="test-key")
        store = create_store("memory", settings)

        assert store is not None

    def test_create_store_none(self):
        """Test disabling store."""
        from agentic_cli.workflow.langgraph.persistence import create_store

        settings = BaseSettings(google_api_key="test-key")
        store = create_store(None, settings)

        assert store is None


class TestManagerCheckpointerOptions:
    """Tests for checkpointer options in LangGraphWorkflowManager."""

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
        return [AgentConfig(name="test", prompt="test")]

    def test_manager_checkpointer_sqlite(self, settings, agent_configs):
        """Test manager accepts sqlite checkpointer type."""
        manager = LangGraphWorkflowManager(
            agent_configs=agent_configs,
            settings=settings,
            checkpointer="sqlite",
        )

        assert manager._checkpointer_type == "sqlite"
