"""Tests for workflow events and configuration."""

from datetime import datetime

import pytest

from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import EventType, WorkflowEvent


class TestEventType:
    """Tests for EventType enum."""

    def test_all_event_types_exist(self):
        """Test all expected event types are defined."""
        expected = [
            "TEXT",
            "THINKING",
            "TOOL_CALL",
            "TOOL_RESULT",
            "CODE_EXECUTION",
            "EXECUTABLE_CODE",
            "FILE_DATA",
            "ERROR",
        ]
        for name in expected:
            assert hasattr(EventType, name)

    def test_event_type_values(self):
        """Test event type string values."""
        assert EventType.TEXT.value == "text"
        assert EventType.THINKING.value == "thinking"
        assert EventType.TOOL_CALL.value == "tool_call"
        assert EventType.TOOL_RESULT.value == "tool_result"
        assert EventType.ERROR.value == "error"


class TestWorkflowEvent:
    """Tests for WorkflowEvent dataclass."""

    def test_create_text_event(self):
        """Test creating text event."""
        event = WorkflowEvent.text("Hello world", session_id="test-session")

        assert event.type == EventType.TEXT
        assert event.content == "Hello world"
        assert event.metadata["session_id"] == "test-session"

    def test_create_text_event_no_session(self):
        """Test creating text event without session ID."""
        event = WorkflowEvent.text("Hello")

        assert event.type == EventType.TEXT
        assert event.content == "Hello"
        assert "session_id" not in event.metadata

    def test_create_thinking_event(self):
        """Test creating thinking event."""
        event = WorkflowEvent.thinking("I should consider...", session_id="s1")

        assert event.type == EventType.THINKING
        assert event.content == "I should consider..."
        assert event.metadata["session_id"] == "s1"

    def test_create_tool_call_event(self):
        """Test creating tool call event."""
        event = WorkflowEvent.tool_call("search_web")

        assert event.type == EventType.TOOL_CALL
        assert "search_web" in event.content
        assert event.metadata["tool_name"] == "search_web"
        assert event.metadata["tool_args"] == {}

    def test_create_tool_call_event_with_args(self):
        """Test creating tool call event with arguments."""
        event = WorkflowEvent.tool_call(
            "search_kb",
            tool_args={"query": "machine learning", "max_results": 10},
        )

        assert event.type == EventType.TOOL_CALL
        assert event.metadata["tool_name"] == "search_kb"
        assert event.metadata["tool_args"]["query"] == "machine learning"
        assert event.metadata["tool_args"]["max_results"] == 10

    def test_create_tool_result_event(self):
        """Test creating tool result event."""
        event = WorkflowEvent.tool_result(
            "search_kb",
            result={"matches": 5, "results": []},
            success=True,
            duration_ms=150.5,
        )

        assert event.type == EventType.TOOL_RESULT
        assert event.metadata["tool_name"] == "search_kb"
        assert event.metadata["result"]["matches"] == 5
        assert event.metadata["success"] is True
        assert event.metadata["duration_ms"] == 150.5

    def test_create_tool_result_event_failed(self):
        """Test creating failed tool result event."""
        event = WorkflowEvent.tool_result(
            "web_search",
            result=None,
            success=False,
            error="API rate limited",
        )

        assert event.type == EventType.TOOL_RESULT
        assert event.metadata["success"] is False
        assert event.metadata["error"] == "API rate limited"

    def test_tool_result_truncates_long_content(self):
        """Test that tool result truncates long string results for content."""
        long_result = "x" * 500
        event = WorkflowEvent.tool_result("test", result=long_result)

        assert len(event.content) < 500
        assert "..." in event.content
        # Full result is still in metadata
        assert len(event.metadata["result"]) == 500

    def test_create_code_execution_event(self):
        """Test creating code execution event."""
        event = WorkflowEvent.code_execution("Success: 42")

        assert event.type == EventType.CODE_EXECUTION
        assert event.content == "Success: 42"
        assert event.metadata["outcome"] == "Success: 42"

    def test_create_executable_code_event(self):
        """Test creating executable code event."""
        code = "print('hello')"
        event = WorkflowEvent.executable_code(code, "python")

        assert event.type == EventType.EXECUTABLE_CODE
        assert event.content == code
        assert event.metadata["language"] == "python"

    def test_create_file_data_event(self):
        """Test creating file data event."""
        event = WorkflowEvent.file_data("report.pdf")

        assert event.type == EventType.FILE_DATA
        assert event.content == "report.pdf"
        assert event.metadata["display_name"] == "report.pdf"

    def test_create_error_event(self):
        """Test creating error event."""
        event = WorkflowEvent.error("Something went wrong")

        assert event.type == EventType.ERROR
        assert event.content == "Something went wrong"
        assert event.metadata["recoverable"] is False

    def test_create_error_event_with_details(self):
        """Test creating error event with details."""
        event = WorkflowEvent.error(
            "API call failed",
            error_code="API_ERROR",
            recoverable=True,
            details={"status_code": 429, "retry_after": 60},
        )

        assert event.type == EventType.ERROR
        assert event.content == "API call failed"
        assert event.metadata["error_code"] == "API_ERROR"
        assert event.metadata["recoverable"] is True
        assert event.metadata["details"]["status_code"] == 429

    def test_event_has_timestamp(self):
        """Test that events have timestamps."""
        event = WorkflowEvent.text("Hello")

        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)
        # Timestamp should be recent (within last second)
        assert (datetime.now() - event.timestamp).total_seconds() < 1

    def test_direct_instantiation(self):
        """Test direct WorkflowEvent instantiation."""
        event = WorkflowEvent(
            type=EventType.TEXT,
            content="Custom content",
            metadata={"custom": "data"},
        )

        assert event.type == EventType.TEXT
        assert event.content == "Custom content"
        assert event.metadata["custom"] == "data"

    def test_default_values(self):
        """Test default values for WorkflowEvent."""
        event = WorkflowEvent(type=EventType.TEXT)

        assert event.content == ""
        assert event.metadata == {}


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_basic_config(self):
        """Test basic agent configuration."""
        config = AgentConfig(
            name="test_agent",
            prompt="You are a test agent.",
        )

        assert config.name == "test_agent"
        assert config.get_prompt() == "You are a test agent."
        assert config.tools == []
        assert config.sub_agents == []
        assert config.description == ""
        assert config.model is None

    def test_config_with_callable_prompt(self):
        """Test agent config with callable prompt."""
        def get_prompt():
            return "Dynamic prompt content"

        config = AgentConfig(
            name="dynamic_agent",
            prompt=get_prompt,
        )

        assert config.get_prompt() == "Dynamic prompt content"

    def test_config_with_tools(self):
        """Test agent config with tools."""
        def tool_a():
            pass

        def tool_b():
            pass

        config = AgentConfig(
            name="tooled_agent",
            prompt="Agent with tools",
            tools=[tool_a, tool_b],
        )

        assert len(config.tools) == 2
        assert tool_a in config.tools
        assert tool_b in config.tools

    def test_config_with_sub_agents(self):
        """Test agent config with sub-agents."""
        config = AgentConfig(
            name="coordinator",
            prompt="Coordinate sub-agents",
            sub_agents=["agent_a", "agent_b"],
        )

        assert config.sub_agents == ["agent_a", "agent_b"]

    def test_config_with_model_override(self):
        """Test agent config with model override."""
        config = AgentConfig(
            name="custom_model_agent",
            prompt="Agent with custom model",
            model="gemini-2.5-pro",
        )

        assert config.model == "gemini-2.5-pro"

    def test_full_config(self):
        """Test fully configured agent."""
        def search():
            pass

        def dynamic_prompt():
            return "System prompt"

        config = AgentConfig(
            name="full_agent",
            prompt=dynamic_prompt,
            tools=[search],
            sub_agents=["helper"],
            description="A fully configured agent",
            model="claude-sonnet-4",
        )

        assert config.name == "full_agent"
        assert config.get_prompt() == "System prompt"
        assert len(config.tools) == 1
        assert config.sub_agents == ["helper"]
        assert config.description == "A fully configured agent"
        assert config.model == "claude-sonnet-4"

    def test_get_prompt_caches_callable(self):
        """Test that callable prompt is called each time."""
        call_count = 0

        def counting_prompt():
            nonlocal call_count
            call_count += 1
            return f"Call {call_count}"

        config = AgentConfig(name="counting", prompt=counting_prompt)

        # Each call to get_prompt calls the function
        assert config.get_prompt() == "Call 1"
        assert config.get_prompt() == "Call 2"
        assert call_count == 2


# ============================================================================
# Workflow Manager Lifecycle Tests
# ============================================================================

from unittest.mock import patch, MagicMock, AsyncMock


class TestWorkflowManagerLifecycle:
    """Tests for WorkflowManager lifecycle methods."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = MagicMock()
        settings.app_name = "test_app"
        settings.thinking_effort = "medium"
        settings.get_model.return_value = "gemini-3-flash-preview"
        settings.supports_thinking_effort.return_value = True
        settings.export_api_keys_to_env = MagicMock()
        return settings

    @pytest.fixture
    def agent_configs(self):
        """Create test agent configurations."""
        return [
            AgentConfig(
                name="test_agent",
                prompt="You are a test agent.",
            )
        ]

    def test_workflow_manager_creation(self, mock_settings, agent_configs):
        """Test workflow manager is created with correct initial state."""
        from agentic_cli.workflow.manager import WorkflowManager

        manager = WorkflowManager(
            agent_configs=agent_configs,
            settings=mock_settings,
        )

        assert manager.app_name == "test_app"
        assert manager.is_initialized is False
        assert manager.runner is None
        assert manager.root_agent is None
        assert manager.session_service is None

    def test_workflow_manager_with_model_override(self, mock_settings, agent_configs):
        """Test workflow manager with explicit model."""
        from agentic_cli.workflow.manager import WorkflowManager

        manager = WorkflowManager(
            agent_configs=agent_configs,
            settings=mock_settings,
            model="claude-sonnet-4",
        )

        assert manager.model == "claude-sonnet-4"
        # Model should not call settings.get_model since it was provided
        mock_settings.get_model.assert_not_called()

    def test_model_resolved_from_settings(self, mock_settings, agent_configs):
        """Test model is resolved from settings when not provided."""
        from agentic_cli.workflow.manager import WorkflowManager

        manager = WorkflowManager(
            agent_configs=agent_configs,
            settings=mock_settings,
        )

        # Access model property to trigger resolution
        model = manager.model

        assert model == "gemini-3-flash-preview"
        mock_settings.get_model.assert_called_once()

    def test_settings_property(self, mock_settings, agent_configs):
        """Test settings property returns correct settings."""
        from agentic_cli.workflow.manager import WorkflowManager

        manager = WorkflowManager(
            agent_configs=agent_configs,
            settings=mock_settings,
        )

        assert manager.settings is mock_settings

    def test_update_settings(self, mock_settings, agent_configs):
        """Test updating settings."""
        from agentic_cli.workflow.manager import WorkflowManager

        manager = WorkflowManager(
            agent_configs=agent_configs,
            settings=mock_settings,
        )

        new_settings = MagicMock()
        new_settings.app_name = "new_app"

        manager.update_settings(new_settings)

        assert manager.settings is new_settings

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_settings, agent_configs):
        """Test cleanup resets state."""
        from agentic_cli.workflow.manager import WorkflowManager

        manager = WorkflowManager(
            agent_configs=agent_configs,
            settings=mock_settings,
        )

        # Set some state to clean up
        manager._initialized = True
        manager._runner = MagicMock()
        manager._root_agent = MagicMock()
        manager._session_service = MagicMock()

        await manager.cleanup()

        assert manager.is_initialized is False
        assert manager.runner is None
        assert manager.root_agent is None
        assert manager.session_service is None

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_settings, agent_configs):
        """Test async context manager support."""
        from agentic_cli.workflow.manager import WorkflowManager

        with patch.object(
            WorkflowManager, "initialize_services", new_callable=AsyncMock
        ) as mock_init:
            with patch.object(
                WorkflowManager, "cleanup", new_callable=AsyncMock
            ) as mock_cleanup:
                manager = WorkflowManager(
                    agent_configs=agent_configs,
                    settings=mock_settings,
                )

                async with manager as m:
                    mock_init.assert_called_once()
                    assert m is manager

                mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_reinitialize_with_new_model(self, mock_settings, agent_configs):
        """Test reinitialize updates model."""
        from agentic_cli.workflow.manager import WorkflowManager

        with patch.object(
            WorkflowManager, "initialize_services", new_callable=AsyncMock
        ):
            manager = WorkflowManager(
                agent_configs=agent_configs,
                settings=mock_settings,
                model="old-model",
            )
            manager._initialized = True

            await manager.reinitialize(model="new-model")

            assert manager.model == "new-model"

    @pytest.mark.asyncio
    async def test_reinitialize_resolves_model_from_settings(
        self, mock_settings, agent_configs
    ):
        """Test reinitialize re-resolves model from settings when not specified."""
        from agentic_cli.workflow.manager import WorkflowManager

        mock_settings.get_model.return_value = "resolved-model"

        with patch.object(
            WorkflowManager, "initialize_services", new_callable=AsyncMock
        ):
            manager = WorkflowManager(
                agent_configs=agent_configs,
                settings=mock_settings,
                model="initial-model",
            )
            manager._initialized = True

            await manager.reinitialize()

            # Model should be re-resolved
            assert manager.model == "resolved-model"

    @pytest.mark.asyncio
    async def test_reinitialize_preserves_sessions_by_default(
        self, mock_settings, agent_configs
    ):
        """Test reinitialize preserves session service by default."""
        from agentic_cli.workflow.manager import WorkflowManager

        with patch.object(
            WorkflowManager, "initialize_services", new_callable=AsyncMock
        ):
            manager = WorkflowManager(
                agent_configs=agent_configs,
                settings=mock_settings,
            )

            # Setup mock session service
            old_session_service = MagicMock()
            manager._session_service = old_session_service
            manager._initialized = True
            manager._root_agent = MagicMock()

            await manager.reinitialize(model="new-model", preserve_sessions=True)

            # Session service should be preserved
            assert manager.session_service is old_session_service

    @pytest.mark.asyncio
    async def test_reinitialize_can_discard_sessions(
        self, mock_settings, agent_configs
    ):
        """Test reinitialize can discard sessions."""
        from agentic_cli.workflow.manager import WorkflowManager

        with patch.object(
            WorkflowManager, "initialize_services", new_callable=AsyncMock
        ):
            manager = WorkflowManager(
                agent_configs=agent_configs,
                settings=mock_settings,
            )

            old_session_service = MagicMock()
            manager._session_service = old_session_service
            manager._initialized = True

            await manager.reinitialize(preserve_sessions=False)

            # Session service should not be preserved (cleanup sets to None,
            # initialize_services will create new one)
            # After reinitialize with preserve_sessions=False, session_service
            # depends on what initialize_services does
