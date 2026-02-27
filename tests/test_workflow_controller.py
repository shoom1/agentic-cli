"""Tests for workflow controller factory and orchestrator routing.

Verifies that Claude models are automatically routed to the LangGraph
orchestrator, while Gemini models use whichever orchestrator is configured.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_cli.cli.workflow_controller import WorkflowController
from agentic_cli.workflow.factory import (
    _is_claude_model,
    _resolve_effective_model,
    create_workflow_manager_from_settings,
)
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.settings import OrchestratorType


# --- Helpers ---


@pytest.fixture
def agent_configs():
    """Minimal agent config list for factory calls."""
    return [AgentConfig(name="test", prompt="You are a test agent.")]


def _make_settings(orchestrator=OrchestratorType.ADK, default_model=None, **extra):
    """Create a mock settings object with required attributes."""
    settings = MagicMock()
    settings.orchestrator = orchestrator
    settings.default_model = default_model
    settings.app_name = "test-app"
    settings.langgraph_checkpointer = "memory"
    for k, v in extra.items():
        setattr(settings, k, v)
    return settings


class _FakeADKWorkflow:
    """Fake ADK workflow manager for testing (class name matters for swap detection)."""

    def __init__(self, model="gemini-2.5-pro"):
        self.model = model
        self.reinitialize = AsyncMock()
        self.initialize_services = AsyncMock()


class _FakeLangGraphWorkflow:
    """Fake LangGraph workflow manager (class name = LangGraphWorkflowManager)."""

    def __init__(self, model="claude-sonnet-4-5"):
        self.model = model
        self.reinitialize = AsyncMock()
        self.initialize_services = AsyncMock()


# Rename the class so type().__name__ matches what the code checks
_FakeLangGraphWorkflow.__name__ = "LangGraphWorkflowManager"
_FakeADKWorkflow.__name__ = "GoogleADKWorkflowManager"


# --- Unit tests for helpers ---


class TestIsClaudeModel:
    def test_claude_models(self):
        assert _is_claude_model("claude-sonnet-4-5") is True
        assert _is_claude_model("claude-opus-4") is True
        assert _is_claude_model("claude-3-haiku") is True

    def test_non_claude_models(self):
        assert _is_claude_model("gemini-2.5-pro") is False
        assert _is_claude_model("gpt-4") is False

    def test_none(self):
        assert _is_claude_model(None) is False


class TestResolveEffectiveModel:
    def test_explicit_model_takes_precedence(self):
        settings = _make_settings(default_model="gemini-2.5-pro")
        assert _resolve_effective_model("claude-sonnet-4", settings) == "claude-sonnet-4"

    def test_falls_back_to_settings(self):
        settings = _make_settings(default_model="claude-opus-4")
        assert _resolve_effective_model(None, settings) == "claude-opus-4"

    def test_returns_none_when_nothing_set(self):
        settings = _make_settings(default_model=None)
        assert _resolve_effective_model(None, settings) is None


# --- Factory routing tests ---


class TestCreateWorkflowManagerRouting:
    """Test that the factory routes Claude models to LangGraph."""

    def test_gemini_model_with_adk_returns_adk(self, agent_configs):
        """Gemini model + ADK orchestrator → ADK manager."""
        settings = _make_settings(orchestrator=OrchestratorType.ADK)
        with patch(
            "agentic_cli.workflow.adk.manager.GoogleADKWorkflowManager"
        ) as mock_adk_cls:
            result = create_workflow_manager_from_settings(
                agent_configs, settings, model="gemini-2.5-pro"
            )
        mock_adk_cls.assert_called_once()
        assert result is mock_adk_cls.return_value

    @patch("agentic_cli.workflow.langgraph.LangGraphWorkflowManager")
    def test_claude_model_with_adk_returns_langgraph(
        self, mock_lg_cls, agent_configs
    ):
        """Claude model + ADK orchestrator → auto-switches to LangGraph."""
        settings = _make_settings(orchestrator=OrchestratorType.ADK)
        result = create_workflow_manager_from_settings(
            agent_configs, settings, model="claude-sonnet-4-5"
        )
        mock_lg_cls.assert_called_once()
        assert result is mock_lg_cls.return_value

    @patch("agentic_cli.workflow.langgraph.LangGraphWorkflowManager")
    def test_claude_model_in_settings_returns_langgraph(
        self, mock_lg_cls, agent_configs
    ):
        """Claude model in settings.default_model → auto-switches to LangGraph."""
        settings = _make_settings(
            orchestrator=OrchestratorType.ADK,
            default_model="claude-opus-4",
        )
        result = create_workflow_manager_from_settings(agent_configs, settings)
        mock_lg_cls.assert_called_once()
        assert result is mock_lg_cls.return_value

    @patch("agentic_cli.workflow.langgraph.LangGraphWorkflowManager")
    def test_langgraph_orchestrator_returns_langgraph(
        self, mock_lg_cls, agent_configs
    ):
        """LangGraph orchestrator setting → LangGraph manager (unchanged behavior)."""
        settings = _make_settings(orchestrator=OrchestratorType.LANGGRAPH)
        result = create_workflow_manager_from_settings(
            agent_configs, settings, model="gemini-2.5-pro"
        )
        mock_lg_cls.assert_called_once()
        assert result is mock_lg_cls.return_value

    def test_no_model_with_adk_returns_adk(self, agent_configs):
        """No model specified + ADK orchestrator → ADK manager."""
        settings = _make_settings(orchestrator=OrchestratorType.ADK, default_model=None)
        with patch(
            "agentic_cli.workflow.adk.manager.GoogleADKWorkflowManager"
        ) as mock_adk_cls:
            result = create_workflow_manager_from_settings(agent_configs, settings)
        mock_adk_cls.assert_called_once()
        assert result is mock_adk_cls.return_value


# --- WorkflowController orchestrator swap tests ---


class TestWorkflowControllerOrchestratorSwap:
    """Test runtime model switch triggers orchestrator swap when needed."""

    def _make_controller(self, orchestrator=OrchestratorType.ADK):
        configs = [AgentConfig(name="test", prompt="Test")]
        settings = _make_settings(orchestrator=orchestrator)
        return WorkflowController(configs, settings)

    def test_needs_swap_gemini_to_claude(self):
        """ADK manager + Claude model → needs swap."""
        controller = self._make_controller()
        controller._workflow = _FakeADKWorkflow()

        assert controller._needs_orchestrator_swap("claude-sonnet-4-5") is True

    def test_no_swap_gemini_to_gemini(self):
        """ADK manager + Gemini model → no swap needed."""
        controller = self._make_controller()
        controller._workflow = _FakeADKWorkflow()

        assert controller._needs_orchestrator_swap("gemini-2.5-pro") is False

    def test_no_swap_claude_to_claude(self):
        """LangGraph manager (auto) + Claude model → no swap needed."""
        controller = self._make_controller()
        controller._workflow = _FakeLangGraphWorkflow("claude-sonnet-4-5")

        assert controller._needs_orchestrator_swap("claude-opus-4") is False

    def test_needs_swap_claude_to_gemini(self):
        """LangGraph manager (auto) + Gemini model → needs swap back to ADK."""
        controller = self._make_controller()
        controller._workflow = _FakeLangGraphWorkflow("claude-sonnet-4-5")

        assert controller._needs_orchestrator_swap("gemini-2.5-pro") is True

    def test_no_swap_when_langgraph_orchestrator_and_gemini(self):
        """LangGraph orchestrator + Gemini model → no swap (user chose LangGraph)."""
        controller = self._make_controller(orchestrator=OrchestratorType.LANGGRAPH)
        controller._workflow = _FakeLangGraphWorkflow("gemini-2.5-pro")

        assert controller._needs_orchestrator_swap("gemini-2.5-flash") is False

    def test_no_swap_when_model_is_none(self):
        """No model specified → no swap."""
        controller = self._make_controller()
        controller._workflow = _FakeADKWorkflow()

        assert controller._needs_orchestrator_swap(None) is False

    def test_no_swap_when_workflow_is_none(self):
        """No workflow initialized → no swap."""
        controller = self._make_controller()
        assert controller._needs_orchestrator_swap("claude-sonnet-4-5") is False

    @patch("agentic_cli.workflow.langgraph.LangGraphWorkflowManager")
    async def test_reinitialize_swaps_orchestrator_for_claude(self, mock_lg_cls):
        """Switching from Gemini (ADK) to Claude triggers full manager replacement."""
        controller = self._make_controller()
        old_workflow = _FakeADKWorkflow("gemini-2.5-pro")
        controller._workflow = old_workflow

        new_workflow = AsyncMock()
        mock_lg_cls.return_value = new_workflow

        await controller.reinitialize(model="claude-sonnet-4-5")

        # Old workflow's reinitialize should NOT have been called
        old_workflow.reinitialize.assert_not_called()
        # New workflow should have been created and initialized
        new_workflow.initialize_services.assert_awaited_once()
        assert controller._workflow is new_workflow

    async def test_reinitialize_same_family_calls_existing_reinitialize(self):
        """Switching Gemini → Gemini calls reinitialize on existing manager."""
        controller = self._make_controller()
        workflow = _FakeADKWorkflow("gemini-2.5-pro")
        controller._workflow = workflow

        await controller.reinitialize(model="gemini-2.5-flash")

        workflow.reinitialize.assert_awaited_once_with(
            model="gemini-2.5-flash", preserve_sessions=True
        )

    async def test_reinitialize_no_model_calls_existing_reinitialize(self):
        """No model specified → delegates to existing manager's reinitialize."""
        controller = self._make_controller()
        workflow = _FakeADKWorkflow()
        controller._workflow = workflow

        await controller.reinitialize()

        workflow.reinitialize.assert_awaited_once_with(
            model=None, preserve_sessions=True
        )

    async def test_reinitialize_raises_when_not_initialized(self):
        """Reinitialize raises RuntimeError if workflow not initialized."""
        controller = self._make_controller()
        with pytest.raises(RuntimeError, match="Cannot reinitialize"):
            await controller.reinitialize(model="claude-sonnet-4-5")

    async def test_swap_claude_to_gemini(self):
        """Switching from Claude (LangGraph auto) back to Gemini (ADK) triggers swap."""
        controller = self._make_controller()
        old_workflow = _FakeLangGraphWorkflow("claude-sonnet-4-5")
        controller._workflow = old_workflow

        with patch(
            "agentic_cli.workflow.adk.manager.GoogleADKWorkflowManager"
        ) as mock_adk_cls:
            new_workflow = AsyncMock()
            mock_adk_cls.return_value = new_workflow
            await controller.reinitialize(model="gemini-2.5-pro")

        old_workflow.reinitialize.assert_not_called()
        new_workflow.initialize_services.assert_awaited_once()
        assert controller._workflow is new_workflow
