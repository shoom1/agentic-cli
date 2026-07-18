"""Tests for workflow controller factory and orchestrator routing.

The backend is chosen purely by ``settings.orchestrator`` and is model-agnostic:
ADK runs Claude natively via the direct-API ``AnthropicLlm`` (no LiteLLM), so
Claude is no longer auto-routed to LangGraph. A model switch alone never forces an
orchestrator swap; only an orchestrator-setting change (leaving a stale manager)
does.
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
    settings.session_store = "memory"
    for k, v in extra.items():
        setattr(settings, k, v)
    return settings


def _FakeADKWorkflow(model="gemini-2.5-pro"):
    """Create a fake ADK workflow manager for testing."""
    from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

    wf = MagicMock(spec=GoogleADKWorkflowManager)
    wf.model = model
    wf.backend_type = "adk"
    wf.reinitialize = AsyncMock()
    wf.initialize_services = AsyncMock()
    return wf


def _FakeLangGraphWorkflow(model="claude-sonnet-4-5"):
    """Create a fake LangGraph workflow manager for testing."""
    from agentic_cli.workflow.langgraph.manager import LangGraphWorkflowManager

    wf = MagicMock(spec=LangGraphWorkflowManager)
    wf.model = model
    wf.backend_type = "langgraph"
    wf.reinitialize = AsyncMock()
    wf.initialize_services = AsyncMock()
    return wf


# --- Unit tests for helpers (predicates retained for callers/back-compat) ---


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


# --- Factory routing tests (backend = orchestrator setting only) ---


class TestCreateWorkflowManagerRouting:
    """The factory routes purely on ``settings.orchestrator`` (model-agnostic)."""

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

    def test_claude_model_with_adk_returns_adk(self, agent_configs):
        """Claude model + ADK orchestrator → ADK manager (native AnthropicLlm)."""
        settings = _make_settings(orchestrator=OrchestratorType.ADK)
        with patch(
            "agentic_cli.workflow.adk.manager.GoogleADKWorkflowManager"
        ) as mock_adk_cls:
            result = create_workflow_manager_from_settings(
                agent_configs, settings, model="claude-sonnet-4-5"
            )
        mock_adk_cls.assert_called_once()
        assert result is mock_adk_cls.return_value

    def test_claude_model_in_settings_returns_adk(self, agent_configs):
        """Claude model in settings.default_model + ADK orchestrator → ADK manager."""
        settings = _make_settings(
            orchestrator=OrchestratorType.ADK,
            default_model="claude-opus-4",
        )
        with patch(
            "agentic_cli.workflow.adk.manager.GoogleADKWorkflowManager"
        ) as mock_adk_cls:
            result = create_workflow_manager_from_settings(agent_configs, settings)
        mock_adk_cls.assert_called_once()
        assert result is mock_adk_cls.return_value

    @patch("agentic_cli.workflow.langgraph.LangGraphWorkflowManager")
    def test_claude_model_with_langgraph_returns_langgraph(
        self, mock_lg_cls, agent_configs
    ):
        """Claude still runs on LangGraph when that orchestrator is chosen."""
        settings = _make_settings(orchestrator=OrchestratorType.LANGGRAPH)
        result = create_workflow_manager_from_settings(
            agent_configs, settings, model="claude-sonnet-4-5"
        )
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
    """A swap happens only when the manager type no longer matches the setting."""

    def _make_controller(self, orchestrator=OrchestratorType.ADK):
        configs = [AgentConfig(name="test", prompt="Test")]
        settings = _make_settings(orchestrator=orchestrator)
        return WorkflowController(configs, settings)

    def test_no_swap_gemini_to_claude_on_adk(self):
        """ADK manager + Claude model → no swap (Claude runs on ADK natively)."""
        controller = self._make_controller()
        controller._workflow = _FakeADKWorkflow()

        assert controller._needs_orchestrator_swap("claude-sonnet-4-5") is False

    def test_no_swap_gemini_to_gemini(self):
        """ADK manager + Gemini model → no swap needed."""
        controller = self._make_controller()
        controller._workflow = _FakeADKWorkflow()

        assert controller._needs_orchestrator_swap("gemini-2.5-pro") is False

    def test_swap_stale_langgraph_manager_on_adk(self):
        """ADK orchestrator + a stale LangGraph manager → swap back to ADK."""
        controller = self._make_controller()
        controller._workflow = _FakeLangGraphWorkflow("claude-sonnet-4-5")

        assert controller._needs_orchestrator_swap("claude-opus-4") is True

    def test_no_swap_when_langgraph_orchestrator(self):
        """LangGraph orchestrator + LangGraph manager → no swap (user chose it)."""
        controller = self._make_controller(orchestrator=OrchestratorType.LANGGRAPH)
        controller._workflow = _FakeLangGraphWorkflow("gemini-2.5-pro")

        assert controller._needs_orchestrator_swap("gemini-2.5-flash") is False

    def test_swap_stale_adk_manager_on_langgraph(self):
        """LangGraph orchestrator + a stale ADK manager → swap to LangGraph."""
        controller = self._make_controller(orchestrator=OrchestratorType.LANGGRAPH)
        controller._workflow = _FakeADKWorkflow("gemini-2.5-pro")

        assert controller._needs_orchestrator_swap("gemini-2.5-flash") is True

    def test_no_swap_when_model_is_none(self):
        """No model specified → no swap."""
        controller = self._make_controller()
        controller._workflow = _FakeADKWorkflow()

        assert controller._needs_orchestrator_swap(None) is False

    def test_no_swap_when_workflow_is_none(self):
        """No workflow initialized → no swap."""
        controller = self._make_controller()
        assert controller._needs_orchestrator_swap("claude-sonnet-4-5") is False

    def test_swap_when_model_none_but_backend_stale(self):
        """A-3: changing the orchestrator alone (model unchanged) must still
        swap a stale manager — model=None must not short-circuit the check."""
        controller = self._make_controller(orchestrator=OrchestratorType.ADK)
        controller._workflow = _FakeLangGraphWorkflow("gemini-2.5-pro")
        assert controller._needs_orchestrator_swap(None) is True

    def test_swap_check_does_not_import_langgraph(self, monkeypatch):
        """A-2: the swap check must route on backend_type, not by importing the
        LangGraph manager — else changing the model on an ADK-only install
        (no `langgraph` extra) raises ImportError."""
        import sys

        monkeypatch.setitem(
            sys.modules, "agentic_cli.workflow.langgraph.manager", None
        )
        controller = self._make_controller(orchestrator=OrchestratorType.ADK)
        controller._workflow = _FakeADKWorkflow()
        # Must not raise ImportError:
        assert controller._needs_orchestrator_swap("claude-sonnet-4-5") is False

    async def test_reinitialize_claude_on_adk_reinits_in_place(self):
        """ADK manager + Claude model → no swap; reinitialize in place."""
        controller = self._make_controller()
        workflow = _FakeADKWorkflow("gemini-2.5-pro")
        controller._workflow = workflow

        await controller.reinitialize(model="claude-sonnet-4-5")

        workflow.reinitialize.assert_awaited_once_with(
            model="claude-sonnet-4-5", preserve_sessions=True
        )
        assert controller._workflow is workflow

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

    async def test_reinitialize_migrates_stale_langgraph_to_adk(self):
        """A stale LangGraph manager under ADK orchestrator is replaced on reinit."""
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


# --- Lifecycle: readiness, atomic swap, close() ---


def _make_lifecycle_controller(orchestrator=OrchestratorType.ADK):
    configs = [AgentConfig(name="test", prompt="Test")]
    return WorkflowController(configs, _make_settings(orchestrator=orchestrator))


def _blocked_init_workflow():
    """Fake manager whose initialize_services blocks until released."""
    import asyncio

    wf = _FakeADKWorkflow()
    started, release = asyncio.Event(), asyncio.Event()

    async def _slow_init():
        started.set()
        await release.wait()

    wf.initialize_services = _slow_init
    return wf, started, release


class TestControllerLifecycle:
    """is_ready / ensure_initialized must reflect completed service init."""

    async def test_not_ready_until_services_initialized(self):
        import asyncio

        controller = _make_lifecycle_controller()
        wf, started, release = _blocked_init_workflow()
        controller._create_fn = lambda: wf

        await controller.start_background_init()
        await asyncio.wait_for(started.wait(), timeout=5)

        assert controller.is_ready is False
        with pytest.raises(RuntimeError):
            controller.workflow

        release.set()
        await controller._init_task
        assert controller.is_ready is True
        assert controller.workflow is wf

    async def test_failed_service_init_is_not_ready_and_cleans_up(self):
        controller = _make_lifecycle_controller()
        wf = _FakeADKWorkflow()
        wf.initialize_services = AsyncMock(side_effect=RuntimeError("boom"))
        controller._create_fn = lambda: wf

        await controller.start_background_init()
        await controller._init_task

        assert controller.is_ready is False
        assert isinstance(controller.init_error, RuntimeError)
        assert await controller.ensure_initialized() is False
        wf.cleanup.assert_awaited_once()

    async def test_ensure_initialized_waits_for_inflight_services(self):
        import asyncio

        controller = _make_lifecycle_controller()
        wf, started, release = _blocked_init_workflow()
        controller._create_fn = lambda: wf

        await controller.start_background_init()
        await asyncio.wait_for(started.wait(), timeout=5)

        ensure_task = asyncio.create_task(controller.ensure_initialized())
        await asyncio.sleep(0.05)
        assert not ensure_task.done()

        release.set()
        assert await ensure_task is True


class TestOrchestratorSwapLifecycle:
    """Swap must initialize the replacement first, then replace atomically."""

    def _controller_needing_swap(self):
        # Live ADK manager while settings demand LangGraph → swap required
        controller = _make_lifecycle_controller(
            orchestrator=OrchestratorType.LANGGRAPH
        )
        old = _FakeADKWorkflow()
        controller._workflow = old
        return controller, old

    async def test_swap_failure_keeps_old_manager(self):
        controller, old = self._controller_needing_swap()
        new = _FakeLangGraphWorkflow()
        new.initialize_services = AsyncMock(side_effect=RuntimeError("init failed"))

        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=new,
        ):
            with pytest.raises(RuntimeError, match="init failed"):
                await controller.reinitialize()

        assert controller._workflow is old
        old.cleanup.assert_not_awaited()
        new.cleanup.assert_awaited_once()

    async def test_swap_success_replaces_then_cleans_old(self):
        controller, old = self._controller_needing_swap()
        new = _FakeLangGraphWorkflow()

        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=new,
        ):
            await controller.reinitialize()

        assert controller._workflow is new
        old.cleanup.assert_awaited_once()


class TestControllerClose:
    """close() releases the manager and is safe to call repeatedly."""

    async def test_close_cleans_manager_and_is_idempotent(self):
        controller = _make_lifecycle_controller()
        wf = _FakeADKWorkflow()
        controller._workflow = wf

        await controller.close()
        wf.cleanup.assert_awaited_once()
        assert controller.is_ready is False

        await controller.close()
        wf.cleanup.assert_awaited_once()  # still once — idempotent

    async def test_background_init_cm_closes_manager_on_exit(self):
        controller = _make_lifecycle_controller()
        wf = _FakeADKWorkflow()
        controller._create_fn = lambda: wf
        ui = MagicMock()

        async with controller.background_init(ui):
            assert await controller.ensure_initialized() is True

        wf.cleanup.assert_awaited_once()
