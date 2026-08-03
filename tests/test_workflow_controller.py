"""Tests for workflow controller factory and orchestrator routing.

The backend is chosen purely by ``settings.orchestrator`` and is model-agnostic:
ADK runs Claude natively via the direct-API ``AnthropicLlm`` (no LiteLLM), so
Claude is no longer auto-routed to LangGraph. A model switch alone never forces an
orchestrator swap; only an orchestrator-setting change (leaving a stale manager)
does.
"""

from types import SimpleNamespace
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
        wf.cleanup.assert_awaited_once()

    async def test_ensure_initialized_retries_a_failed_attempt(self):
        """A FAILED controller retries, so a corrected setting can recover it."""
        controller = _make_lifecycle_controller()
        failing = _FakeADKWorkflow()
        failing.initialize_services = AsyncMock(side_effect=RuntimeError("boom"))
        good = _FakeADKWorkflow()
        managers = [failing, good]
        controller._create_fn = lambda: managers.pop(0)

        await controller.start_background_init()
        await controller._init_task
        assert controller.is_ready is False

        assert await controller.ensure_initialized() is True
        assert controller.workflow is good
        assert controller.init_error is None

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


class TestControllerLifecycleSerialization:
    """init / reinitialize / swap / close must not interleave.

    They all read-modify-write ``_workflow`` across awaits, so two of them in
    flight could publish two managers (leaking one), or publish one *after*
    ``close()`` had already run — leaving a live backend behind at shutdown.
    """

    async def test_nothing_is_published_after_close(self):
        import asyncio

        controller = _make_lifecycle_controller()
        wf, started, release = _blocked_init_workflow()
        wf.is_initialized = True
        controller._create_fn = lambda: wf

        await controller.start_background_init()
        await asyncio.wait_for(started.wait(), timeout=5)

        closing = asyncio.create_task(controller.close())
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.wait_for(closing, timeout=5)

        assert controller._workflow is None, "a manager was published after close()"
        wf.cleanup.assert_awaited()
        assert controller.state.value == "closed"

    async def test_close_waits_for_an_in_flight_reinitialize(self):
        import asyncio

        controller = _make_lifecycle_controller()
        old = _FakeADKWorkflow()
        old.is_initialized = True
        controller._workflow = old

        entered, release = asyncio.Event(), asyncio.Event()

        async def _slow_reinit(model=None, preserve_sessions=True):
            entered.set()
            await release.wait()

        old.reinitialize = _slow_reinit

        reinit = asyncio.create_task(controller.reinitialize())
        await asyncio.wait_for(entered.wait(), timeout=5)

        closing = asyncio.create_task(controller.close())
        await asyncio.sleep(0.05)
        assert not closing.done(), "close() ran through a live reinitialization"

        release.set()
        await asyncio.wait_for(reinit, timeout=5)
        await asyncio.wait_for(closing, timeout=5)

        assert controller._workflow is None
        old.cleanup.assert_awaited()

    async def test_concurrent_swaps_clean_every_losing_candidate(self):
        import asyncio

        controller = _make_lifecycle_controller(
            orchestrator=OrchestratorType.LANGGRAPH
        )
        old = _FakeADKWorkflow()
        old.is_initialized = True
        controller._workflow = old

        built = []

        def _build(**kwargs):
            new = _FakeLangGraphWorkflow()
            new.is_initialized = True
            built.append(new)
            return new

        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            side_effect=_build,
        ):
            await asyncio.gather(
                controller.reinitialize(), controller.reinitialize()
            )

        assert controller._workflow in built
        survivors = [m for m in built if m is controller._workflow]
        losers = [m for m in built if m is not controller._workflow]
        assert len(survivors) == 1
        for loser in losers:
            loser.cleanup.assert_awaited(), "a losing swap candidate leaked"
        old.cleanup.assert_awaited()

    async def test_init_started_during_reinitialize_does_not_race(self):
        import asyncio

        controller = _make_lifecycle_controller()
        old = _FakeADKWorkflow()
        old.is_initialized = True
        controller._workflow = old

        entered, release = asyncio.Event(), asyncio.Event()

        async def _slow_reinit(model=None, preserve_sessions=True):
            entered.set()
            await release.wait()

        old.reinitialize = _slow_reinit

        reinit = asyncio.create_task(controller.reinitialize())
        await asyncio.wait_for(entered.wait(), timeout=5)

        starting = asyncio.create_task(controller.start_background_init())
        await asyncio.sleep(0.05)
        assert not starting.done(), "an init was scheduled mid-reinitialization"

        release.set()
        await asyncio.wait_for(reinit, timeout=5)
        await asyncio.wait_for(starting, timeout=5)

        assert controller._workflow is old  # already READY → no new manager
        await controller.close()


class TestConstructionInTheExecutorIsTracked:
    """A manager built by the init thread must never be silently dropped.

    ``run_in_executor``'s future was awaited unshielded, so cancelling the init
    task cancelled the *asyncio* future while the worker thread carried on.
    When the thread returned, asyncio discarded the result — a fully
    constructed manager (with whatever it had already opened) that nothing
    could reach, let alone clean up.
    """

    def _blocking_create(self):
        """A ``_create_fn`` that blocks in the worker thread until released."""
        import threading

        entered = threading.Event()
        release = threading.Event()
        made: list = []

        def _create():
            entered.set()
            release.wait(timeout=5)
            wf = _FakeADKWorkflow()
            wf.is_initialized = True
            made.append(wf)
            return wf

        return _create, entered, release, made

    async def test_cancelled_construction_is_cleaned_exactly_once(self):
        import asyncio

        controller = _make_lifecycle_controller()
        create, entered, release, made = self._blocking_create()
        controller._create_fn = create

        await controller.start_background_init()
        await asyncio.to_thread(entered.wait, 5)

        controller._init_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await controller._init_task

        release.set()
        await controller.close()

        assert made, "the worker thread never produced a manager"
        assert controller._workflow is None, "a cancelled build was published"
        made[0].cleanup.assert_awaited_once()

    async def test_close_during_construction_cleans_the_manager(self):
        import asyncio

        controller = _make_lifecycle_controller()
        create, entered, release, made = self._blocking_create()
        controller._create_fn = create

        await controller.start_background_init()
        await asyncio.to_thread(entered.wait, 5)

        closing = asyncio.create_task(controller.close())
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.wait_for(closing, timeout=5)

        assert made
        assert controller._workflow is None
        made[0].cleanup.assert_awaited_once()

    async def test_manager_built_after_close_is_never_published(self):
        import asyncio

        from agentic_cli.cli.workflow_controller import WorkflowState

        controller = _make_lifecycle_controller()
        create, entered, release, made = self._blocking_create()
        controller._create_fn = create

        await controller.start_background_init()
        await asyncio.to_thread(entered.wait, 5)

        closing = asyncio.create_task(controller.close())
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.wait_for(closing, timeout=5)

        assert controller.state is WorkflowState.CLOSED
        with pytest.raises(RuntimeError):
            controller.workflow


class TestCloseIsCancellationSafe:
    """Shutdown must finish even if whoever asked for it goes away.

    ``close()`` did its teardown inline, so a cancelled caller abandoned it
    mid-way: a manager still being built in the executor was never released, a
    manager already published was left half-cleaned, and a later ``close()``
    returned immediately because ``_closed`` was already True — reporting a
    shutdown that never happened.
    """

    async def test_cancelled_close_still_releases_a_pending_construction(self):
        import asyncio
        import threading

        controller = _make_lifecycle_controller()
        entered, release = threading.Event(), threading.Event()
        made: list = []

        def _create():
            entered.set()
            release.wait(timeout=5)
            wf = _FakeADKWorkflow()
            wf.is_initialized = True
            made.append(wf)
            return wf

        controller._create_fn = _create
        await controller.start_background_init()
        await asyncio.to_thread(entered.wait, 5)

        closing = asyncio.create_task(controller.close())
        await asyncio.sleep(0.05)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing

        release.set()
        await controller.close()  # joins the teardown the cancelled caller started

        assert made, "the worker thread never produced a manager"
        assert controller._workflow is None
        made[0].cleanup.assert_awaited_once()

    async def test_cancelled_close_still_cleans_a_published_manager(self):
        import asyncio

        controller = _make_lifecycle_controller()
        wf = _FakeADKWorkflow()
        wf.is_initialized = True
        controller._workflow = wf

        entered, release = asyncio.Event(), asyncio.Event()
        finished: list[str] = []

        async def _slow_cleanup():
            entered.set()
            await release.wait()
            finished.append("cleanup")

        wf.cleanup = AsyncMock(side_effect=_slow_cleanup)

        closing = asyncio.create_task(controller.close())
        await asyncio.wait_for(entered.wait(), timeout=5)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing

        release.set()
        await controller.close()

        wf.cleanup.assert_awaited_once()
        assert finished == ["cleanup"], "cleanup was abandoned mid-way"
        assert controller._workflow is None

    async def test_a_later_close_joins_the_first(self):
        import asyncio

        controller = _make_lifecycle_controller()
        wf = _FakeADKWorkflow()
        wf.is_initialized = True
        controller._workflow = wf

        entered, release = asyncio.Event(), asyncio.Event()
        finished: list[str] = []

        async def _slow_cleanup():
            entered.set()
            await release.wait()
            finished.append("cleanup")

        wf.cleanup = AsyncMock(side_effect=_slow_cleanup)

        first = asyncio.create_task(controller.close())
        await asyncio.wait_for(entered.wait(), timeout=5)

        second = asyncio.create_task(controller.close())
        await asyncio.sleep(0.05)
        assert not second.done(), "a second close() reported a shutdown still running"

        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
        wf.cleanup.assert_awaited_once()
        assert finished == ["cleanup"]

    async def test_cancelling_the_join_does_not_stop_the_teardown(self):
        """The teardown is the controller's, not the caller's."""
        import asyncio

        controller = _make_lifecycle_controller()
        wf = _FakeADKWorkflow()
        wf.is_initialized = True
        controller._workflow = wf

        entered, release = asyncio.Event(), asyncio.Event()
        finished: list[str] = []

        async def _slow_cleanup():
            entered.set()
            await release.wait()
            finished.append("cleanup")

        wf.cleanup = AsyncMock(side_effect=_slow_cleanup)

        first = asyncio.create_task(controller.close())
        await asyncio.wait_for(entered.wait(), timeout=5)
        second = asyncio.create_task(controller.close())
        await asyncio.sleep(0.05)

        first.cancel()
        second.cancel()
        for task in (first, second):
            with pytest.raises(asyncio.CancelledError):
                await task

        release.set()
        await controller.close()

        wf.cleanup.assert_awaited_once()
        assert finished == ["cleanup"], "a cancelled caller abandoned the teardown"
        assert controller.state.value == "closed"


class TestCancelInitIsCancellationSafe:
    """``cancel_init()`` is public and may be awaited directly.

    Cancelling it mid-settle consumed the construction claim and then never
    released the manager: the claim is single-shot, so the fallback callback
    could not take over either, and a later ``close()`` found nothing to settle.
    """

    def _blocking_create(self):
        import threading

        entered, release = threading.Event(), threading.Event()
        made: list = []

        def _create():
            entered.set()
            release.wait(timeout=5)
            wf = _FakeADKWorkflow()
            wf.is_initialized = True
            made.append(wf)
            return wf

        return _create, entered, release, made

    async def test_cancelled_cancel_init_does_not_strand_the_manager(self):
        import asyncio

        controller = _make_lifecycle_controller()
        create, entered, release, made = self._blocking_create()
        controller._create_fn = create

        await controller.start_background_init()
        await asyncio.to_thread(entered.wait, 5)

        cancelling = asyncio.create_task(controller.cancel_init())
        await asyncio.sleep(0.05)
        cancelling.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelling

        release.set()
        await controller.close()

        assert made, "the worker thread never produced a manager"
        assert controller._workflow is None, "a stranded manager was published"
        made[0].cleanup.assert_awaited_once()

    async def test_cancelled_cancel_init_without_close_still_cleans(self):
        """The fallback callback must still own the result."""
        import asyncio

        controller = _make_lifecycle_controller()
        create, entered, release, made = self._blocking_create()
        controller._create_fn = create

        await controller.start_background_init()
        await asyncio.to_thread(entered.wait, 5)

        cancelling = asyncio.create_task(controller.cancel_init())
        await asyncio.sleep(0.05)
        cancelling.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelling

        release.set()
        for _ in range(100):
            if made and made[0].cleanup.await_count:
                break
            await asyncio.sleep(0.02)

        assert made
        made[0].cleanup.assert_awaited_once()
        assert controller._workflow is None


class TestConstructionCleanupSurvivesCancellation:
    """The *cleanup* of an abandoned construction is the controller's too.

    ``_settle_construction()`` was cancellation-safe only while awaiting the
    worker thread. Once it had the manager it awaited ``manager.cleanup()``
    inline, so a caller cancelled at that point cancelled the cleanup itself —
    with the construction claim already consumed and ``_construction`` cleared,
    neither the fallback callback nor a later ``close()`` could finish it.
    """

    def _controller_with_slow_cleanup(self):
        """A controller whose next-built manager blocks inside ``cleanup()``."""
        import asyncio
        import threading

        controller = _make_lifecycle_controller()
        made: list = []
        finished: list[str] = []
        entered, release = threading.Event(), threading.Event()
        cleanup_entered, cleanup_release = asyncio.Event(), asyncio.Event()

        async def _slow_cleanup() -> None:
            cleanup_entered.set()
            await cleanup_release.wait()
            finished.append("cleanup")

        def _create():
            entered.set()
            release.wait(timeout=5)
            wf = _FakeADKWorkflow()
            wf.is_initialized = True
            wf.cleanup = AsyncMock(side_effect=_slow_cleanup)
            made.append(wf)
            return wf

        controller._create_fn = _create
        return SimpleNamespace(
            controller=controller,
            made=made,
            finished=finished,
            entered=entered,
            release=release,
            cleanup_entered=cleanup_entered,
            cleanup_release=cleanup_release,
        )

    async def _cancel_inside_cleanup(self, h):
        """Drive ``cancel_init()`` until cleanup has entered, then cancel it."""
        import asyncio

        await h.controller.start_background_init()
        await asyncio.to_thread(h.entered.wait, 5)

        cancelling = asyncio.create_task(h.controller.cancel_init())
        await asyncio.sleep(0.05)  # reach the await on the worker thread
        h.release.set()
        await asyncio.wait_for(h.cleanup_entered.wait(), timeout=5)

        cancelling.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelling
        return cancelling

    async def test_close_joins_a_cleanup_the_cancelled_caller_left_running(self):
        import asyncio

        h = self._controller_with_slow_cleanup()
        await self._cancel_inside_cleanup(h)

        closing = asyncio.create_task(h.controller.close())
        await asyncio.sleep(0.05)
        assert not closing.done(), "close() did not join the pending cleanup"

        h.cleanup_release.set()
        await asyncio.wait_for(closing, timeout=5)

        assert h.finished == ["cleanup"], "the cancelled caller aborted the cleanup"
        assert h.made[0].cleanup.await_count == 1
        assert h.controller._workflow is None

    async def test_cleanup_completes_without_a_later_close(self):
        import asyncio

        h = self._controller_with_slow_cleanup()
        await self._cancel_inside_cleanup(h)

        h.cleanup_release.set()
        for _ in range(200):
            if h.finished:
                break
            await asyncio.sleep(0.02)

        assert h.finished == ["cleanup"], "nobody finished the abandoned cleanup"
        assert h.made[0].cleanup.await_count == 1

        await h.controller.close()
        assert h.made[0].cleanup.await_count == 1, "the manager was cleaned twice"


class TestInitErrorIsClearedOnSuccess:
    """A recorded failure must not outlive the recovery.

    ``_init_error`` was only cleared at the *start* of a background init, so
    after a failed reinitialization that then succeeded the controller was
    ``READY`` while still holding the old exception — and every consumer keyed
    off a different field: ``ensure_initialized()`` returned False (it checked
    the error), ``state``/``workflow`` said ready, and the status bar showed
    "Init failed - check API keys" over a working session.
    """

    def _recovered_controller(self):
        controller = _make_lifecycle_controller()
        wf = _FakeADKWorkflow()
        wf.is_initialized = True
        controller._workflow = wf
        controller._init_error = RuntimeError("an earlier failure")
        return controller, wf

    async def test_successful_reinitialize_clears_the_error(self):
        controller, _ = self._recovered_controller()

        await controller.reinitialize(model="gemini-2.5-flash")

        assert controller.init_error is None

    async def test_successful_swap_clears_the_error(self):
        controller = _make_lifecycle_controller(
            orchestrator=OrchestratorType.LANGGRAPH
        )
        old = _FakeADKWorkflow()
        old.is_initialized = True
        controller._workflow = old
        controller._init_error = RuntimeError("an earlier failure")

        new = _FakeLangGraphWorkflow()
        new.is_initialized = True
        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=new,
        ):
            await controller.reinitialize()

        assert controller.init_error is None

    async def test_successful_background_init_clears_the_error(self):
        controller = _make_lifecycle_controller()
        wf = _FakeADKWorkflow()
        wf.is_initialized = True
        controller._create_fn = lambda: wf
        controller._init_error = RuntimeError("an earlier failure")

        assert await controller.ensure_initialized() is True
        assert controller.init_error is None

    async def test_everything_agrees_after_recovery(self):
        from agentic_cli.cli.workflow_controller import WorkflowState

        controller, wf = self._recovered_controller()
        await controller.reinitialize()

        assert controller.state is WorkflowState.READY
        assert controller.is_ready is True
        assert controller.workflow is wf
        assert await controller.ensure_initialized() is True

        ui = MagicMock()
        controller.update_status_bar(ui)
        status = ui.set_status.call_args[0][0]
        assert "Init failed" not in status
        assert wf.model in status


class TestFailedReinitPreservesSessions:
    """A failed in-place reinitialization must not discard the conversation.

    ``GoogleADKWorkflowManager.reinitialize(preserve_sessions=True)`` restores
    the session service it was carrying when initialization fails. The
    controller then cleaned the manager up anyway, which closed that service —
    with ``session_store='memory'`` the whole conversation went with it, for a
    failure the user could fix (a bad model id) and retry.
    """

    def _controller_with_failing_reinit(self):
        controller = _make_lifecycle_controller()
        wf = _FakeADKWorkflow()
        wf.is_initialized = True
        controller._workflow = wf

        async def _fail(model=None, preserve_sessions=True):
            wf.is_initialized = False  # the manager rolled itself back
            raise RuntimeError("reinit boom")

        wf.reinitialize = _fail
        return controller, wf

    async def test_failed_reinit_keeps_the_manager_alive(self):
        from agentic_cli.cli.workflow_controller import WorkflowState

        controller, wf = self._controller_with_failing_reinit()

        with pytest.raises(RuntimeError, match="reinit boom"):
            await controller.reinitialize()

        assert controller.state is WorkflowState.FAILED
        assert controller.is_ready is False
        wf.cleanup.assert_not_awaited(), "the preserved session service was closed"

    async def test_retry_revives_the_same_manager(self):
        controller, wf = self._controller_with_failing_reinit()

        async def _init_ok():
            wf.is_initialized = True

        wf.initialize_services = _init_ok
        replacement = _FakeADKWorkflow()
        controller._create_fn = lambda: replacement

        with pytest.raises(RuntimeError):
            await controller.reinitialize()

        assert await controller.ensure_initialized() is True
        assert controller.workflow is wf, "sessions were dropped for a fresh manager"
        replacement.cleanup.assert_not_awaited()

    async def test_unrevivable_manager_is_released_and_replaced(self):
        controller, wf = self._controller_with_failing_reinit()
        wf.initialize_services = AsyncMock(side_effect=RuntimeError("still broken"))

        replacement = _FakeADKWorkflow()
        replacement.is_initialized = True
        controller._create_fn = lambda: replacement

        with pytest.raises(RuntimeError):
            await controller.reinitialize()

        assert await controller.ensure_initialized() is False
        wf.cleanup.assert_awaited_once()

        assert await controller.ensure_initialized() is True
        assert controller.workflow is replacement

    async def test_failed_manager_is_never_handed_out(self):
        controller, wf = self._controller_with_failing_reinit()

        with pytest.raises(RuntimeError, match="reinit boom"):
            await controller.reinitialize()

        with pytest.raises(RuntimeError, match="not initialized"):
            controller.workflow


class TestControllerStateMachine:
    """Explicit lifecycle states, derived from the controller's internals."""

    async def test_state_progression_uninitialized_to_ready(self):
        import asyncio

        from agentic_cli.cli.workflow_controller import WorkflowState

        controller = _make_lifecycle_controller()
        wf, started, release = _blocked_init_workflow()
        controller._create_fn = lambda: wf

        assert controller.state is WorkflowState.UNINITIALIZED

        await controller.start_background_init()
        await asyncio.wait_for(started.wait(), timeout=5)
        assert controller.state is WorkflowState.INITIALIZING

        release.set()
        await controller._init_task
        assert controller.state is WorkflowState.READY

        await controller.close()
        assert controller.state is WorkflowState.CLOSED

    async def test_state_failed_after_init_error(self):
        from agentic_cli.cli.workflow_controller import WorkflowState

        controller = _make_lifecycle_controller()
        wf = _FakeADKWorkflow()
        wf.initialize_services = AsyncMock(side_effect=RuntimeError("boom"))
        controller._create_fn = lambda: wf

        await controller.start_background_init()
        await controller._init_task

        assert controller.state is WorkflowState.FAILED

    async def test_closed_controller_is_not_ready_and_refuses_init(self):
        controller = _make_lifecycle_controller()
        await controller.close()

        assert controller.is_ready is False
        assert await controller.ensure_initialized() is False
        with pytest.raises(RuntimeError, match="closed"):
            await controller.start_background_init()


class TestControllerSingleFlightInit:
    """Two callers must never build two managers for one controller."""

    async def test_concurrent_start_background_init_creates_one_manager(self):
        import asyncio

        controller = _make_lifecycle_controller()
        wf, started, release = _blocked_init_workflow()
        created = []

        def _create():
            created.append(wf)
            return wf

        controller._create_fn = _create

        await asyncio.gather(*(controller.start_background_init() for _ in range(5)))
        await asyncio.wait_for(started.wait(), timeout=5)
        release.set()
        await controller._init_task

        assert len(created) == 1
        assert controller.workflow is wf

    async def test_concurrent_ensure_initialized_callers_all_see_ready(self):
        import asyncio

        controller = _make_lifecycle_controller()
        wf, started, release = _blocked_init_workflow()
        controller._create_fn = lambda: wf

        await controller.start_background_init()
        await asyncio.wait_for(started.wait(), timeout=5)

        waiters = [
            asyncio.create_task(controller.ensure_initialized()) for _ in range(4)
        ]
        await asyncio.sleep(0.05)
        assert not any(w.done() for w in waiters)

        release.set()
        assert await asyncio.gather(*waiters) == [True] * 4

    async def test_cancelled_caller_does_not_kill_shared_init(self):
        """One caller giving up must not cancel the shared init for everyone."""
        import asyncio

        controller = _make_lifecycle_controller()
        wf, started, release = _blocked_init_workflow()
        controller._create_fn = lambda: wf

        await controller.start_background_init()
        await asyncio.wait_for(started.wait(), timeout=5)

        giving_up = asyncio.create_task(controller.ensure_initialized())
        await asyncio.sleep(0.05)
        giving_up.cancel()
        with pytest.raises(asyncio.CancelledError):
            await giving_up

        release.set()
        assert await controller.ensure_initialized() is True
        assert controller.workflow is wf


class TestControllerInitRetry:
    """After a failure, a fresh start_background_init() retries cleanly."""

    async def test_retry_after_failure_succeeds_and_clears_error(self):
        controller = _make_lifecycle_controller()
        failing = _FakeADKWorkflow()
        failing.initialize_services = AsyncMock(side_effect=RuntimeError("boom"))
        good = _FakeADKWorkflow()
        managers = [failing, good]
        controller._create_fn = lambda: managers.pop(0)

        await controller.start_background_init()
        await controller._init_task
        assert controller.init_error is not None
        failing.cleanup.assert_awaited_once()

        await controller.start_background_init()
        await controller._init_task

        assert controller.init_error is None
        assert controller.is_ready is True
        assert controller.workflow is good

    async def test_start_is_noop_once_ready(self):
        controller = _make_lifecycle_controller()
        wf = _FakeADKWorkflow()
        created = []

        def _create():
            created.append(wf)
            return wf

        controller._create_fn = _create

        await controller.start_background_init()
        await controller._init_task
        await controller.start_background_init()

        assert len(created) == 1


class TestReinitializeTransaction:
    """A failed in-place reinitialization must not leave the controller READY."""

    def _ready_controller(self):
        from agentic_cli.cli.workflow_controller import WorkflowState

        controller = _make_lifecycle_controller()
        wf = _FakeADKWorkflow()
        wf.is_initialized = True
        controller._workflow = wf
        assert controller.state is WorkflowState.READY
        return controller, wf

    async def test_failed_inplace_reinit_enters_failed_and_withholds(self):
        """FAILED, and the manager is not handed out — but it is kept.

        See ``TestFailedReinitPreservesSessions``: releasing it here closed
        the session service the manager had just preserved.
        """
        from agentic_cli.cli.workflow_controller import WorkflowState

        controller, wf = self._ready_controller()

        async def _fail(model=None, preserve_sessions=True):
            wf.is_initialized = False  # the manager rolled itself back
            raise RuntimeError("reinit boom")

        wf.reinitialize = _fail

        with pytest.raises(RuntimeError, match="reinit boom"):
            await controller.reinitialize(model="gemini-2.5-flash")

        assert controller.state is WorkflowState.FAILED
        assert controller.is_ready is False
        assert isinstance(controller.init_error, RuntimeError)
        with pytest.raises(RuntimeError):
            controller.workflow
        wf.cleanup.assert_not_awaited()

    async def test_uninitialized_manager_is_never_reported_ready(self):
        """State is derived from the manager, not from 'we published one'."""
        from agentic_cli.cli.workflow_controller import WorkflowState

        controller, wf = self._ready_controller()
        wf.is_initialized = False

        assert controller.state is WorkflowState.FAILED
        assert controller.is_ready is False

    async def test_successful_reinit_stays_ready(self):
        from agentic_cli.cli.workflow_controller import WorkflowState

        controller, wf = self._ready_controller()
        await controller.reinitialize(model="gemini-2.5-flash")

        assert controller.state is WorkflowState.READY
        assert controller.workflow is wf

    async def test_recovery_after_failed_reinit_restores_readiness(self):
        controller, wf = self._ready_controller()

        async def _fail(model=None, preserve_sessions=True):
            wf.is_initialized = False
            raise RuntimeError("reinit boom")

        async def _init_ok():
            wf.is_initialized = True

        wf.reinitialize = _fail
        wf.initialize_services = _init_ok
        replacement = _FakeADKWorkflow()
        replacement.is_initialized = True
        controller._create_fn = lambda: replacement

        from agentic_cli.cli.workflow_controller import WorkflowState

        with pytest.raises(RuntimeError):
            await controller.reinitialize()

        assert await controller.ensure_initialized() is True
        assert controller.state is WorkflowState.READY
        # Recovery revives the failed manager (see
        # TestFailedReinitPreservesSessions), so no replacement is built.
        replacement.initialize_services.assert_not_awaited()
