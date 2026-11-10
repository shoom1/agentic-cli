"""Tests for workflow controller factory and orchestrator routing.

The factory builds the backend ``settings.orchestrator`` names, or the one it is
given, and is model-agnostic: ADK runs Claude natively via the direct-API
``AnthropicLlm`` (no LiteLLM), so Claude is no longer auto-routed to LangGraph.
The controller runs the conversation's backend: a model switch never changes it,
and only ``switch_backend`` (a new conversation on another backend) does.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_cli.cli.workflow_controller import WorkflowController
from agentic_cli.workflow.factory import create_workflow_manager_from_settings
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

    @patch("agentic_cli.workflow.langgraph.LangGraphWorkflowManager")
    def test_the_orchestrator_given_is_built_whatever_the_settings_say(
        self, mock_lg_cls, agent_configs
    ):
        """A new conversation on another backend (``/new langgraph``) while
        the settings name ADK for new conversations."""
        settings = _make_settings(orchestrator=OrchestratorType.ADK)
        result = create_workflow_manager_from_settings(
            agent_configs, settings, orchestrator="langgraph"
        )
        mock_lg_cls.assert_called_once()
        assert result is mock_lg_cls.return_value

    def test_check_backend_names_an_unknown_one_and_a_missing_extra(self, monkeypatch):
        import sys

        from agentic_cli.workflow.factory import check_backend

        assert check_backend("deepagents") is OrchestratorType.DEEPAGENTS
        with pytest.raises(ValueError, match="nope"):
            check_backend("nope")
        monkeypatch.setitem(sys.modules, "agentic_cli.workflow.deepagents", None)
        with pytest.raises(ImportError, match=r"pip install 'agentic-cli\[deepagents\]'"):
            check_backend("deepagents")

    def test_no_model_with_adk_returns_adk(self, agent_configs):
        """No model specified + ADK orchestrator → ADK manager."""
        settings = _make_settings(orchestrator=OrchestratorType.ADK, default_model=None)
        with patch(
            "agentic_cli.workflow.adk.manager.GoogleADKWorkflowManager"
        ) as mock_adk_cls:
            result = create_workflow_manager_from_settings(agent_configs, settings)
        mock_adk_cls.assert_called_once()
        assert result is mock_adk_cls.return_value


# --- WorkflowController: the backend it runs ---


class TestTheBackend:
    """The controller runs the backend of the conversation, not the setting.

    ``settings.orchestrator`` is the backend for new conversations: the
    controller takes it at its first start, and changes backend only through
    ``switch_backend`` (a new conversation on another backend). A model switch
    (``reinitialize``) never changes it, whatever the settings say, since a
    conversation's history lives in its backend's store alone.
    """

    def _make_controller(self, orchestrator=OrchestratorType.ADK):
        configs = [AgentConfig(name="test", prompt="Test")]
        settings = _make_settings(orchestrator=orchestrator)
        return WorkflowController(configs, settings)

    def test_it_is_the_settings_until_one_is_chosen(self):
        assert self._make_controller(OrchestratorType.LANGGRAPH).backend == "langgraph"

    @pytest.mark.parametrize("orchestrator, live", [
        (OrchestratorType.ADK, _FakeLangGraphWorkflow),
        (OrchestratorType.LANGGRAPH, _FakeADKWorkflow),
    ], ids=["langgraph-runs-settings-say-adk", "adk-runs-settings-say-langgraph"])
    async def test_a_model_switch_never_changes_it(self, orchestrator, live):
        controller = self._make_controller(orchestrator)
        workflow = live("gemini-2.5-pro")
        controller._workflow = workflow

        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings"
        ) as create:
            await controller.reinitialize(model="claude-sonnet-4-5")

        create.assert_not_called()
        workflow.reinitialize.assert_awaited_once_with(
            model="claude-sonnet-4-5", preserve_sessions=True
        )
        assert controller._workflow is workflow

    async def test_a_model_switch_imports_no_other_backend(self, monkeypatch):
        """Changing the model on an ADK-only install (no ``langgraph`` extra)
        must not import the LangGraph manager."""
        import sys

        monkeypatch.setitem(sys.modules, "agentic_cli.workflow.langgraph.manager", None)
        controller = self._make_controller(OrchestratorType.ADK)
        controller._workflow = _FakeADKWorkflow()

        await controller.reinitialize(model="claude-sonnet-4-5")  # no ImportError

    async def test_switch_backend_starts_the_backend_named(self):
        controller = self._make_controller(OrchestratorType.ADK)
        old = _FakeADKWorkflow()
        controller._workflow = old
        new = _FakeLangGraphWorkflow()

        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=new,
        ) as create:
            await controller.switch_backend("langgraph")

        assert create.call_args.kwargs["orchestrator"] == "langgraph"
        assert controller._workflow is new and controller.backend == "langgraph"
        old.cleanup.assert_awaited_once()

    async def test_switching_to_the_backend_it_runs_does_nothing(self):
        controller = self._make_controller(OrchestratorType.ADK)
        workflow = _FakeADKWorkflow()
        controller._workflow = workflow

        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings"
        ) as create:
            await controller.switch_backend("adk")

        create.assert_not_called()
        assert controller._workflow is workflow

    async def test_while_none_runs_the_next_start_builds_it(self):
        controller = self._make_controller(OrchestratorType.ADK)
        built = _FakeLangGraphWorkflow()

        await controller.switch_backend("langgraph")
        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=built,
        ) as create:
            assert await controller.ensure_initialized() is False  # not started yet
            await controller.start_background_init()
            assert await controller.ensure_initialized() is True

        assert create.call_args.kwargs["orchestrator"] == "langgraph"
        assert controller._workflow is built and controller.backend == "langgraph"
        await controller.close()

    async def test_a_manager_kept_to_revive_is_released_for_another_backend(self):
        """A failed reinitialization keeps its manager for the next start to
        revive; a new conversation on another backend releases it instead,
        so that start builds the new backend rather than reviving the old."""
        controller = self._make_controller(OrchestratorType.ADK)
        kept = _FakeADKWorkflow()
        kept.is_initialized = False  # a failed reinitialization left it
        controller._workflow = kept
        controller._init_error = RuntimeError("reinitialize failed")
        built = _FakeLangGraphWorkflow()

        await controller.switch_backend("langgraph")
        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=built,
        ):
            assert await controller.ensure_initialized() is True
            await controller._join_releases()

        kept.cleanup.assert_awaited_once()
        kept.initialize_services.assert_not_awaited()
        assert controller._workflow is built and controller.backend == "langgraph"
        await controller.close()

    async def test_a_retry_after_a_failed_start_keeps_the_backend_chosen(self):
        """The settings may say otherwise by then: the backend is the
        conversation's."""
        controller = self._make_controller(OrchestratorType.LANGGRAPH)
        builds: list = []

        def create(**kwargs):
            builds.append(kwargs["orchestrator"])
            manager = _FakeLangGraphWorkflow()
            if len(builds) == 1:
                manager.initialize_services = AsyncMock(side_effect=RuntimeError("no key"))
            return manager

        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            side_effect=create,
        ):
            await controller.start_background_init()
            assert await controller.ensure_initialized() is False
            controller._settings.orchestrator = OrchestratorType.ADK  # /settings, used conversation
            assert await controller.ensure_initialized() is True

        assert builds == ["langgraph", "langgraph"]
        await controller.close()

    async def test_every_manager_of_the_run_is_built_on_its_store(self):
        """The session store is fixed for the run: a retry after a failed
        start, and a new conversation on another backend, are built on the
        store the run started with, whatever ``/settings`` changed since."""
        controller = self._make_controller(OrchestratorType.LANGGRAPH)  # memory
        builds: list = []

        def create(**kwargs):
            builds.append(kwargs["session_store"])
            manager = _FakeLangGraphWorkflow()
            if len(builds) == 1:
                manager.initialize_services = AsyncMock(side_effect=RuntimeError("no key"))
            manager.is_initialized = True
            return manager

        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            side_effect=create,
        ):
            await controller.start_background_init()
            assert await controller.ensure_initialized() is False
            controller._settings.session_store = "sqlite"  # /settings: the next app start
            assert await controller.ensure_initialized() is True
            await controller.switch_backend("adk")

        assert builds == ["memory", "memory", "memory"]
        assert controller.store == "memory"
        await controller.close()

    async def test_an_unknown_backend_is_refused(self):
        controller = self._make_controller()

        with pytest.raises(ValueError):
            await controller.switch_backend("nope")

        assert controller.backend == "adk"

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
    """A swap (``switch_backend``) initializes the replacement first, then
    replaces atomically."""

    def _controller_needing_swap(self):
        # A live ADK manager, and a new conversation on LangGraph to start.
        controller = _make_lifecycle_controller(orchestrator=OrchestratorType.ADK)
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
                await controller.switch_backend("langgraph")

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
            await controller.switch_backend("langgraph")

        assert controller._workflow is new
        old.cleanup.assert_awaited_once()

    async def test_a_swap_cancelled_while_releasing_the_old_manager_still_releases_it(self):
        """Ctrl+C once the replacement is published stops only the wait. The
        old manager's cleanup is the controller's, as an abandoned
        construction's is: it runs to its end, and ``close()`` joins it.
        Cancelled with it, the old manager's MCP servers, kernels and store
        stayed open for the rest of the process."""
        import asyncio

        controller, old = self._controller_needing_swap()
        new = _FakeLangGraphWorkflow()
        releasing, released = asyncio.Event(), asyncio.Event()
        finished: list[str] = []

        async def _slow_cleanup() -> None:
            releasing.set()
            await released.wait()
            finished.append("cleanup")

        old.cleanup = AsyncMock(side_effect=_slow_cleanup)
        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=new,
        ):
            swapping = asyncio.create_task(controller.switch_backend("langgraph"))
            await asyncio.wait_for(releasing.wait(), timeout=5)
            swapping.cancel()
            with pytest.raises(asyncio.CancelledError):
                await swapping

        assert controller._workflow is new
        closing = asyncio.create_task(controller.close())
        await asyncio.sleep(0.05)
        assert not closing.done(), "close() did not join the old manager's cleanup"
        released.set()
        await asyncio.wait_for(closing, timeout=5)
        assert finished == ["cleanup"], "the cancel aborted the old manager's cleanup"

    async def test_the_next_reinitialize_waits_for_that_release(self):
        """One lifecycle transition at a time: a switch back would otherwise
        start a manager beside the old one still being released."""
        import asyncio

        controller, old = self._controller_needing_swap()
        new = _FakeLangGraphWorkflow()
        releasing, released = asyncio.Event(), asyncio.Event()

        async def _slow_cleanup() -> None:
            releasing.set()
            await released.wait()

        old.cleanup = AsyncMock(side_effect=_slow_cleanup)
        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=new,
        ):
            swapping = asyncio.create_task(controller.switch_backend("langgraph"))
            await asyncio.wait_for(releasing.wait(), timeout=5)
            swapping.cancel()
            with pytest.raises(asyncio.CancelledError):
                await swapping

            new.reinitialize = AsyncMock()
            again = asyncio.create_task(controller.reinitialize(model="gemini-2.5-flash"))
            await asyncio.sleep(0.05)
            assert not again.done(), "it began while the old manager was released"
            new.reinitialize.assert_not_awaited()
            released.set()
            await asyncio.wait_for(again, timeout=5)

        new.reinitialize.assert_awaited_once()
        await controller.close()


class TestTheRunsServicesMoveWithTheBackend:
    """A switch of backend hands the run's services to the new manager.

    The session grants, the jobs, the sandbox kernels, the knowledge bases
    and the embedding model belong to the run, not to a backend: the new
    manager is started with the running one's services lent
    (``initialize_services(services=loan)``), the running one gives them
    over (``give_services``), the new one is published at once, with no
    await between the give and the publish, and the old one is released.
    A new manager that fails to start, or a give that fails, leaves the
    running one with its services, and nothing changes.
    (``tests/workflow/test_service_hand_over.py`` runs the hand-over on the
    real managers.)
    """

    _controller_needing_swap = TestOrchestratorSwapLifecycle._controller_needing_swap

    async def test_the_new_manager_starts_on_the_services_lent_and_is_given_them(self):
        import asyncio

        controller, old = self._controller_needing_swap()
        new = _FakeLangGraphWorkflow()
        steps: list[str] = []
        loan = old.lend_services.return_value

        async def start(**kwargs):
            steps.append("start")

        async def give(lent):
            assert lent is loan
            steps.append("give")
            # What the loop runs next, after the give: the new manager is
            # published by then (no await between the two).
            asyncio.get_running_loop().call_soon(
                lambda: steps.append(f"next: {controller._workflow.backend_type}")
            )

        async def released():
            steps.append("old released")

        new.initialize_services = AsyncMock(side_effect=start)
        old.give_services = AsyncMock(side_effect=give)
        old.cleanup = AsyncMock(side_effect=released)
        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=new,
        ):
            await controller.switch_backend("langgraph")

        new.initialize_services.assert_awaited_once_with(services=loan)
        assert steps == ["start", "give", "next: langgraph", "old released"]
        assert controller._workflow is new and controller.backend == "langgraph"

    async def test_a_failed_start_gives_nothing_and_keeps_the_running_one(self):
        controller, old = self._controller_needing_swap()
        new = _FakeLangGraphWorkflow()
        new.initialize_services = AsyncMock(side_effect=RuntimeError("no key"))

        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=new,
        ):
            with pytest.raises(RuntimeError, match="no key"):
                await controller.switch_backend("langgraph")

        old.give_services.assert_not_awaited()
        old.cleanup.assert_not_awaited()
        new.cleanup.assert_awaited_once()  # it closes nothing it borrowed
        assert controller._workflow is old and controller.backend == "adk"

    async def test_a_failed_give_keeps_the_running_one(self):
        controller, old = self._controller_needing_swap()
        new = _FakeLangGraphWorkflow()
        old.give_services = AsyncMock(side_effect=RuntimeError("released while lent"))
        old.lend_services.return_value.given = False

        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=new,
        ):
            with pytest.raises(RuntimeError, match="released while lent"):
                await controller.switch_backend("langgraph")

        old.cleanup.assert_not_awaited()
        new.cleanup.assert_awaited_once()
        assert controller._workflow is old and controller.backend == "adk"

    async def test_a_ctrl_c_while_the_new_one_starts_releases_it_and_keeps_the_old(self):
        import asyncio

        controller, old = self._controller_needing_swap()
        new = _FakeLangGraphWorkflow()
        starting = asyncio.Event()

        async def hangs(**kwargs):
            starting.set()
            await asyncio.Event().wait()

        new.initialize_services = AsyncMock(side_effect=hangs)
        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=new,
        ):
            switching = asyncio.create_task(controller.switch_backend("langgraph"))
            await asyncio.wait_for(starting.wait(), timeout=5)
            switching.cancel()
            with pytest.raises(asyncio.CancelledError):
                await switching
            await controller._join_releases()

        old.give_services.assert_not_awaited()
        new.cleanup.assert_awaited_once()
        assert controller._workflow is old and controller.backend == "adk"

    async def test_a_manager_kept_after_a_failed_model_switch_lends_to_the_next_build(self):
        """The workflow is down, its manager kept for a revive (a failed
        reinitialization: it still holds the run's services), and a
        conversation starts or opens on another backend: the next start
        builds that backend on the kept manager's services, which are given
        over, then releases it. Nothing restarts there either."""
        controller = _make_lifecycle_controller()
        kept = _FakeADKWorkflow()
        kept.is_initialized = False
        controller._workflow = kept
        controller._init_error = RuntimeError("reinitialize failed")
        built = _FakeLangGraphWorkflow()
        loan = kept.lend_services.return_value
        steps: list[str] = []
        kept.give_services = AsyncMock(side_effect=lambda lent: steps.append("give"))
        kept.cleanup = AsyncMock(side_effect=lambda: steps.append("kept released"))
        built.initialize_services = AsyncMock(side_effect=lambda **kw: steps.append("start"))

        await controller.switch_backend("langgraph")
        kept.cleanup.assert_not_awaited()  # it holds the services until they are given
        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=built,
        ):
            assert await controller.ensure_initialized() is True
            await controller._join_releases()

        built.initialize_services.assert_awaited_once_with(services=loan)
        kept.give_services.assert_awaited_once_with(loan)
        assert steps == ["start", "give", "kept released"]
        assert controller._workflow is built and controller.backend == "langgraph"
        await controller.close()

    async def test_a_build_that_fails_on_the_kept_services_keeps_them_for_a_retry(self):
        controller = _make_lifecycle_controller()
        kept = _FakeADKWorkflow()
        kept.is_initialized = False
        controller._workflow = kept
        controller._init_error = RuntimeError("reinitialize failed")
        failing, working = _FakeLangGraphWorkflow(), _FakeLangGraphWorkflow()
        failing.initialize_services = AsyncMock(side_effect=RuntimeError("no key"))

        await controller.switch_backend("langgraph")
        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            side_effect=[failing, working],
        ):
            assert await controller.ensure_initialized() is False
            kept.give_services.assert_not_awaited()
            kept.cleanup.assert_not_awaited()
            failing.cleanup.assert_awaited_once()
            assert await controller.ensure_initialized() is True
            await controller._join_releases()

        working.initialize_services.assert_awaited_once_with(
            services=kept.lend_services.return_value
        )
        kept.cleanup.assert_awaited_once()
        await controller.close()

    async def test_close_releases_a_kept_manager_no_build_took(self):
        controller = _make_lifecycle_controller()
        kept = _FakeADKWorkflow()
        kept.is_initialized = False
        controller._workflow = kept

        await controller.switch_backend("langgraph")
        await controller.close()

        kept.cleanup.assert_awaited_once()

    async def test_a_manager_that_lends_nothing_starts_on_its_own(self):
        """An application's own manager, not a ``BaseWorkflowManager``: the
        new one builds its own services, as before."""
        controller = _make_lifecycle_controller()
        old = SimpleNamespace(
            backend_type="adk", model="m", is_initialized=True, cleanup=AsyncMock(),
        )
        controller._workflow = old
        new = _FakeLangGraphWorkflow()

        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=new,
        ):
            await controller.switch_backend("langgraph")

        new.initialize_services.assert_awaited_once_with()
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

        controller = _make_lifecycle_controller(orchestrator=OrchestratorType.ADK)
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
                controller.switch_backend("langgraph"), controller.switch_backend("langgraph")
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
        controller = _make_lifecycle_controller(orchestrator=OrchestratorType.ADK)
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
            await controller.switch_backend("langgraph")

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
