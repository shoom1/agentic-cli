"""Every workflow manager keeps the sub-agent policy it is built with (``subagents=``).

``BaseWorkflowManager`` takes ``subagents=`` (a ``SubAgentPolicy``, or None
for ``SubAgentPolicy()``) and keeps it as ``subagent_policy``, read-only;
the ADK and Deep Agents managers pass it on, and the LangGraph manager takes
it too, which its own delegation leaves unread. The factory passes it to
whichever backend it builds. A value that is no policy is a ``TypeError``
where it is passed: the factory's comes before any backend is imported.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

pytest.importorskip("google.adk")

from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.agents.policy import AgentLimits, SubAgentPolicy  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.factory import create_workflow_manager_from_settings  # noqa: E402
from agentic_cli.workflow.settings import OrchestratorType  # noqa: E402
from tests.conftest import MockContext  # noqa: E402

POLICY = SubAgentPolicy(max_active=2, max_runs_per_turn=5, limits=AgentLimits(timeout=60))
CONFIGS = [AgentConfig(name="solo", prompt="p")]


def _needs(extra: str):
    return pytest.mark.skipif(
        importlib.util.find_spec(extra) is None, reason=f"needs the {extra} extra"
    )


def adk() -> type:
    return GoogleADKWorkflowManager


def deepagents() -> type:
    from agentic_cli.workflow.deepagents import DeepAgentsWorkflowManager

    return DeepAgentsWorkflowManager


def langgraph() -> type:
    from agentic_cli.workflow.langgraph import LangGraphWorkflowManager

    return LangGraphWorkflowManager


BACKENDS = [
    pytest.param(adk, id="adk"),
    pytest.param(deepagents, id="deepagents", marks=_needs("deepagents")),
    pytest.param(langgraph, id="langgraph", marks=_needs("langgraph")),
]
ORCHESTRATORS = [
    pytest.param(OrchestratorType.ADK, id="adk"),
    pytest.param(OrchestratorType.DEEPAGENTS, id="deepagents", marks=_needs("deepagents")),
    pytest.param(OrchestratorType.LANGGRAPH, id="langgraph", marks=_needs("langgraph")),
]


class TestTheManager:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_it_keeps_the_policy_it_is_given(self, backend, mock_context):
        manager = backend()(agent_configs=CONFIGS, settings=mock_context.settings, subagents=POLICY)
        assert manager.subagent_policy is POLICY

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_without_one_it_has_the_default(self, backend, mock_context):
        manager = backend()(agent_configs=CONFIGS, settings=mock_context.settings)
        assert manager.subagent_policy == SubAgentPolicy()

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_none_is_the_default(self, backend, mock_context):
        manager = backend()(agent_configs=CONFIGS, settings=mock_context.settings, subagents=None)
        assert manager.subagent_policy == SubAgentPolicy()

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_a_value_that_is_no_policy_is_a_type_error(self, backend, mock_context):
        with pytest.raises(TypeError, match="subagents must be a SubAgentPolicy or None, not dict"):
            backend()(agent_configs=CONFIGS, settings=mock_context.settings,
                      subagents={"max_active": 2})

    def test_it_is_read_only(self, mock_context):
        manager = GoogleADKWorkflowManager(agent_configs=CONFIGS, settings=mock_context.settings)
        with pytest.raises(AttributeError):
            manager.subagent_policy = POLICY  # type: ignore[misc]

    def test_a_manager_made_without_its_constructor_has_the_default(self):
        # A test harness's ``__new__``.
        manager = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
        assert manager.subagent_policy == SubAgentPolicy()

    async def test_a_model_switch_keeps_it(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        with MockContext(google_api_key="a-test-key") as context:
            manager = GoogleADKWorkflowManager(
                agent_configs=CONFIGS, settings=context.settings, model="gemini-2.5-flash",
                subagents=POLICY,
            )
            await manager.initialize_services()
            try:
                await manager.reinitialize(model="gemini-2.5-pro", preserve_sessions=True)
                assert manager.is_initialized
                assert manager.subagent_policy is POLICY
            finally:
                await manager.cleanup()


class TestTheFactory:
    @pytest.mark.parametrize("orchestrator", ORCHESTRATORS)
    def test_the_backend_it_builds_gets_the_policy(self, orchestrator, mock_context):
        manager = create_workflow_manager_from_settings(
            CONFIGS, mock_context.settings, orchestrator=orchestrator, subagents=POLICY,
        )
        assert manager.backend_type == orchestrator.value
        assert manager.subagent_policy is POLICY

    def test_without_one_the_default(self, mock_context):
        manager = create_workflow_manager_from_settings(CONFIGS, mock_context.settings)
        assert manager.subagent_policy == SubAgentPolicy()

    def test_a_native_adk_app_gets_it_too(self, tmp_path, mock_context):
        manager = create_workflow_manager_from_settings(
            [], mock_context.settings, adk_config_path=str(tmp_path / "root_agent.yaml"),
            subagents=POLICY,
        )
        assert isinstance(manager, GoogleADKWorkflowManager)
        assert manager.subagent_policy is POLICY

    @_needs("deepagents")
    def test_a_value_that_is_no_policy_fails_before_a_backend_is_imported(
        self, monkeypatch, mock_context
    ):
        # Without the extra, the backend's import would fail with the
        # install hint: the wrong argument is named first.
        monkeypatch.setitem(sys.modules, "agentic_cli.workflow.deepagents", None)
        with pytest.raises(TypeError, match="subagents must be a SubAgentPolicy or None, not int"):
            create_workflow_manager_from_settings(
                CONFIGS, mock_context.settings, orchestrator="deepagents", subagents=4,
            )
