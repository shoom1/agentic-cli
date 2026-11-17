"""An app's sub-agent policy reaches every manager it builds (``BaseCLIApp(subagents=)``).

The app hands its ``subagents=`` to its ``WorkflowController``, which passes
it to each manager it builds: the first start's, and the one a conversation
on another backend starts (``switch_backend``), while the workflow runs or
while it is down. A value that is no policy is a ``TypeError`` before the
app builds anything. These run the real ADK and Deep Agents managers,
offline: no turn is taken, so no model is called.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("deepagents")

from agentic_cli.cli import AppInfo  # noqa: E402
from agentic_cli.cli import app as app_module  # noqa: E402
from agentic_cli.cli.app import BaseCLIApp  # noqa: E402
from agentic_cli.cli.workflow_controller import WorkflowController  # noqa: E402
from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.agents.policy import AgentLimits, SubAgentPolicy  # noqa: E402
from agentic_cli.workflow.base_manager import BaseWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402

POLICY = SubAgentPolicy(max_active=1, max_runs_per_turn=3, limits=AgentLimits(max_tokens=9_000))


@pytest.fixture(autouse=True)
def configured(tmp_path: Path, monkeypatch) -> list:
    """A temporary cwd and HOME with a fake Claude key, and an app that leaves
    no logging configuration or warning hook behind: the calls the app made
    to ``configure_logging``."""
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    calls: list = []
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(app_module, "install_log_filters", lambda: None)
    return calls


def app(**kwargs) -> BaseCLIApp:
    return BaseCLIApp(
        app_info=AppInfo(name="Policy test", version="0"),
        agent_configs=[AgentConfig(name="solo", prompt="Answer.", include_state_tools=False)],
        settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6"),
        **kwargs,
    )


async def started(controller: WorkflowController) -> BaseWorkflowManager:
    """The manager ``controller``'s start builds, once it is ready."""
    await controller.start_background_init()
    assert await controller.ensure_initialized() is True, controller.init_error
    return controller.workflow


class TestEveryManagerTheAppBuilds:
    async def test_the_first_one_gets_the_policy(self):
        controller = app(subagents=POLICY)._workflow_controller
        try:
            workflow = await started(controller)
            assert workflow.backend_type == "adk"
            assert workflow.subagent_policy is POLICY
        finally:
            await controller.close()

    async def test_a_conversation_on_another_backend_keeps_it(self):
        controller = app(subagents=POLICY)._workflow_controller
        try:
            first = await started(controller)
            await controller.switch_backend("deepagents")
            workflow = controller.workflow
            assert workflow is not first and workflow.backend_type == "deepagents"
            assert workflow.subagent_policy is POLICY
        finally:
            await controller.close()

    async def test_a_switch_while_the_workflow_is_down_keeps_it(self):
        controller = app(subagents=POLICY)._workflow_controller
        try:
            await controller.switch_backend("deepagents")
            workflow = await started(controller)
            assert workflow.backend_type == "deepagents"
            assert workflow.subagent_policy is POLICY
        finally:
            await controller.close()


class TestAValueThatIsNoPolicy:
    def test_the_app_refuses_it_before_it_builds_anything(self, configured, monkeypatch):
        built: list = []
        monkeypatch.setattr(app_module, "WorkflowController", lambda *a, **k: built.append(k))
        with pytest.raises(TypeError, match="subagents must be a SubAgentPolicy or None, not type"):
            app(subagents=SubAgentPolicy)
        assert configured == [] and built == []

    def test_the_controller_refuses_it(self):
        with pytest.raises(TypeError, match="subagents must be a SubAgentPolicy or None, not dict"):
            WorkflowController(
                [AgentConfig(name="solo", prompt="p")], BaseSettings(session_store="memory"),
                subagents={"max_active": 1},
            )
