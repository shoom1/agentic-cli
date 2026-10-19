"""The ADK runner is built from an ``App`` that carries the framework's plugins.

ADK deprecates ``Runner(plugins=...)``: plugins belong to an ``App``, and the
runner is built from it. Both ways the manager builds its agent tree (from
``AgentConfig``s, or natively from an ADK ``root_agent.yaml``) must end with a
runner whose app carries the permission, tool-error, task-progress and
LLM-logging plugins, without ADK's deprecation warning.
"""

from __future__ import annotations

import textwrap
import warnings

import pytest

pytest.importorskip("google.adk")

from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin  # noqa: E402
from agentic_cli.workflow.adk.plugins import LLMLoggingPlugin  # noqa: E402
from agentic_cli.workflow.adk.task_progress_plugin import TaskProgressPlugin  # noqa: E402
from agentic_cli.workflow.adk.tool_error_plugin import ToolErrorPlugin  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from tests.conftest import MockContext  # noqa: E402

PLUGIN_TYPES = (PermissionPlugin, ToolErrorPlugin, TaskProgressPlugin, LLMLoggingPlugin)
DEPRECATED_PLUGINS = "`plugins` argument is deprecated"


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    with MockContext(
        google_api_key="test-key", session_store="memory", raw_llm_logging=True
    ) as context:
        yield context


def _native_yaml(tmp_path):
    path = tmp_path / "root_agent.yaml"
    path.write_text(
        textwrap.dedent(
            """
            agent_class: LlmAgent
            name: root
            model: gemini-2.5-flash
            instruction: You are root.
            """
        ),
        encoding="utf-8",
    )
    return str(path)


async def _initialized(manager: GoogleADKWorkflowManager) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await manager.initialize_services()
    return caught


def _assert_built_from_an_app(manager, caught) -> None:
    app = manager._runner.app
    assert app.name == manager.app_name
    # The runner looks sessions up by its app_name: 0.6.x sessions are stored
    # under the manager's, so it must be the same string.
    assert manager._runner.app_name == manager.app_name
    assert app.root_agent is manager._root_agent
    for plugin_type in PLUGIN_TYPES:
        assert [p for p in app.plugins if isinstance(p, plugin_type)], plugin_type.__name__
    # What runs is what the app carries.
    assert manager._runner.plugin_manager.plugins == app.plugins
    deprecated = [str(w.message) for w in caught if DEPRECATED_PLUGINS in str(w.message)]
    assert deprecated == []


async def test_a_manager_built_from_agent_configs_runs_an_app(ctx):
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="assistant", prompt="Be brief.")],
        settings=ctx.settings,
        model="gemini-2.5-flash",
    )
    caught = await _initialized(manager)
    try:
        _assert_built_from_an_app(manager, caught)
    finally:
        await manager.cleanup()


async def test_a_native_adk_config_runs_the_same_app(ctx, tmp_path):
    manager = GoogleADKWorkflowManager(
        agent_configs=[],
        settings=ctx.settings,
        model="gemini-2.5-flash",
        adk_config_path=_native_yaml(tmp_path),
    )
    caught = await _initialized(manager)
    try:
        _assert_built_from_an_app(manager, caught)
        assert manager._runner.app.root_agent.name == "root"
    finally:
        await manager.cleanup()


async def test_a_model_switch_keeps_the_plugins(ctx):
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="assistant", prompt="Be brief.")],
        settings=ctx.settings,
        model="gemini-2.5-flash",
    )
    await _initialized(manager)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await manager.reinitialize(model="gemini-2.5-pro", preserve_sessions=True)
        _assert_built_from_an_app(manager, caught)
    finally:
        await manager.cleanup()
