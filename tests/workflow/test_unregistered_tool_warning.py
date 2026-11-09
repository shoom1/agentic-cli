"""An agent's tool the registry did not issue is named at startup.

The permission gate denies every call to a plain callable the registry did not
issue ("tool is not registered"), so an app that forgot ``@register_tool``
built an agent whose tool never runs, and learned it only from the model's
refusals. The tool assembly every backend shares now says so once, when the
agent is built: a ``tool_not_registered`` warning naming the agent and the
tool. A tool object (an ADK built-in such as ``GoogleSearchTool`` runs inside
the model and never reaches the gate) is not a plain callable and is not named.
"""

from __future__ import annotations

import functools

import pytest

from agentic_cli.tools import job_list
from agentic_cli.workflow import base_manager
from agentic_cli.workflow.config import AgentConfig
from tests.conftest import MockContext


class _LogRecorder:
    """Stands in for the base manager's logger and records every call, so the
    test does not depend on how an earlier test configured structlog."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.events.append({"event": event, "log_level": level, **fields})

        return log

    def unregistered(self) -> list[tuple[str, str]]:
        return [
            (e["agent"], e["tool"]) for e in self.events if e["event"] == "tool_not_registered"
        ]


@pytest.fixture
def logs(monkeypatch) -> _LogRecorder:
    recorder = _LogRecorder()
    monkeypatch.setattr(base_manager, "logger", recorder)
    return recorder


def plain_tool(text: str) -> dict:
    """A tool an app forgot to register."""
    return {"success": True, "text": text}


class _CallableTool:
    """A callable object, not a tool class."""

    __name__ = "callable_tool"

    def __call__(self, text: str) -> dict:
        return {"success": True}


def _stub_manager(tools: list) -> tuple[base_manager.BaseWorkflowManager, AgentConfig]:
    from unittest.mock import MagicMock

    class _Manager(base_manager.BaseWorkflowManager):
        def _get_state_tools(self):
            return []

        @property
        def backend_type(self) -> str:
            return "test"

        async def _do_initialize(self) -> None:
            return None

        async def process(self, message, user_id, session_id=None):
            raise NotImplementedError

        async def reinitialize(self, model=None, preserve_sessions=True):
            return None

        async def cleanup(self):
            return None

    settings = MagicMock()
    settings.app_name = "test-app"
    config = AgentConfig(name="assistant", prompt="p", tools=list(tools), include_state_tools=False)
    return _Manager(agent_configs=[config], settings=settings), config


class TestTheAssemblyNamesAnUnregisteredTool:
    def test_a_plain_function_is_named_with_its_agent(self, logs):
        manager, config = _stub_manager([plain_tool, job_list])

        manager._build_tools(config, service_map={})

        assert logs.unregistered() == [("assistant", "plain_tool")]
        (event,) = [e for e in logs.events if e["event"] == "tool_not_registered"]
        assert event["log_level"] == "warning"

    def test_a_registered_tool_is_not_named(self, logs):
        manager, config = _stub_manager([job_list])

        manager._build_tools(config, service_map={})

        assert logs.unregistered() == []

    @pytest.mark.parametrize(
        "tool, name",
        [
            (functools.partial(plain_tool, "x"), "plain_tool"),
            (_CallableTool(), "callable_tool"),
            (lambda text: {"success": True}, "<lambda>"),
        ],
        ids=["partial", "callable-object", "lambda"],
    )
    def test_any_plain_callable_is_named(self, logs, tool, name):
        manager, config = _stub_manager([tool])

        manager._build_tools(config, service_map={})

        assert logs.unregistered() == [("assistant", name)]

    def test_an_adk_built_in_tool_object_is_not_named(self, logs):
        pytest.importorskip("google.adk")
        from google.adk.tools import google_search

        manager, config = _stub_manager([google_search])

        manager._build_tools(config, service_map={})

        assert logs.unregistered() == []

    def test_an_adk_function_tool_is_judged_by_the_function_it_runs(self, logs):
        pytest.importorskip("google.adk")
        from google.adk.tools import FunctionTool, LongRunningFunctionTool

        manager, config = _stub_manager([
            FunctionTool(func=plain_tool),
            LongRunningFunctionTool(func=plain_tool),
            FunctionTool(func=job_list),
        ])

        manager._build_tools(config, service_map={})

        assert logs.unregistered() == [("assistant", "plain_tool")]

    def test_a_langchain_tool_is_judged_by_the_function_it_runs(self, logs):
        pytest.importorskip("langchain_core")
        from langchain_core.tools import StructuredTool, tool

        @tool
        def decorated(text: str) -> dict:
            """A LangChain tool an app forgot to register."""
            return {"success": True}

        manager, config = _stub_manager([decorated, StructuredTool.from_function(func=job_list)])

        manager._build_tools(config, service_map={})

        assert logs.unregistered() == [("assistant", "decorated")]

    def test_an_apps_own_tool_class_is_named(self, logs):
        pytest.importorskip("google.adk")
        from google.adk.tools.base_tool import BaseTool

        class LookupTool(BaseTool):
            def __init__(self):
                super().__init__(name="lookup", description="Looks something up.")

        manager, config = _stub_manager([LookupTool()])

        manager._build_tools(config, service_map={})

        assert logs.unregistered() == [("assistant", "lookup")]

    def test_a_strange_callable_never_stops_the_build(self, logs):
        class _Odd:
            def __call__(self):
                return {"success": True}

            def __getattr__(self, name):
                raise RuntimeError("no attributes here")

        _Odd.__module__ = None
        manager, config = _stub_manager([_Odd(), plain_tool])

        built = manager._build_tools(config, service_map={})

        assert len(built) == 2
        assert ("assistant", "plain_tool") in logs.unregistered()

    def test_it_is_named_once_however_often_the_agent_is_built(self, logs):
        manager, config = _stub_manager([plain_tool])

        manager._build_tools(config, service_map={})
        manager._build_tools(config, service_map={})

        assert logs.unregistered() == [("assistant", "plain_tool")]


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    with MockContext(google_api_key="test-key", session_store="memory") as context:
        yield context


def _adk_manager_cls():
    pytest.importorskip("google.adk")
    from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

    return GoogleADKWorkflowManager


def _deepagents_manager_cls():
    pytest.importorskip("deepagents")
    from agentic_cli.workflow.deepagents import DeepAgentsWorkflowManager

    return DeepAgentsWorkflowManager


@pytest.mark.parametrize("manager_cls", [_adk_manager_cls, _deepagents_manager_cls], ids=["adk", "deepagents"])
async def test_each_backend_names_it_once_across_a_model_switch(ctx, logs, manager_cls):
    manager = manager_cls()(
        agent_configs=[AgentConfig(name="assistant", prompt="p", tools=[plain_tool, job_list])],
        settings=ctx.settings,
        model="gemini-2.5-flash",
    )
    await manager.initialize_services()
    try:
        await manager.reinitialize(model="gemini-2.5-pro", preserve_sessions=True)
    finally:
        await manager.cleanup()

    assert logs.unregistered() == [("assistant", "plain_tool")]
