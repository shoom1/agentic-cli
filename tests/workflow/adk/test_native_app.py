"""ADK: a native ADK YAML app (``adk_config_path``) is built from its YAML and run as any app.

The YAML only decides how the agents are built. It is loaded when the
manager validates its agents, before any model, service or store: a root
that is not an ``LlmAgent``, or a task-mode agent anywhere in the tree, is
refused there with the agent named. Agent configs passed with it are
ignored, with a warning.

Every provider base URL points at a dead local port, and both Anthropic
client bindings (the framework's and the one ADK's own ``AnthropicLlm``
builds for a bare ``claude-*`` model) are routed to a fake API.
"""

from __future__ import annotations

import textwrap

import pytest

pytest.importorskip("google.adk")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk import anthropic_llm  # noqa: E402
from agentic_cli.workflow.adk import manager as adk_manager  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig, AgentGraphError  # noqa: E402
from tests.workflow.adk.fakes import dead_base_urls  # noqa: E402

MODEL = "claude-sonnet-4-6"


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    dead_base_urls(monkeypatch)
    return tmp_path


class _LogRecorder:
    """Stands in for the manager's logger and records every call."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __getattr__(self, level: str):
        def log(event: str, *args, **fields) -> None:
            self.events.append({"event": event, "log_level": level, **fields})

        return log


@pytest.fixture
def logged(monkeypatch) -> list[dict]:
    recorder = _LogRecorder()
    monkeypatch.setattr(adk_manager, "logger", recorder)
    return recorder.events


def settings(**overrides) -> BaseSettings:
    return BaseSettings(**{"session_store": "memory", "default_model": MODEL, **overrides})


def write(project, name: str, text: str, extra: str = "") -> str:
    """``text`` (dedented) and then ``extra`` (as it is) in ``name``."""
    path = project / name
    path.write_text(textwrap.dedent(text) + extra, encoding="utf-8")
    return str(path)


def root_yaml(project, *, model: str = MODEL, extra: str = "") -> str:
    return write(project, "root_agent.yaml", f"""
        agent_class: LlmAgent
        name: root
        model: {model}
        instruction: You are root.
    """, extra)


def native(path: str, configs=(), **overrides) -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=list(configs), settings=settings(**overrides), adk_config_path=path
    )


class TestTheYAMLIsLoadedWhenItIsValidated:
    async def test_a_root_that_is_not_an_llm_agent_is_refused(self, project):
        write(project, "helper.yaml", """
            agent_class: LlmAgent
            name: helper
            model: claude-sonnet-4-6
            instruction: You help.
        """)
        path = write(project, "root_agent.yaml", """
            agent_class: SequentialAgent
            name: pipeline
            sub_agents:
              - config_path: helper.yaml
        """)
        manager = native(path)

        with pytest.raises(AgentGraphError, match="root agent 'pipeline' is a SequentialAgent"):
            await manager.initialize_services()

        # Refused before anything was allocated.
        assert not manager.is_initialized
        assert manager._session_service is None
        assert manager.catalog is None

    @pytest.mark.parametrize("nested", [False, True], ids=["a-sub-agent", "deeper"])
    async def test_a_task_mode_agent_is_refused_by_name(self, project, nested):
        task = "sub_agents:\n  - code: tests.workflow.adk.native_agents.asker\n"
        if nested:
            write(project, "helper.yaml", """
                agent_class: LlmAgent
                name: helper
                model: claude-sonnet-4-6
                instruction: You help.
            """, task)
            path = root_yaml(project, extra="sub_agents:\n  - config_path: helper.yaml\n")
        else:
            path = root_yaml(project, extra=task)
        manager = native(path)

        with pytest.raises(AgentGraphError) as raised:
            await manager.initialize_services()

        assert str(raised.value) == (
            "agent 'asker' is in task mode, which is not supported; use single_turn or chat"
        )
        assert manager._session_service is None

    async def test_its_root_is_the_main_agent(self, project):
        manager = native(root_yaml(project))
        await manager.initialize_services()
        try:
            assert manager.root_agent_name == "root"
        finally:
            await manager.cleanup()

    async def test_agent_configs_passed_with_it_are_ignored_with_a_warning(
        self, project, logged
    ):
        manager = native(root_yaml(project), configs=[AgentConfig(name="a", prompt="Help.")])

        assert manager.agent_configs == []
        (warning,) = [e for e in logged if e["event"] == "adk_native_configs_ignored"]
        assert (warning["log_level"], warning["agents"]) == ("warning", ["a"])

    async def test_only_a_native_app_points_bare_claude_models_at_anthropics_api(
        self, project, monkeypatch
    ):
        calls: list[str] = []
        monkeypatch.setattr(
            anthropic_llm, "ensure_direct_anthropic", lambda: calls.append("registered")
        )
        built = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt="Help.")], settings=settings()
        )
        await built.initialize_services()
        await built.cleanup()
        assert calls == []

        manager = native(root_yaml(project))
        assert calls == []
        await manager.initialize_services()
        await manager.cleanup()
        assert calls == ["registered"]
