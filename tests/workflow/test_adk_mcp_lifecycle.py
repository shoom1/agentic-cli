"""On ADK, an agent's MCP servers live and die with the manager's runner.

The manager dropped its runner without closing it, so each agent's
``McpToolset`` stayed open: a stdio server outlived ``cleanup()``, and each
model switch (a reinitialize builds new toolsets) left one more running.
``Runner.close()`` closes the agents' toolsets and plugins and only flushes
the session service, which a model switch keeps.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("mcp")

import anthropic  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.mcp import MCPServerConfig  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402

#: The stdio MCP server the Deep Agents tests start; it records its pid.
SERVER = Path(__file__).parent / "deepagents" / "mcp_server.py"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A hermetic cwd and HOME, with a fake Claude key."""
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    return tmp_path


@pytest.fixture
def pid_dir(project):
    """Where each server records its pid; afterwards, none may be alive."""
    folder = project / "server-pids"
    folder.mkdir()
    yield folder
    left = [int(p.name) for p in folder.iterdir() if _alive(int(p.name))]
    for pid in left:  # never leak into the next test
        os.kill(pid, signal.SIGKILL)


def alive(pid_dir: Path) -> list[int]:
    return [int(p.name) for p in pid_dir.iterdir() if _alive(int(p.name))]


async def turn(manager) -> None:
    async for _ in manager.process("Hello", "u", session_id="s"):
        pass


async def test_cleanup_stops_the_servers_and_a_model_switch_leaves_one(
    project, pid_dir, monkeypatch
):
    route(monkeypatch, anthropic, "AsyncAnthropic",
          FakeAPI(anthropic_message({"type": "text", "text": "ok"})))
    server = MCPServerConfig(
        name="atlas", command=sys.executable, args=[str(SERVER)], timeout=30,
        env={"MCP_TEST_PID_DIR": str(pid_dir)},
    )
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="Answer.", mcp_servers=[server])],
        settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6"),
    )
    try:
        await turn(manager)
        assert len(alive(pid_dir)) == 1
        await manager.reinitialize(model="claude-haiku-4-5")  # a /settings model switch
        await turn(manager)
        assert len(alive(pid_dir)) == 1, "the switch left the old server running"
    finally:
        await manager.cleanup()

    assert alive(pid_dir) == [], "a server outlived cleanup()"


async def test_a_native_apps_toolset_outlives_its_runner_and_serves_the_next(
    project, pid_dir, monkeypatch
):
    """A native ADK YAML app's agents are loaded once per manager, so a model
    switch builds the new runner on the same toolset object: the old runner
    closed it (its server stops), and the new one starts it again."""
    import json
    import textwrap

    from tests.workflow.adk.fakes import dead_base_urls, route_anthropic

    dead_base_urls(monkeypatch)
    api = FakeAPI(anthropic_message({"type": "text", "text": "ok"}, model="claude-sonnet-4-6"))
    route_anthropic(monkeypatch, api)
    (project / "atlas_tools.py").write_text(textwrap.dedent(f"""
        import sys

        from agentic_cli.workflow.mcp import MCPServerConfig, to_adk_toolset

        atlas = to_adk_toolset(MCPServerConfig(
            name="atlas", command=sys.executable, args=[{str(SERVER)!r}], timeout=30,
            env={{"MCP_TEST_PID_DIR": {str(pid_dir)!r}}},
        ))
    """), encoding="utf-8")
    monkeypatch.syspath_prepend(str(project))
    monkeypatch.delitem(sys.modules, "atlas_tools", raising=False)
    (project / "root_agent.yaml").write_text(textwrap.dedent("""
        agent_class: LlmAgent
        name: root
        model: claude-sonnet-4-6
        instruction: Answer.
        tools:
          - name: atlas_tools.atlas
    """), encoding="utf-8")
    manager = GoogleADKWorkflowManager(
        agent_configs=[],
        settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6"),
        adk_config_path=str(project / "root_agent.yaml"),
    )
    try:
        await turn(manager)
        assert len(alive(pid_dir)) == 1
        await manager.reinitialize(model="claude-haiku-4-5")
        assert alive(pid_dir) == [], "the closed runner left the server running"
        await turn(manager)
        assert len(alive(pid_dir)) == 1
    finally:
        await manager.cleanup()

    assert alive(pid_dir) == [], "a server outlived cleanup()"
    tools = [sorted(t["name"] for t in json.loads(r.content).get("tools", [])) for r in api.requests]
    assert tools == [["add", "lookup"], ["add", "lookup"]]
