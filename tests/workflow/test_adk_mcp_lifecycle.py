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
        await manager.reinitialize(model="claude-sonnet-4-5")  # a /settings model switch
        await turn(manager)
        assert len(alive(pid_dir)) == 1, "the switch left the old server running"
    finally:
        await manager.cleanup()

    assert alive(pid_dir) == [], "a server outlived cleanup()"
