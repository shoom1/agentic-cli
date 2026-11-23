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
from tests.loop_guard import no_blocking_calls  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402
from tests.workflow.test_subagents_contract import (  # noqa: E402,F401 - fixtures
    ADK_MCP_WARNINGS,
    CHOICE,
    FOLLOWING,
    GENERAL,
    agent,
    backend,
    calls,
    delegate,
    envelope_read,
    make_crew,
    message,
    says,
)

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


async def test_a_sub_agents_mcp_tools_are_listed_when_first_asked_and_stop_with_it(
    project, pid_dir, monkeypatch
):
    """What ``task``'s ``tools`` may name includes a sub-agent's MCP tools, by
    their prefixed names: ADK connects its toolset to list them, once a
    build, and the toolset's runner stops the server."""
    route(monkeypatch, anthropic, "AsyncAnthropic",
          FakeAPI(anthropic_message({"type": "text", "text": "ok"})))
    server = MCPServerConfig(
        name="atlas", command=sys.executable, args=[str(SERVER)], timeout=30,
        env={"MCP_TEST_PID_DIR": str(pid_dir)}, tool_name_prefix="atlas",
    )
    manager = GoogleADKWorkflowManager(
        agent_configs=[
            AgentConfig(name="lead", prompt="Lead.", sub_agents=["researcher"]),
            AgentConfig(name="researcher", prompt="Find.", mcp_servers=[server]),
        ],
        settings=BaseSettings(session_store="memory",
                              default_model="claude-sonnet-4-6"),
    )
    try:
        await manager.initialize_services()
        assert list(pid_dir.iterdir()) == []  # ADK connects when it is asked

        with no_blocking_calls():
            names = await manager._agent_tool_names("researcher")
            again = await manager._agent_tool_names("researcher")

        assert names == again == {"atlas_add", "atlas_lookup"}
        assert len(list(pid_dir.iterdir())) == 1  # listed once
    finally:
        await manager.cleanup()

    assert alive(pid_dir) == [], "a server outlived cleanup()"


async def test_a_listing_that_fails_is_said_so_and_asked_again(
    project, pid_dir, monkeypatch
):
    """A toolset whose listing fails does not leave its names out of the answer
    (the type would seem to lack them): the manager raises, and the next ask lists
    again. A ``task`` call naming tools is told to try again, no run started."""
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

    from agentic_cli.workflow.agents.delegate import (
        DelegationRequest,
        ToolsUnlistedError,
    )
    from agentic_cli.workflow.sessions import (
        SessionRef,
        reset_active_turn,
        set_active_turn,
    )

    route(monkeypatch, anthropic, "AsyncAnthropic",
          FakeAPI(anthropic_message({"type": "text", "text": "ok"})))
    server = MCPServerConfig(
        name="atlas", command=sys.executable, args=[str(SERVER)], timeout=30,
        env={"MCP_TEST_PID_DIR": str(pid_dir)}, tool_name_prefix="atlas",
    )
    manager = GoogleADKWorkflowManager(
        agent_configs=[
            AgentConfig(name="lead", prompt="Lead.", sub_agents=["researcher"]),
            AgentConfig(name="researcher", prompt="Find.", description="Finds.",
                        mcp_servers=[server]),
        ],
        settings=BaseSettings(session_store="memory",
                              default_model="claude-sonnet-4-6"),
    )
    real = McpToolset.get_tools_with_prefix
    failing = [True]

    async def listing(self, *args, **kwargs):
        if failing[0]:
            raise OSError("the server is down")
        return await real(self, *args, **kwargs)

    monkeypatch.setattr(McpToolset, "get_tools_with_prefix", listing)
    try:
        await manager.initialize_services()

        with pytest.raises(ToolsUnlistedError) as caught:
            await manager._agent_tool_names("researcher")
        assert caught.value.reason == "OSError: the server is down"

        conversation = SessionRef(manager.app_name, "u", "s")
        token = set_active_turn(conversation)
        try:
            refused = await manager._delegations.delegate(DelegationRequest(
                caller=None, caller_agent="lead", call_id="c1",
                arguments={"agent_type": "researcher", "title": "T", "objective": "O",
                           "tools": ["atlas_lookup"]},
            ))
        finally:
            reset_active_turn(token)
        assert refused.result == {"success": False, "error": (
            "The tools of 'researcher' could not be listed just now "
            "(OSError: the server is down); try again."
        )}
        assert refused.run is None

        failing[0] = False
        assert await manager._agent_tool_names("researcher") == {"atlas_add",
                                                                 "atlas_lookup"}
    finally:
        await manager.cleanup()


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


async def test_a_model_choices_runner_is_released_with_the_backend(
    make_crew, backend, pid_dir
):
    """A sub-agent type with an MCP server and a model choice, run on the
    choice: the release closes what the variant holds. ADK's variant has a
    toolset (a server) of its own, stopped by a model switch and by cleanup
    and started again by the next run; Deep Agents' variant graph is dropped
    by the release, the manager's MCP sessions its tools."""
    server = MCPServerConfig(
        name="atlas", command=sys.executable, args=[str(SERVER)], timeout=30,
        env={"MCP_TEST_PID_DIR": str(pid_dir)}, tool_name_prefix="atlas",
    )
    crew = make_crew(
        [agent("coordinator", sub_agents=("researcher",)),
         agent("researcher", mcp_servers=[server], model_choices=(CHOICE,))],
        subagents=FOLLOWING,
        coordinator=[calls(delegate("researcher", "Find X.", "c1", model=CHOICE)),
                     says("Asked."),
                     calls(message("researcher-1", "Now find Y.", "c2")),
                     says("Done.")],
        researcher=[says("X is 42."), says("Y is 7.")],
    )
    manager = crew.manager
    try:
        await crew.turn()
        assert alive(pid_dir)
        assert len(manager._variant_graphs if backend == "deepagents"
                   else manager._variant_runners) == 1

        await manager.reinitialize(model="claude-opus-4-7", preserve_sessions=True)
        if backend == "adk":
            assert alive(pid_dir) == []  # no runner of the old build is left open
        await crew.turn("Find Y too.")
        assert alive(pid_dir)
        assert envelope_read(
            crew.results_read("coordinator")["c2"]
        )["result"] == "Y is 7."
    finally:
        await manager.cleanup()

    assert alive(pid_dir) == [], "a server outlived cleanup()"
    assert getattr(manager, "_variant_graphs", {}) == {}
    assert dict(getattr(manager, "_variant_runners", {})) == {}


@ADK_MCP_WARNINGS
async def test_a_general_purpose_agents_runner_is_released_with_the_backend(
    make_crew, backend, pid_dir
):
    """A type with an MCP server that starts the built-in general-purpose type:
    its agent has a server of its own (ADK: a toolset of its runner's; Deep
    Agents: sessions of its own), which a model switch and cleanup stop."""
    server = MCPServerConfig(
        name="atlas", command=sys.executable, args=[str(SERVER)], timeout=30,
        env={"MCP_TEST_PID_DIR": str(pid_dir)}, tool_name_prefix="atlas",
    )
    crew = make_crew(
        [agent("coordinator", mcp_servers=[server])], subagents=GENERAL,
        coordinator=[calls(delegate("general_purpose", "Find X.", "c1")), says("Done.")],
        general_purpose=[says("X is 42.")],
    )
    manager = crew.manager
    try:
        await crew.turn()
        assert len(manager._general_graphs if backend == "deepagents"
                   else manager._general_runners) == 1
        # The spawner's own server and the general-purpose agent's.
        assert len(alive(pid_dir)) == 2

        await manager.reinitialize(model="claude-opus-4-7", preserve_sessions=True)
        if backend == "adk":
            assert alive(pid_dir) == []  # no runner of the old build is left open
    finally:
        await manager.cleanup()

    assert alive(pid_dir) == [], "a server outlived cleanup()"
    assert getattr(manager, "_general_graphs", {}) == {}
    assert dict(getattr(manager, "_general_runners", {})) == {}
