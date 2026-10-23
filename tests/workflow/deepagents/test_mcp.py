"""MCP tools on Deep Agents (``workflow/deepagents/mcp_tools.py``).

An agent's ``mcp_servers`` open at initialization, one session per (agent,
server), each owned by its own task: MCP's anyio contexts must be left by the
task that entered them. Their tools load through ``langchain-mcp-adapters``,
with ``tool_filter`` and ``tool_name_prefix`` applied as ADK applies them, and
each tool object is bound, in the manager's own map, to a definition gated
under the synthetic ``mcp`` capability with the tool's name as the target. A
server that cannot start, and a tool whose name is taken, leave the agent
without them, and the next turn opens with a notice. Cleanup, a failed
initialization and a model switch close the sessions.

The server is real: ``mcp_server.py`` started with this interpreter over
stdio. Every test checks that no server process outlives it. Agent models are
scripted; nothing reaches a provider.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("langchain_mcp_adapters")

from langchain_core.messages import ToolMessage  # noqa: E402
from langchain_core.tools import BaseTool, StructuredTool  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.settings_persistence import get_user_config_path  # noqa: E402
from agentic_cli.tools import read_file  # noqa: E402
from agentic_cli.tools.registry import identify_tool  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.context import ContextBudgetError  # noqa: E402
from agentic_cli.workflow.deepagents import manager as manager_module  # noqa: E402
from agentic_cli.workflow.deepagents import mcp_tools  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.mcp_tools import MCPSessions  # noqa: E402
from agentic_cli.workflow.events import EventType, UserInputRequest  # noqa: E402
from agentic_cli.workflow.mcp import MCPServerConfig  # noqa: E402
from agentic_cli.workflow.permissions.capabilities import Capability  # noqa: E402
from agentic_cli.workflow.permissions.prompt import ALLOW_ONCE_CHOICE, DENY_CHOICE  # noqa: E402
from tests.conftest import MockContext  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedModels, ai, task  # noqa: E402

SERVER = Path(__file__).with_name("mcp_server.py")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def servers(tmp_path):
    """Configs for the test server; afterwards, no server process may be left.

    Each server records its process id in a directory of this test's own.
    """
    pid_dir = tmp_path / "server-pids"
    pid_dir.mkdir()

    def config(name: str = "atlas", *, env: dict | None = None, **fields) -> MCPServerConfig:
        fields.setdefault("command", sys.executable)
        fields.setdefault("args", [str(SERVER)])
        fields.setdefault("timeout", 30)  # a slow CI host still starts Python in time
        return MCPServerConfig(
            name=name, env={"MCP_TEST_PID_DIR": str(pid_dir), **(env or {})}, **fields
        )

    config.pids = lambda: sorted(int(p.name) for p in pid_dir.iterdir())
    yield config
    left = [pid for pid in config.pids() if _alive(pid)]
    for pid in left:  # never leak into the next test
        os.kill(pid, signal.SIGKILL)
    assert left == [], f"MCP server processes outlived the test: {left}"


class _Recorder:
    """Stands in for a module's structlog logger."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.calls.append((level, event, fields))

        return log

    def events(self, level: str | None = None) -> list[str]:
        return [e for lvl, e, _ in self.calls if level is None or lvl == level]


def _call(tool: BaseTool, args: dict, call_id: str = "c1") -> dict:
    return {"type": "tool_call", "id": call_id, "name": tool.name, "args": args}


# --- the sessions ---------------------------------------------------------------------


class TestMCPSessions:
    async def test_it_loads_the_servers_tools_bound_under_mcp(self, servers):
        sessions = MCPSessions()
        try:
            tools, notices = await sessions.start("researcher", [servers()])
            add, lookup = tools

            assert notices == []
            assert [tool.name for tool in tools] == ["add", "lookup"]
            assert add.description == "Add two integers."
            definition = sessions.identify(add)
            assert definition.name == "add"
            assert definition.capabilities == [Capability("mcp", target="add")]
            assert sessions.identify(lookup).capabilities == [
                Capability("mcp", target="lookup")
            ]
            # Bound in this manager's map only, never the framework registry's.
            assert identify_tool(add) is None
            assert sessions.identify(add.model_copy()) is None
        finally:
            await sessions.aclose()

    async def test_a_tool_runs_on_the_open_session_and_its_text_is_the_result(
        self, servers
    ):
        sessions = MCPSessions()
        try:
            (add, _), _ = await sessions.start("researcher", [servers()])

            message = await add.ainvoke(_call(add, {"a": 2, "b": 3}))
        finally:
            await sessions.aclose()

        assert isinstance(message, ToolMessage)
        assert (message.content, message.status) == ("5", "success")

    async def test_an_error_result_is_a_failed_result(self, servers):
        sessions = MCPSessions()
        try:
            (_, lookup), _ = await sessions.start("researcher", [servers()])

            message = await lookup.ainvoke(_call(lookup, {"city": "Atlantis"}))
        finally:
            await sessions.aclose()

        assert message.status == "error"
        assert json.loads(message.content) == {
            "success": False, "error": "No such city: Atlantis",
        }

    async def test_images_pass_through_as_content_blocks(self, monkeypatch):
        image = {"type": "image", "base64": "iVBORw0KGgo=", "mime_type": "image/png"}

        async def call_tool(**arguments):
            return [{"type": "text", "text": "A chart:"}, image], None

        _fake_sessions(monkeypatch, tools=[StructuredTool(
            name="chart", description="Draw a chart.", args_schema={"type": "object"},
            coroutine=call_tool, response_format="content_and_artifact",
        )])
        sessions = MCPSessions()
        try:
            (chart,), _ = await sessions.start(
                "researcher", [MCPServerConfig(name="s", command="srv")]
            )
            message = await chart.ainvoke(_call(chart, {}))
        finally:
            await sessions.aclose()

        assert message.content == [{"type": "text", "text": "A chart:"}, image]

    async def test_the_filter_selects_by_the_servers_names_then_the_prefix_applies(
        self, servers
    ):
        # As ADK: the filter names the server's own tools, the prefix joins
        # with an underscore, and the server is still called by its own name.
        sessions = MCPSessions()
        config = servers(tool_filter=["add", "atlas_lookup"], tool_name_prefix="atlas")
        try:
            (add,), _ = await sessions.start("researcher", [config])
            message = await add.ainvoke(_call(add, {"a": 1, "b": 1}))
        finally:
            await sessions.aclose()

        assert add.name == "atlas_add"
        assert sessions.identify(add) is None  # closed: nothing is identified
        assert message.content == "2"

    async def test_the_prefixed_name_is_the_permission_target(self, servers):
        sessions = MCPSessions()
        try:
            (add, _), _ = await sessions.start(
                "researcher", [servers(tool_name_prefix="atlas")]
            )

            definition = sessions.identify(add)
        finally:
            await sessions.aclose()

        assert (definition.name, definition.capabilities) == (
            "atlas_add", [Capability("mcp", target="atlas_add")]
        )

    @pytest.mark.parametrize("broken, reason", [
        ({"env": {"MCP_TEST_FAIL": "no licence"}}, "McpError: Connection closed"),
        ({"command": "/nonexistent/mcp-server"}, "FileNotFoundError"),
        ({"env": {"MCP_TEST_HANG": "1"}, "timeout": 0.5}, "it did not start within 0.5 s"),
    ], ids=["exits", "no-such-command", "never-answers"])
    async def test_a_server_that_cannot_start_is_a_notice(self, servers, broken, reason):
        sessions = MCPSessions()
        try:
            tools, notices = await sessions.start(
                "researcher", [servers("atlas"), servers("broken", **broken)]
            )
            # The broken server is gone already; the other one runs.
            assert len([pid for pid in servers.pids() if _alive(pid)]) == 1
        finally:
            await sessions.aclose()

        assert [tool.name for tool in tools] == ["add", "lookup"]  # the other one's
        (notice,) = notices
        assert notice.startswith(
            "MCP server 'broken' is unavailable, so agent 'researcher' runs without its tools: "
        )
        assert reason in notice

    @pytest.mark.parametrize("transport", ["sse", "http"])
    async def test_a_remote_server_that_refuses_is_a_notice(self, transport):
        with socket.socket() as probe:  # a local port nothing listens on
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        sessions = MCPSessions()
        try:
            tools, notices = await sessions.start("researcher", [MCPServerConfig(
                name="remote", transport=transport, url=f"http://127.0.0.1:{port}/mcp",
                timeout=10,
            )])
        finally:
            await sessions.aclose()

        assert tools == []
        (notice,) = notices
        assert notice.startswith("MCP server 'remote' is unavailable, so agent 'researcher'")
        assert "ConnectError" in notice

    async def test_closing_from_another_task_is_clean(self, servers, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(mcp_tools, "logger", recorder)
        sessions = MCPSessions()
        tools, _ = await asyncio.create_task(
            sessions.start("researcher", [servers("one"), servers("two")])
        )

        await asyncio.create_task(sessions.aclose())

        # Each owner task left its own session; none had to be cancelled.
        assert recorder.events("debug").count("mcp_session_closed") == 2
        assert recorder.events("warning") == []
        assert [sessions.identify(tool) for tool in tools] == [None] * 4

    async def test_closing_twice_or_before_anything_opened_is_harmless(self, servers):
        sessions = MCPSessions()
        await sessions.aclose()
        await sessions.start("researcher", [servers()])

        await sessions.aclose()
        await sessions.aclose()

    async def test_a_session_that_fails_to_close_never_raises(self, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(mcp_tools, "logger", recorder)
        _fake_sessions(monkeypatch, exit_error=RuntimeError("pipe burst"))
        sessions = MCPSessions()
        await sessions.start("researcher", [MCPServerConfig(name="s", command="srv")])

        await sessions.aclose()

        ((_, fields),) = [(e, f) for lvl, e, f in recorder.calls if lvl == "warning"]
        assert fields["server"] == "s" and "pipe burst" in fields["error"]

    async def test_closing_is_bounded(self, monkeypatch):
        """A session that does not close in time is cancelled, not waited for."""
        recorder = _Recorder()
        monkeypatch.setattr(mcp_tools, "logger", recorder)
        monkeypatch.setattr(mcp_tools, "CLOSE_TIMEOUT", 0.1)
        exits = _fake_sessions(monkeypatch, exit_hangs=True)
        sessions = MCPSessions()
        await sessions.start("researcher", [MCPServerConfig(name="s", command="srv")])

        await asyncio.wait_for(sessions.aclose(), timeout=2)

        assert exits == ["cancelled"]
        assert "mcp_session_close_timed_out" in recorder.events("warning")


def _fake_sessions(
    monkeypatch, *, tools=(), exit_error=None, exit_hangs=False
) -> list[str]:
    """Replace the adapter's session with one that has ``tools`` and leaves as told.

    Returns what happened as it left (``"cancelled"``).
    """
    exits: list[str] = []

    class _Session:
        async def initialize(self) -> None:
            return None

    @asynccontextmanager
    async def create_session(connection):
        yield _Session()
        if exit_hangs:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                exits.append("cancelled")
                raise
        if exit_error is not None:
            raise exit_error

    async def load_mcp_tools(session, **kwargs):
        return list(tools)

    monkeypatch.setattr(mcp_tools, "create_session", create_session)
    monkeypatch.setattr(mcp_tools, "load_mcp_tools", load_mcp_tools)
    return exits


# --- in the manager -------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """A hermetic cwd and HOME, and an in-memory session store."""
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    with MockContext(google_api_key="test-key", session_store="memory") as context:
        yield context


def scripted(monkeypatch, **scripts) -> ScriptedModels:
    models = ScriptedModels(**scripts)
    monkeypatch.setattr(manager_module, "build_chat_model", models)
    return models


def _manager(settings, configs) -> DeepAgentsWorkflowManager:
    return DeepAgentsWorkflowManager(
        agent_configs=configs, settings=settings, model="gemini-2.5-flash"
    )


class Answers:
    """The input callback: answers each prompt in turn, recording them."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.asked: list[UserInputRequest] = []

    async def __call__(self, request: UserInputRequest) -> str:
        self.asked.append(request)
        return self.answers.pop(0)


async def turn(manager, text: str = "Go.") -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


def of(events, kind: EventType) -> list:
    return [event for event in events if event.type is kind]


def _researcher(*servers: MCPServerConfig, **fields) -> AgentConfig:
    fields.setdefault("include_state_tools", False)
    return AgentConfig(name="researcher", prompt="Research.", mcp_servers=list(servers),
                       **fields)


class TestATurn:
    async def test_a_call_is_gated_under_mcp_and_asks_first(self, ctx, servers, monkeypatch):
        models = scripted(monkeypatch, researcher=[
            ai(calls=[("add", {"a": 2, "b": 3}, "c1")]), ai("Five."),
        ])
        manager = _manager(ctx.settings, [_researcher(servers())])
        answers = Answers(ALLOW_ONCE_CHOICE)
        manager.set_input_callback(answers)
        try:
            events = await turn(manager)
        finally:
            await manager.cleanup()

        assert of(events, EventType.ERROR) == []
        assert [request.tool_name for request in answers.asked] == ["add"]
        (result,) = of(events, EventType.TOOL_RESULT)
        assert (result.metadata["tool_name"], result.metadata["result"]) == ("add", "5")
        assert result.metadata["success"] is True
        assert result.metadata["agent"] == "researcher"
        assert models.chats["researcher"].tool_names() == ["add", "lookup"]
        answered = models.chats["researcher"].conversation(1)[-1]
        assert (answered.tool_call_id, answered.text) == ("c1", "5")
        assert [e.content for e in of(events, EventType.TEXT)] == ["Five."]

    async def test_a_denied_call_is_a_failed_result(self, ctx, servers, monkeypatch):
        scripted(monkeypatch, researcher=[
            ai(calls=[("add", {"a": 2, "b": 3}, "c1")]), ai("Could not."),
        ])
        manager = _manager(ctx.settings, [_researcher(servers())])
        manager.set_input_callback(Answers(DENY_CHOICE))
        try:
            events = await turn(manager)
        finally:
            await manager.cleanup()

        (result,) = of(events, EventType.TOOL_RESULT)
        assert result.metadata["success"] is False
        assert result.metadata["error"] == "Permission denied: no rule + user denied"

    async def test_an_allow_rule_skips_the_prompt(self, ctx, servers, monkeypatch):
        user_config = get_user_config_path(ctx.settings.app_name)
        user_config.parent.mkdir(parents=True, exist_ok=True)
        user_config.write_text(json.dumps({
            "permissions": {"allow": [{"capability": "mcp", "target": "atlas_*"}]},
        }))
        scripted(monkeypatch, researcher=[
            ai(calls=[("atlas_add", {"a": 2, "b": 3}, "c1")]), ai("Five."),
        ])
        manager = _manager(ctx.settings, [_researcher(servers(tool_name_prefix="atlas"))])
        answers = Answers()
        manager.set_input_callback(answers)
        try:
            events = await turn(manager)
        finally:
            await manager.cleanup()

        assert answers.asked == []
        (result,) = of(events, EventType.TOOL_RESULT)
        assert (result.metadata["success"], result.metadata["result"]) == (True, "5")

    async def test_an_error_result_reaches_the_model_as_a_failure(
        self, ctx, servers, monkeypatch
    ):
        models = scripted(monkeypatch, researcher=[
            ai(calls=[("lookup", {"city": "Atlantis"}, "c1")]), ai("No such place."),
        ])
        manager = _manager(ctx.settings, [_researcher(servers())])
        manager.set_input_callback(Answers(ALLOW_ONCE_CHOICE))
        try:
            events = await turn(manager)
        finally:
            await manager.cleanup()

        (result,) = of(events, EventType.TOOL_RESULT)
        assert result.metadata["success"] is False
        assert result.metadata["error"] == "No such city: Atlantis"
        answered = models.chats["researcher"].conversation(1)[-1]
        assert answered.status == "error"
        assert json.loads(answered.text) == {
            "success": False, "error": "No such city: Atlantis",
        }

    async def test_a_server_that_went_away_fails_the_call_not_the_turn(
        self, ctx, servers, monkeypatch
    ):
        scripted(monkeypatch, researcher=[
            ai(calls=[("add", {"a": 2, "b": 3}, "c1")]), ai("The server is gone."),
        ])
        manager = _manager(ctx.settings, [_researcher(servers())])
        manager.set_input_callback(Answers(ALLOW_ONCE_CHOICE))
        try:
            await manager.initialize_services()
            (pid,) = servers.pids()
            os.kill(pid, signal.SIGKILL)

            events = await turn(manager)
        finally:
            await manager.cleanup()

        (result,) = of(events, EventType.TOOL_RESULT)
        assert result.metadata["success"] is False
        assert result.metadata["error"].startswith("add failed: ")
        assert [e.content for e in of(events, EventType.TEXT)] == ["The server is gone."]

    async def test_a_sub_agents_server_is_its_own(self, ctx, servers, monkeypatch):
        models = scripted(
            monkeypatch,
            lead=[ai(calls=[task("researcher", "Add 2 and 3.", "t1")]), ai("It is 5.")],
            researcher=[ai(calls=[("add", {"a": 2, "b": 3}, "c1")]), ai("5.")],
        )
        manager = _manager(ctx.settings, [
            AgentConfig(name="lead", prompt="Lead.", sub_agents=["researcher"],
                        include_state_tools=False),
            _researcher(servers(), description="Researches."),
        ])
        manager.set_input_callback(Answers(ALLOW_ONCE_CHOICE))
        try:
            events = await turn(manager)
        finally:
            await manager.cleanup()

        assert models.chats["lead"].tool_names() == ["task"]
        (added,) = [e for e in of(events, EventType.TOOL_RESULT)
                    if e.metadata["tool_name"] == "add"]
        assert (added.metadata["agent"], added.metadata["result"]) == ("researcher", "5")


class TestNotices:
    async def test_an_unreachable_server_leaves_the_agent_without_its_tools(
        self, ctx, servers, monkeypatch
    ):
        models = scripted(monkeypatch, researcher=[ai("Without it."), ai("Still.")])
        manager = _manager(ctx.settings, [_researcher(
            servers("atlas"), servers("broken", env={"MCP_TEST_FAIL": "no licence"}),
        )])
        try:
            first = await turn(manager)
            second = await turn(manager, "Again.")
        finally:
            await manager.cleanup()

        assert models.chats["researcher"].tool_names() == ["add", "lookup"]
        notice = first[0]
        assert notice.type is EventType.ERROR
        assert notice.metadata["error_code"] == "mcp_unavailable"
        assert notice.metadata["recoverable"] is True
        assert notice.metadata["agent"] == "researcher"
        assert notice.content.startswith(
            "MCP server 'broken' is unavailable, so agent 'researcher' runs without its tools: "
        )
        assert [e.content for e in of(first, EventType.TEXT)] == ["Without it."]
        assert of(second, EventType.ERROR) == []

    @pytest.mark.parametrize("rename, configs, taken_by", [
        ("add=read_file", lambda server: [_researcher(server, tools=[read_file])],
         "the agent already has a tool of that name"),
        ("add=task", lambda server: [
            AgentConfig(name="researcher", prompt="Research.", sub_agents=["helper"],
                        mcp_servers=[server], include_state_tools=False),
            AgentConfig(name="helper", prompt="Help.", description="Helps."),
        ], "that name is reserved for delegating to sub-agents"),
    ], ids=["a-framework-tool", "task"])
    async def test_a_tool_whose_name_is_taken_is_skipped(
        self, ctx, servers, monkeypatch, rename, configs, taken_by
    ):
        name = rename.split("=")[1]
        models = scripted(monkeypatch, researcher=[ai("Fine.")], helper=[])
        manager = _manager(ctx.settings, configs(servers(env={"MCP_TEST_RENAME": rename})))
        try:
            events = await turn(manager)
            # The name still runs the agent's own tool, not the server's.
            assert manager._mcp.identify(_tool(manager, name)) is None
        finally:
            await manager.cleanup()

        assert sorted(models.chats["researcher"].tool_names()) == sorted([name, "lookup"])
        (notice,) = of(events, EventType.ERROR)
        assert notice.metadata["error_code"] == "mcp_tool_skipped"
        assert notice.metadata["recoverable"] is True
        assert notice.content == (
            f"MCP tool '{name}' of server 'atlas' is skipped for agent 'researcher': "
            f"{taken_by}. Set tool_name_prefix on the server to rename its tools."
        )

    async def test_two_servers_with_one_tool_name_keep_the_first(
        self, ctx, servers, monkeypatch
    ):
        models = scripted(monkeypatch, researcher=[ai("Fine.")])
        manager = _manager(ctx.settings, [_researcher(servers("atlas"), servers("globe"))])
        try:
            events = await turn(manager)
        finally:
            await manager.cleanup()

        assert models.chats["researcher"].tool_names() == ["add", "lookup"]
        notices = [e.content for e in of(events, EventType.ERROR)]
        assert notices == [
            f"MCP tool '{name}' of server 'globe' is skipped for agent 'researcher': "
            "the agent already has a tool of that name. Set tool_name_prefix on the "
            "server to rename its tools."
            for name in ("add", "lookup")
        ]


def _tool(manager, name: str) -> BaseTool:
    """The tool object the root agent's tool node runs under ``name``."""
    return manager._root.nodes["tools"].bound.tools_by_name[name]


class TestLifecycle:
    async def test_cleanup_closes_every_session(self, ctx, servers, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(mcp_tools, "logger", recorder)
        scripted(monkeypatch, researcher=[])
        manager = _manager(ctx.settings, [_researcher(
            servers("one", tool_name_prefix="one"), servers("two", tool_name_prefix="two"),
        )])
        await manager.initialize_services()
        add = _tool(manager, "one_add")
        assert [_alive(pid) for pid in servers.pids()] == [True, True]

        await manager.cleanup()
        await manager.cleanup()

        assert [_alive(pid) for pid in servers.pids()] == [False, False]
        # Each session was told to stop and left; none had to be cancelled.
        assert recorder.events("debug").count("mcp_session_closed") == 2
        assert recorder.events("warning") == []
        assert manager._identify_tool(add) is None

    async def test_a_failed_initialization_closes_the_sessions(
        self, ctx, servers, monkeypatch
    ):
        scripted(monkeypatch, researcher=[])

        def broken_tree(*args, **kwargs):
            raise RuntimeError("tree broke")

        monkeypatch.setattr(manager_module, "build_agent_tree", broken_tree)
        manager = _manager(ctx.settings, [_researcher(
            servers(), servers("broken", env={"MCP_TEST_FAIL": "no licence"}),
        )])

        with pytest.raises(RuntimeError, match="tree broke"):
            await manager.initialize_services()

        assert [_alive(pid) for pid in servers.pids()] == [False, False]
        # The notice was about sessions that no longer exist.
        assert manager._pending_notices == []

    async def test_a_model_too_small_for_its_budget_starts_nothing(
        self, ctx, servers, monkeypatch
    ):
        """Each agent's budget is computed before any MCP server starts: a
        model whose window leaves no room to work in fails initialization
        with no server started and the checkpointer's connection closed."""
        opened: list = []
        real_open = manager_module.open_checkpointer

        async def open_checkpointer(*args, **kwargs):
            saver, connection = await real_open(*args, **kwargs)
            opened.append(connection)
            return saver, connection

        monkeypatch.setattr(manager_module, "open_checkpointer", open_checkpointer)
        settings = BaseSettings(
            session_store="sqlite", default_model="local:small",
            model_endpoints={"local": {"base_url": "http://localhost:8000/v1",
                                       "models": {"small": {"context_window": 4_000}}}},
        )
        manager = DeepAgentsWorkflowManager(agent_configs=[_researcher(servers())],
                                            settings=settings)
        closed: list = []
        real_close = manager._close_connection

        async def close_connection(connection):
            closed.append(connection)
            await real_close(connection)

        monkeypatch.setattr(manager, "_close_connection", close_connection)

        with pytest.raises(ContextBudgetError, match="local:small"):
            await manager.initialize_services()
        try:
            assert servers.pids() == []  # no MCP server was ever started
            assert len(opened) == 1 and opened[0] is not None
            assert closed == opened  # the sqlite connection, closed by the rollback
            assert (manager._checkpointer, manager._sessions, manager._budgets) == (
                None, None, {}
            )
            with pytest.raises(RuntimeError, match="not initialized"):
                manager.context_budget()
        finally:
            await manager.cleanup()

    async def test_a_model_switch_reopens_the_sessions(self, ctx, servers, monkeypatch):
        scripted(monkeypatch, researcher=[
            ai(calls=[("add", {"a": 2, "b": 3}, "c1")]), ai("Five."),
        ])
        manager = _manager(ctx.settings, [_researcher(servers())])
        manager.set_input_callback(Answers(ALLOW_ONCE_CHOICE))
        try:
            await manager.initialize_services()
            before = _tool(manager, "add")
            (first,) = servers.pids()

            await manager.reinitialize(model="gemini-2.5-pro")

            second = [pid for pid in servers.pids() if pid != first]
            after = _tool(manager, "add")
            events = await turn(manager)
        finally:
            await manager.cleanup()

        assert not _alive(first) and len(second) == 1
        assert manager._identify_tool(before) is None
        assert after is not before
        (result,) = of(events, EventType.TOOL_RESULT)
        assert (result.metadata["success"], result.metadata["result"]) == (True, "5")

    async def test_a_failed_switch_closes_the_sessions_and_keeps_the_checkpointer(
        self, ctx, servers, monkeypatch
    ):
        scripted(monkeypatch, researcher=[])
        manager = _manager(ctx.settings, [_researcher(servers())])
        await manager.initialize_services()
        saver = manager._checkpointer

        def broken_tree(*args, **kwargs):
            raise RuntimeError("tree broke")

        monkeypatch.setattr(manager_module, "build_agent_tree", broken_tree)
        try:
            with pytest.raises(RuntimeError, match="tree broke"):
                await manager.reinitialize(model="gemini-2.5-pro")

            assert [_alive(pid) for pid in servers.pids()] == [False, False]
            assert manager._checkpointer is saver
        finally:
            await manager.cleanup()
