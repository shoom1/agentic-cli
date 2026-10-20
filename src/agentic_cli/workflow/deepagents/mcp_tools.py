"""MCP servers' tools for Deep Agents agents (``AgentConfig.mcp_servers``).

Each (agent, server) pair gets one persistent session, as ADK gives each agent
one toolset per server. The session is opened through ``langchain-mcp-adapters``
(``workflow/mcp.py::to_langchain_connection``) by a task of its own, which
loads the server's tools, reports them, and holds the session until it is
told to stop: MCP's anyio contexts must be left by the task that entered
them, so nothing else ever closes a session, and closing works from any task.

What a server's tools become, as ADK has them:

- ``tool_filter`` names the server's own tools; ``tool_name_prefix`` then
  renames each to ``<prefix>_<name>``, and the server is still called by its
  own name.
- Each tool object is bound, in this manager's own map (never the framework
  registry), to a definition gated under the synthetic ``mcp`` capability with
  the tool's name as the target: ASK when no rule matches. The framework built
  the object, so it binds it; nothing is resolved by name or class.
- A text result is the tool's text; images and files pass through as content
  blocks; an MCP error result is ``{"success": false, "error": <its text>}``.

A server that cannot start leaves its agent without its tools, and a tool
whose name the agent already uses, or ``task``, is skipped. Neither stops the
app: each is a notice the manager opens the next turn with.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import load_mcp_tools

from agentic_cli.logging import Loggers
from agentic_cli.tools._core.delegation import TASK_TOOL
from agentic_cli.tools.registry import ToolDefinition, ToolRegistry
from agentic_cli.workflow.mcp import MCPServerConfig, to_langchain_connection
from agentic_cli.workflow.permissions.capabilities import Capability

logger = Loggers.workflow()

#: Seconds a server has to start (connect, initialize, list its tools) when its
#: config sets no ``timeout``: ADK's connection timeout.
STARTUP_TIMEOUT = 5.0

#: Seconds closing waits for the sessions to be left, and again for those it
#: then cancels.
CLOSE_TIMEOUT = 5.0

#: The ``BaseTool.metadata`` key naming the server a tool came from.
SERVER_KEY = "mcp_server"

# What a server's start ended in: its tools, or why it has none.
_Started = list[BaseTool] | str


@dataclass
class _Owner:
    """The task holding one (agent, server) session, and how to reach it."""

    agent: str
    server: str
    ready: "asyncio.Future[_Started]"
    stop: asyncio.Event
    task: "asyncio.Task[None] | None" = None

    @property
    def open(self) -> bool:
        return self.ready.done() and isinstance(self.ready.result(), list)


class MCPSessions:
    """One manager's MCP sessions, and the identities of the tools they loaded.

    ``start`` opens an agent's servers, ``aclose`` closes every session (from
    any task, bounded, never raising), and ``identify`` says which tool
    objects are the open sessions' tools. After ``aclose`` the same instance
    can ``start`` again (a model switch reopens the sessions).
    """

    def __init__(self) -> None:
        self._owners: list[_Owner] = []
        # This manager's bindings, by object identity (weak, confirmed with
        # `is`): the framework registry never learns an MCP tool.
        self._identities = ToolRegistry()

    async def start(
        self, agent: str, servers: Sequence[MCPServerConfig]
    ) -> tuple[list[BaseTool], list[str]]:
        """Open ``agent``'s servers, at once, and load their tools.

        Args:
            agent: The agent the sessions are for.
            servers: Its servers, in order.

        Returns:
            The tools, server by server, filtered, prefixed and bound; and a
            notice for each server that could not start (none started within
            its ``timeout``, or ``STARTUP_TIMEOUT``).
        """
        owners = [self._open(agent, server) for server in servers]
        outcomes = await asyncio.gather(
            *(self._started(owner, server) for owner, server in zip(owners, servers))
        )
        tools: list[BaseTool] = []
        notices: list[str] = []
        for server, outcome in zip(servers, outcomes):
            if isinstance(outcome, str):
                logger.warning(
                    "mcp_server_unavailable",
                    agent=agent,
                    server=server.name,
                    reason=outcome,
                )
                notices.append(
                    f"MCP server '{server.name}' is unavailable, so agent '{agent}' "
                    f"runs without its tools: {outcome}"
                )
                continue
            loaded = [self._ours(server, tool) for tool in _selected(server, outcome)]
            logger.info(
                "mcp_tools_loaded",
                agent=agent,
                server=server.name,
                tools=[tool.name for tool in loaded],
            )
            tools += loaded
        return tools, notices

    def identify(self, obj: object) -> ToolDefinition | None:
        """The definition an open session's tool object is bound to, or None."""
        return self._identities.identify(obj)

    async def aclose(self) -> None:
        """Close every session. Idempotent, bounded, never raises.

        Each owner is told to stop and leaves its session itself; one still
        starting is cancelled. Whatever has not finished after
        ``CLOSE_TIMEOUT`` is cancelled and waited for once more.
        """
        owners, self._owners = self._owners, []
        # Their tools stop being tools at once, whatever closing takes.
        self._identities = ToolRegistry()
        if not owners:
            return
        for owner in owners:
            if owner.open:
                owner.stop.set()
            else:
                owner.task.cancel()
        try:
            tasks = {owner.task for owner in owners}
            _, pending = await asyncio.wait(tasks, timeout=CLOSE_TIMEOUT)
            for owner in owners:
                if owner.task in pending:
                    logger.warning(
                        "mcp_session_close_timed_out",
                        agent=owner.agent,
                        server=owner.server,
                    )
                    owner.task.cancel()
            if pending:
                await asyncio.wait(pending, timeout=CLOSE_TIMEOUT)
        except Exception as error:  # noqa: BLE001 - shutdown must not fail
            logger.warning("mcp_sessions_close_failed", error=str(error))

    # -- the owner tasks -------------------------------------------------------

    def _open(self, agent: str, server: MCPServerConfig) -> _Owner:
        """Start the task that will hold ``server``'s session for ``agent``."""
        owner = _Owner(
            agent=agent,
            server=server.name,
            ready=asyncio.get_running_loop().create_future(),
            stop=asyncio.Event(),
        )
        owner.task = asyncio.create_task(
            self._own(owner, server), name=f"mcp:{agent}:{server.name}"
        )
        # Tracked before anything is awaited: a cancelled start still closes.
        self._owners.append(owner)
        return owner

    @staticmethod
    async def _own(owner: _Owner, server: MCPServerConfig) -> None:
        """Enter the session, report its tools, hold it until told to stop."""
        try:
            async with create_session(to_langchain_connection(server)) as session:
                await session.initialize()
                tools = await load_mcp_tools(session, server_name=server.name)
                owner.ready.set_result(tools)
                await owner.stop.wait()
        except asyncio.CancelledError:
            if not owner.ready.done():
                owner.ready.set_result("its start was cancelled")
            raise
        except Exception as error:  # noqa: BLE001 - reported, never raised
            if not owner.ready.done():
                owner.ready.set_result(_reason(error))
            else:
                logger.warning(
                    # Told to stop, leaving failed; otherwise the server went away.
                    "mcp_session_close_failed"
                    if owner.stop.is_set()
                    else "mcp_session_lost",
                    agent=owner.agent,
                    server=owner.server,
                    error=_reason(error),
                )
            return
        logger.debug("mcp_session_closed", agent=owner.agent, server=owner.server)

    @staticmethod
    async def _started(owner: _Owner, server: MCPServerConfig) -> _Started:
        """``owner``'s tools, or why it has none, within the server's timeout."""
        timeout = server.timeout if server.timeout is not None else STARTUP_TIMEOUT
        try:
            return await asyncio.wait_for(asyncio.shield(owner.ready), timeout)
        except TimeoutError:
            # Still starting: stop it now, so its server is gone when start returns.
            owner.task.cancel()
            await asyncio.wait({owner.task}, timeout=CLOSE_TIMEOUT)
            return f"it did not start within {timeout:g} s"

    # -- the tools -------------------------------------------------------------

    def _ours(self, server: MCPServerConfig, loaded: BaseTool) -> BaseTool:
        """``loaded`` under its final name, with our results, bound under ``mcp``."""
        prefix = server.tool_name_prefix
        name = f"{prefix}_{loaded.name}" if prefix else loaded.name
        tool = StructuredTool(
            name=name,
            description=loaded.description,
            args_schema=loaded.args_schema,
            coroutine=_text_results(loaded.coroutine),
            response_format="content_and_artifact",
            metadata={**(loaded.metadata or {}), SERVER_KEY: server.name},
            handle_tool_error=_error_result,
        )
        self._identities.bind_identity(
            tool,
            ToolDefinition(
                name=name,
                description=tool.description,
                func=None,
                capabilities=[Capability("mcp", target=name)],
            ),
        )
        return tool


def without_clashes(
    agent: str, tools: Sequence[BaseTool], mcp_tools: Sequence[BaseTool]
) -> tuple[list[BaseTool], list[str]]:
    """The MCP tools whose names are free on ``agent``, and a notice per one skipped.

    A name is taken by one of the agent's own ``tools``, by an MCP tool kept
    before it (servers in order), and always by ``task``, the delegation tool.

    Returns:
        The MCP tools to add, and the notices.
    """
    taken = {tool.name for tool in tools}
    kept: list[BaseTool] = []
    notices: list[str] = []
    for tool in mcp_tools:
        if tool.name == TASK_TOOL:
            why = "that name is reserved for delegating to sub-agents"
        elif tool.name in taken:
            why = "the agent already has a tool of that name"
        else:
            kept.append(tool)
            taken.add(tool.name)
            continue
        server = (tool.metadata or {}).get(SERVER_KEY)
        logger.warning("mcp_tool_skipped", agent=agent, server=server, tool=tool.name)
        notices.append(
            f"MCP tool '{tool.name}' of server '{server}' is skipped for agent "
            f"'{agent}': {why}. Set tool_name_prefix on the server to rename its tools."
        )
    return kept, notices


def _selected(server: MCPServerConfig, tools: list[BaseTool]) -> list[BaseTool]:
    """The tools ``tool_filter`` selects by the server's names (all, without one)."""
    if not server.tool_filter:
        return tools
    return [tool for tool in tools if tool.name in server.tool_filter]


def _text_results(
    call: Callable[..., Awaitable[tuple[Any, Any]]],
) -> Callable[..., Awaitable[tuple[Any, Any]]]:
    """The adapter's call, with a result of text blocks only as its text."""

    async def run(**arguments: Any) -> tuple[Any, Any]:
        content, artifact = await call(**arguments)
        if isinstance(content, list) and all(
            isinstance(block, dict) and block.get("type") == "text" for block in content
        ):
            content = "\n".join(block["text"] for block in content)
        return content, artifact

    return run


def _error_result(error: ToolException) -> str:
    """An MCP error result (``isError``), as the framework's failed result."""
    return json.dumps({"success": False, "error": str(error)})


def _reason(error: BaseException) -> str:
    """Why a session failed, from its innermost error (anyio wraps them in groups)."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    text = str(error)
    return f"{type(error).__name__}: {text}" if text else type(error).__name__
