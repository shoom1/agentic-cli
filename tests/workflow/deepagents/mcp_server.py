"""A small stdio MCP server for the Deep Agents MCP tests.

Run as a script with the test interpreter (``sys.executable``); it speaks MCP
over stdin/stdout through ``mcp.server.fastmcp``. Two tools:

- ``add(a, b)`` returns the sum as text;
- ``lookup(city)`` always answers with an MCP error result (``isError``).

The environment steers it:

- ``MCP_TEST_PID_DIR``: the server writes a file named after its process id
  there on start, so a test can check that no server outlives it;
- ``MCP_TEST_RENAME``: ``add=read_file,lookup=task`` publishes the tools
  under other names (to clash with a framework tool or ``task``);
- ``MCP_TEST_FAIL``: the server exits before it speaks MCP, with this text
  on stderr (a server that cannot start);
- ``MCP_TEST_HANG``: the server never answers, and exits when its stdin
  closes (a server that does not start in time).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent


def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


def lookup(city: str) -> CallToolResult:
    """Look a city up in the atlas."""
    return CallToolResult(
        isError=True,
        content=[TextContent(type="text", text=f"No such city: {city}")],
    )


def _names() -> dict[str, str]:
    renames = dict(
        pair.split("=", 1)
        for pair in os.environ.get("MCP_TEST_RENAME", "").split(",")
        if "=" in pair
    )
    return {tool: renames.get(tool, tool) for tool in ("add", "lookup")}


def main() -> None:
    pid_dir = os.environ.get("MCP_TEST_PID_DIR")
    if pid_dir:
        (Path(pid_dir) / str(os.getpid())).touch()
    failure = os.environ.get("MCP_TEST_FAIL")
    if failure:
        sys.stderr.write(failure + "\n")
        sys.exit(3)
    if os.environ.get("MCP_TEST_HANG"):
        sys.stdin.read()  # until the client closes the pipe
        return

    server = FastMCP("agentic-cli-test", log_level="WARNING")
    names = _names()
    server.tool(name=names["add"])(add)
    server.tool(name=names["lookup"])(lookup)
    server.run()  # stdio


if __name__ == "__main__":
    main()
