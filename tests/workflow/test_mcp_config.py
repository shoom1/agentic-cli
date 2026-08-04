"""Tests for MCP server config + ADK materialization (Phase 4)."""

from __future__ import annotations

import textwrap

import pytest

pytest.importorskip("google.adk")

from pydantic import ValidationError  # noqa: E402

from agentic_cli.workflow.agent_loader import load_agents_from_yaml  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.mcp import (  # noqa: E402
    MCPServerConfig,
    build_connection_params,
    to_adk_toolset,
)


class TestMCPServerConfigValidation:
    def test_stdio_requires_command(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(name="x", transport="stdio")

    def test_sse_requires_url(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(name="x", transport="sse")

    def test_http_requires_url(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(name="x", transport="http")

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(name="x", command="npx", bogus=1)

    def test_stdio_valid(self):
        cfg = MCPServerConfig(name="x", command="npx")
        assert cfg.transport == "stdio"


class TestBuildConnectionParams:
    def test_stdio(self):
        from google.adk.tools.mcp_tool import StdioConnectionParams

        cfg = MCPServerConfig(
            name="notion", transport="stdio", command="npx",
            args=["-y", "srv"], env={"A": "B"},
        )
        conn = build_connection_params(cfg)
        assert isinstance(conn, StdioConnectionParams)
        assert conn.server_params.command == "npx"
        assert conn.server_params.args == ["-y", "srv"]
        assert conn.server_params.env == {"A": "B"}

    def test_sse(self):
        from google.adk.tools.mcp_tool import SseConnectionParams

        cfg = MCPServerConfig(
            name="s", transport="sse", url="https://x/sse", headers={"H": "V"}
        )
        conn = build_connection_params(cfg)
        assert isinstance(conn, SseConnectionParams)
        assert conn.url == "https://x/sse"
        assert conn.headers == {"H": "V"}

    def test_http(self):
        from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams

        cfg = MCPServerConfig(name="s", transport="http", url="https://x/mcp")
        conn = build_connection_params(cfg)
        assert isinstance(conn, StreamableHTTPConnectionParams)
        assert conn.url == "https://x/mcp"


class TestToAdkToolset:
    def test_returns_mcptoolset(self):
        from google.adk.tools.mcp_tool import McpToolset

        ts = to_adk_toolset(MCPServerConfig(name="s", command="echo"))
        assert isinstance(ts, McpToolset)

    def test_with_tool_filter(self):
        from google.adk.tools.mcp_tool import McpToolset

        ts = to_adk_toolset(
            MCPServerConfig(name="s", command="echo", tool_filter=["a", "b"])
        )
        assert isinstance(ts, McpToolset)


class TestAgentLoaderMCP:
    def test_yaml_mcp_servers_parsed(self, tmp_path):
        path = tmp_path / "a.yaml"
        path.write_text(
            textwrap.dedent(
                """
                agents:
                  - name: a
                    instruction: hi
                    mcp_servers:
                      - name: notion
                        transport: stdio
                        command: npx
                        args: ["-y", "notion-mcp"]
                """
            ),
            encoding="utf-8",
        )
        cfg = load_agents_from_yaml(path)[0]
        assert len(cfg.mcp_servers) == 1
        assert cfg.mcp_servers[0].name == "notion"
        assert cfg.mcp_servers[0].command == "npx"


class TestADKManagerAttachesMCP:
    def test_agent_gets_mcp_toolset(self, mock_context):
        from google.adk.tools.mcp_tool import McpToolset

        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        cfg = AgentConfig(
            name="a", prompt="p", tools=[], include_state_tools=False,
            mcp_servers=[MCPServerConfig(name="s", command="echo")],
        )
        mgr = GoogleADKWorkflowManager(
            agent_configs=[cfg], settings=mock_context.settings, model="gemini-2.5-flash"
        )
        tools = mgr._assemble_agent_tools(cfg, mgr._get_service_tool_map())
        assert any(isinstance(t, McpToolset) for t in tools)
