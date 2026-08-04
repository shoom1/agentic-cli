"""Backend-neutral MCP (Model Context Protocol) server configuration.

``MCPServerConfig`` describes how to reach an MCP server (stdio subprocess or
remote SSE/HTTP). ``to_adk_toolset`` materializes it into an ADK ``MCPToolset``
that can be appended to an agent's ``tools`` list; ADK connects lazily on first
use, so construction is synchronous.

LangGraph materialization (via ``langchain-mcp-adapters``) is deferred — see the
implementation plan.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MCPServerConfig(BaseModel):
    """Connection config for a single MCP server.

    Attributes:
        name: Logical name for the server (used for logging / permission targets).
        transport: ``stdio`` (local subprocess), ``sse``, or ``http``
            (streamable HTTP).
        command/args/env: Used by ``stdio`` — the executable, its arguments, and
            extra environment variables.
        url/headers: Used by ``sse``/``http`` — the server URL and request headers.
        tool_filter: Optional allowlist of MCP tool names to expose.
        tool_name_prefix: Optional prefix added to each MCP tool name.
        timeout: Optional connection timeout (seconds).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    transport: Literal["stdio", "sse", "http"] = "stdio"

    # stdio
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    # sse / http
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    tool_filter: list[str] | None = None
    tool_name_prefix: str | None = None
    timeout: float | None = None

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> "MCPServerConfig":
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("stdio transport requires 'command'")
        else:  # sse / http
            if not self.url:
                raise ValueError(f"{self.transport} transport requires 'url'")
        return self


def build_connection_params(cfg: MCPServerConfig) -> Any:
    """Build the ADK connection-params object for an MCP server config.

    Returns one of ``StdioConnectionParams`` / ``SseConnectionParams`` /
    ``StreamableHTTPConnectionParams`` depending on transport.
    """
    from google.adk.tools.mcp_tool import (
        SseConnectionParams,
        StdioConnectionParams,
        StreamableHTTPConnectionParams,
    )

    if cfg.transport == "stdio":
        from mcp import StdioServerParameters

        server = StdioServerParameters(
            command=cfg.command,
            args=list(cfg.args),
            env=dict(cfg.env) if cfg.env else None,
        )
        kwargs: dict[str, Any] = {"server_params": server}
        if cfg.timeout is not None:
            kwargs["timeout"] = cfg.timeout
        return StdioConnectionParams(**kwargs)

    kwargs = {"url": cfg.url, "headers": dict(cfg.headers) if cfg.headers else None}
    if cfg.timeout is not None:
        kwargs["timeout"] = cfg.timeout
    if cfg.transport == "sse":
        return SseConnectionParams(**kwargs)
    return StreamableHTTPConnectionParams(**kwargs)


def to_adk_toolset(cfg: MCPServerConfig) -> Any:
    """Materialize an MCPServerConfig into an ADK ``McpToolset``."""
    try:
        from google.adk.tools.mcp_tool import McpToolset as _McpToolset
    except ImportError:  # older ADK only has the (now-deprecated) MCPToolset
        from google.adk.tools.mcp_tool import MCPToolset as _McpToolset

    kwargs: dict[str, Any] = {"connection_params": build_connection_params(cfg)}
    if cfg.tool_filter is not None:
        kwargs["tool_filter"] = list(cfg.tool_filter)
    if cfg.tool_name_prefix:
        kwargs["tool_name_prefix"] = cfg.tool_name_prefix
    return _McpToolset(**kwargs)
