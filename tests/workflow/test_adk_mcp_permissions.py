"""MCP tools are gated through the permission engine, not hard-denied (Phase 4)."""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin  # noqa: E402
from agentic_cli.workflow.permissions import PermissionEngine  # noqa: E402
from agentic_cli.workflow.permissions.prompt import ALLOW_ONCE_CHOICE  # noqa: E402
from agentic_cli.workflow.permissions.rules import (  # noqa: E402
    Effect,
    Rule,
    RuleSource,
)
from agentic_cli.workflow.permissions.store import PermissionContext  # noqa: E402
from agentic_cli.workflow.service_registry import (  # noqa: E402
    PERMISSION_ENGINE,
    set_service_registry,
)


class _MCPTool:
    """Stand-in for an ADK MCP tool (detected by class name)."""

    def __init__(self, name: str):
        self.name = name


_MCPTool.__name__ = "MCPTool"


class _PlainTool:
    def __init__(self, name: str):
        self.name = name


class _StubWorkflow:
    def __init__(self, response: str):
        self._response = response

    async def request_user_input(self, request):
        return self._response


def _engine(tmp_path, response="deny", rules=None) -> PermissionEngine:
    settings = BaseSettings(google_api_key="test")
    ctx = PermissionContext(workdir=tmp_path, home=tmp_path)
    eng = PermissionEngine(
        settings=settings, workflow=_StubWorkflow(response), ctx=ctx
    )
    if rules:
        eng._session_rules.extend(rules)
    return eng


async def _check(engine, tool):
    token = set_service_registry({PERMISSION_ENGINE: engine})
    try:
        return await PermissionPlugin().before_tool_callback(
            tool=tool, tool_args={}, tool_context=None
        )
    finally:
        token.var.reset(token)


class TestMCPPermissions:
    async def test_allow_rule_allows(self, tmp_path):
        eng = _engine(
            tmp_path,
            rules=[Rule("mcp", "*", Effect.ALLOW, RuleSource.SESSION)],
        )
        assert await _check(eng, _MCPTool("notion_search")) is None

    async def test_deny_rule_denies(self, tmp_path):
        eng = _engine(
            tmp_path,
            rules=[Rule("mcp", "*", Effect.DENY, RuleSource.SESSION)],
        )
        res = await _check(eng, _MCPTool("notion_search"))
        assert res is not None and res["success"] is False

    async def test_no_rule_asks_user_and_allows(self, tmp_path):
        eng = _engine(tmp_path, response=ALLOW_ONCE_CHOICE)
        assert await _check(eng, _MCPTool("notion_search")) is None

    async def test_no_rule_asks_user_and_denies(self, tmp_path):
        eng = _engine(tmp_path, response="deny")  # unknown choice -> DENY
        res = await _check(eng, _MCPTool("notion_search"))
        assert res is not None and res["success"] is False

    async def test_engine_absent_allows(self, tmp_path):
        # No engine in the registry -> test/dev fallback allows.
        token = set_service_registry({})
        try:
            res = await PermissionPlugin().before_tool_callback(
                tool=_MCPTool("notion_search"), tool_args={}, tool_context=None
            )
        finally:
            token.var.reset(token)
        assert res is None

    async def test_non_mcp_unknown_tool_still_denied(self, tmp_path):
        eng = _engine(tmp_path)
        res = await _check(eng, _PlainTool("random_tool"))
        assert res is not None and res["success"] is False
