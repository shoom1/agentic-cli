# tests/integration/test_permission_langgraph.py
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_cli.workflow.permissions import Capability, EXEMPT
from agentic_cli.workflow.service_registry import PERMISSION_ENGINE


class TestWrapToolForPermissionUnit:
    def test_exempt_tool_returned_unmodified(self):
        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.langgraph.permission_wrap import wrap_tool_for_permission

        reg = get_registry()

        @reg.register(name="ex_lg", capabilities=EXEMPT)
        def ex_lg():
            return {"success": True}

        wrapped = wrap_tool_for_permission(ex_lg)
        assert wrapped is ex_lg

    @pytest.mark.asyncio
    async def test_missing_declaration_denies(self, monkeypatch):
        from agentic_cli.workflow.langgraph.permission_wrap import wrap_tool_for_permission

        def unregistered(path: str):
            return {"ok": True}

        wrapped = wrap_tool_for_permission(unregistered)
        result = await wrapped(path="/x")
        assert result == {
            "success": False,
            "error": "Permission denied: tool has no capability declaration",
        }

    @pytest.mark.asyncio
    async def test_engine_allow_runs_tool(self, monkeypatch):
        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.langgraph.permission_wrap import wrap_tool_for_permission
        from agentic_cli.workflow.permissions.rules import CheckResult

        engine = MagicMock()
        engine.check = AsyncMock(return_value=CheckResult(True, "rule: test/allow"))
        monkeypatch.setattr(
            "agentic_cli.workflow.langgraph.permission_wrap.get_service",
            lambda k: engine if k == PERMISSION_ENGINE else None,
        )

        reg = get_registry()

        @reg.register(
            name="read_lg",
            capabilities=[Capability("filesystem.read", target_arg="path")],
        )
        def read_lg(path: str):
            return {"content": f"read:{path}"}

        wrapped = wrap_tool_for_permission(read_lg)
        result = await wrapped(path="/abs/x")
        assert result == {"content": "read:/abs/x"}

    @pytest.mark.asyncio
    async def test_engine_deny_returns_error_dict(self, monkeypatch):
        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.langgraph.permission_wrap import wrap_tool_for_permission
        from agentic_cli.workflow.permissions.rules import CheckResult

        engine = MagicMock()
        engine.check = AsyncMock(return_value=CheckResult(False, "rule: builtin/deny"))
        monkeypatch.setattr(
            "agentic_cli.workflow.langgraph.permission_wrap.get_service",
            lambda k: engine if k == PERMISSION_ENGINE else None,
        )

        reg = get_registry()

        @reg.register(
            name="write_lg",
            capabilities=[Capability("filesystem.write", target_arg="path")],
        )
        def write_lg(path: str):
            return {"ok": True}

        wrapped = wrap_tool_for_permission(write_lg)
        result = await wrapped(path="/etc/x")
        assert result == {"success": False, "error": "Permission denied: rule: builtin/deny"}

    @pytest.mark.asyncio
    async def test_engine_absent_allows(self, monkeypatch):
        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.langgraph.permission_wrap import wrap_tool_for_permission

        monkeypatch.setattr(
            "agentic_cli.workflow.langgraph.permission_wrap.get_service",
            lambda k: None,
        )
        reg = get_registry()

        @reg.register(
            name="read_lg2",
            capabilities=[Capability("filesystem.read", target_arg="path")],
        )
        def read_lg2(path: str):
            return {"ok": True}

        wrapped = wrap_tool_for_permission(read_lg2)
        result = await wrapped(path="/x")
        assert result == {"ok": True}
