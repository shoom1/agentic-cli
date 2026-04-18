# tests/integration/test_permission_adk.py  (unit-level for Task 20)
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_cli.workflow.permissions import Capability, EXEMPT


@pytest.fixture
def stub_engine():
    from agentic_cli.workflow.permissions.rules import CheckResult
    engine = MagicMock()
    engine.check = AsyncMock(return_value=CheckResult(True, "rule: test/allow"))
    return engine


class TestPermissionPluginUnit:
    @pytest.mark.asyncio
    async def test_exempt_tool_passes_through(self, monkeypatch, stub_engine):
        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin

        monkeypatch.setattr(
            "agentic_cli.workflow.adk.permission_plugin.get_service",
            lambda k: stub_engine if k == "permission_engine" else None,
        )
        reg = get_registry()

        @reg.register(name="exempt_x", capabilities=EXEMPT)
        def exempt_x():
            return {}

        plugin = PermissionPlugin()
        result = await plugin.before_tool_callback(
            tool=SimpleNamespace(name="exempt_x"), tool_args={}, tool_context=None,
        )
        assert result is None
        stub_engine.check.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_declaration_denies(self, monkeypatch, stub_engine):
        from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin

        monkeypatch.setattr(
            "agentic_cli.workflow.adk.permission_plugin.get_service",
            lambda k: stub_engine if k == "permission_engine" else None,
        )
        plugin = PermissionPlugin()
        result = await plugin.before_tool_callback(
            tool=SimpleNamespace(name="never_registered"),
            tool_args={},
            tool_context=None,
        )
        assert result == {
            "success": False,
            "error": "Permission denied: tool has no capability declaration",
        }

    @pytest.mark.asyncio
    async def test_allow_calls_engine_and_passes(self, monkeypatch, stub_engine):
        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin

        monkeypatch.setattr(
            "agentic_cli.workflow.adk.permission_plugin.get_service",
            lambda k: stub_engine if k == "permission_engine" else None,
        )
        reg = get_registry()

        @reg.register(
            name="reader_x",
            capabilities=[Capability("filesystem.read", target_arg="path")],
        )
        def reader_x(path: str):
            return {}

        plugin = PermissionPlugin()
        result = await plugin.before_tool_callback(
            tool=SimpleNamespace(name="reader_x"),
            tool_args={"path": "/tmp/x"},
            tool_context=None,
        )
        assert result is None
        stub_engine.check.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_deny_returns_error_dict(self, monkeypatch):
        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin
        from agentic_cli.workflow.permissions.rules import CheckResult

        denying_engine = MagicMock()
        denying_engine.check = AsyncMock(return_value=CheckResult(False, "rule: builtin/deny"))
        monkeypatch.setattr(
            "agentic_cli.workflow.adk.permission_plugin.get_service",
            lambda k: denying_engine if k == "permission_engine" else None,
        )
        reg = get_registry()

        @reg.register(
            name="writer_x",
            capabilities=[Capability("filesystem.write", target_arg="path")],
        )
        def writer_x(path: str):
            return {}

        plugin = PermissionPlugin()
        result = await plugin.before_tool_callback(
            tool=SimpleNamespace(name="writer_x"),
            tool_args={"path": "/etc/x"},
            tool_context=None,
        )
        assert result == {"success": False, "error": "Permission denied: rule: builtin/deny"}

    @pytest.mark.asyncio
    async def test_engine_absent_allows(self, monkeypatch):
        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin

        monkeypatch.setattr(
            "agentic_cli.workflow.adk.permission_plugin.get_service",
            lambda k: None,
        )
        reg = get_registry()

        @reg.register(
            name="reader_y",
            capabilities=[Capability("filesystem.read", target_arg="path")],
        )
        def reader_y(path: str):
            return {}

        plugin = PermissionPlugin()
        result = await plugin.before_tool_callback(
            tool=SimpleNamespace(name="reader_y"),
            tool_args={"path": "/tmp/x"},
            tool_context=None,
        )
        assert result is None  # fallback allow
