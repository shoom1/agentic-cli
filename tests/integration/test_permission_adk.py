# tests/integration/test_permission_adk.py  (unit-level for Task 20)
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_cli.workflow.permissions import Capability, EXEMPT
from agentic_cli.workflow.service_registry import PERMISSION_ENGINE


def _adk_tool(func):
    """Wrap a registered callable the way ADK does before dispatching it.

    The plugin resolves capabilities by *identity*, never by ``tool.name``, so
    a name-only stand-in is (correctly) denied as unregistered — see
    ``tests/workflow/test_permission_tool_identity.py``.
    """
    from google.adk.tools import FunctionTool

    return FunctionTool(func=func)


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
            lambda k: stub_engine if k == PERMISSION_ENGINE else None,
        )
        reg = get_registry()

        @reg.register(name="exempt_x", capabilities=EXEMPT)
        def exempt_x():
            return {}

        plugin = PermissionPlugin()
        result = await plugin.before_tool_callback(
            tool=_adk_tool(exempt_x), tool_args={}, tool_context=None,
        )
        assert result is None
        stub_engine.check.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_declaration_denies(self, monkeypatch, stub_engine):
        from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin

        monkeypatch.setattr(
            "agentic_cli.workflow.adk.permission_plugin.get_service",
            lambda k: stub_engine if k == PERMISSION_ENGINE else None,
        )
        def never_registered():
            """Never passed through @register_tool."""
            return {}

        plugin = PermissionPlugin()
        result = await plugin.before_tool_callback(
            tool=_adk_tool(never_registered),
            tool_args={},
            tool_context=None,
        )
        assert result is not None and result["success"] is False
        assert "not registered" in result["error"]

    @pytest.mark.asyncio
    async def test_allow_calls_engine_and_passes(self, monkeypatch, stub_engine):
        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin

        monkeypatch.setattr(
            "agentic_cli.workflow.adk.permission_plugin.get_service",
            lambda k: stub_engine if k == PERMISSION_ENGINE else None,
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
            tool=_adk_tool(reader_x),
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
            lambda k: denying_engine if k == PERMISSION_ENGINE else None,
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
            tool=_adk_tool(writer_x),
            tool_args={"path": "/etc/x"},
            tool_context=None,
        )
        assert result == {"success": False, "error": "Permission denied: rule: builtin/deny"}

    @pytest.mark.asyncio
    async def test_engine_absent_denies_when_permissions_enabled(self, monkeypatch):
        """Fail closed: permissions on but no engine wired -> deny, not allow."""
        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin

        monkeypatch.setattr(
            "agentic_cli.workflow.adk.permission_plugin.get_service",
            lambda k: None,
        )
        monkeypatch.setattr(
            "agentic_cli.config.get_settings",
            lambda: SimpleNamespace(permissions_enabled=True),
        )
        reg = get_registry()

        @reg.register(
            name="reader_y_deny",
            capabilities=[Capability("filesystem.read", target_arg="path")],
        )
        def reader_y_deny(path: str):
            return {}

        plugin = PermissionPlugin()
        result = await plugin.before_tool_callback(
            tool=_adk_tool(reader_y_deny),
            tool_args={"path": "/tmp/x"},
            tool_context=None,
        )
        assert result is not None and result["success"] is False

    @pytest.mark.asyncio
    async def test_engine_absent_allows_when_permissions_disabled(self, monkeypatch):
        """Permissions off is the only case a missing engine allows."""
        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin

        monkeypatch.setattr(
            "agentic_cli.workflow.adk.permission_plugin.get_service",
            lambda k: None,
        )
        monkeypatch.setattr(
            "agentic_cli.config.get_settings",
            lambda: SimpleNamespace(permissions_enabled=False),
        )
        reg = get_registry()

        @reg.register(
            name="reader_y_allow",
            capabilities=[Capability("filesystem.read", target_arg="path")],
        )
        def reader_y_allow(path: str):
            return {}

        plugin = PermissionPlugin()
        result = await plugin.before_tool_callback(
            tool=_adk_tool(reader_y_allow),
            tool_args={"path": "/tmp/x"},
            tool_context=None,
        )
        assert result is None
