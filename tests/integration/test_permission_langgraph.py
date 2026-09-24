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
    async def test_engine_absent_denies_when_permissions_enabled(self, monkeypatch):
        """Fail closed: permissions on but no engine wired -> deny, not run."""
        from types import SimpleNamespace

        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.langgraph.permission_wrap import wrap_tool_for_permission

        monkeypatch.setattr(
            "agentic_cli.workflow.langgraph.permission_wrap.get_service",
            lambda k: None,
        )
        monkeypatch.setattr(
            "agentic_cli.config.get_settings",
            lambda: SimpleNamespace(permissions_enabled=True),
        )
        reg = get_registry()

        @reg.register(
            name="read_lg_deny",
            capabilities=[Capability("filesystem.read", target_arg="path")],
        )
        def read_lg_deny(path: str):
            return {"ok": True}

        wrapped = wrap_tool_for_permission(read_lg_deny)
        result = await wrapped(path="/x")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_engine_absent_allows_when_permissions_disabled(self, monkeypatch):
        """Permissions off is the only case a missing engine runs the tool."""
        from types import SimpleNamespace

        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.langgraph.permission_wrap import wrap_tool_for_permission

        monkeypatch.setattr(
            "agentic_cli.workflow.langgraph.permission_wrap.get_service",
            lambda k: None,
        )
        monkeypatch.setattr(
            "agentic_cli.config.get_settings",
            lambda: SimpleNamespace(permissions_enabled=False),
        )
        reg = get_registry()

        @reg.register(
            name="read_lg_allow",
            capabilities=[Capability("filesystem.read", target_arg="path")],
        )
        def read_lg_allow(path: str):
            return {"ok": True}

        wrapped = wrap_tool_for_permission(read_lg_allow)
        result = await wrapped(path="/x")
        assert result == {"ok": True}


class TestWrapToolUsesRegistryIdentity:
    """LangGraph must resolve a tool by the object the registry issued, never
    by its name, exactly as the ADK PermissionPlugin does. A plain function
    that merely shares a registered tool's name is not that tool."""

    @pytest.fixture(autouse=True)
    def _builtins(self):
        import agentic_cli.tools  # noqa: F401  (registers the framework tools)

    @pytest.mark.asyncio
    async def test_same_named_function_does_not_inherit_exempt(self):
        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.langgraph.permission_wrap import wrap_tool_for_permission

        assert get_registry().get("ask_clarification").capabilities is EXEMPT

        def ask_clarification(question: str):
            return {"success": True, "ran": True}

        wrapped = wrap_tool_for_permission(ask_clarification)
        assert wrapped is not ask_clarification
        result = await wrapped(question="q")
        assert result["success"] is False
        assert "ran" not in result

    @pytest.mark.asyncio
    async def test_same_named_function_does_not_inherit_capabilities(self, monkeypatch):
        from agentic_cli.workflow.langgraph.permission_wrap import wrap_tool_for_permission
        from agentic_cli.workflow.permissions.rules import CheckResult

        engine = MagicMock()
        engine.check = AsyncMock(return_value=CheckResult(True, "rule: test/allow"))
        monkeypatch.setattr(
            "agentic_cli.workflow.langgraph.permission_wrap.get_service",
            lambda k: engine if k == PERMISSION_ENGINE else None,
        )

        def read_file(path: str):
            return {"success": True, "ran": True}

        result = await wrap_tool_for_permission(read_file)(path="/x")
        assert result["success"] is False
        assert "ran" not in result
        engine.check.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_renamed_tool_is_checked_under_its_registered_name(self, monkeypatch):
        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.langgraph.permission_wrap import wrap_tool_for_permission
        from agentic_cli.workflow.permissions.rules import CheckResult

        engine = MagicMock()
        engine.check = AsyncMock(return_value=CheckResult(True, "rule: test/allow"))
        monkeypatch.setattr(
            "agentic_cli.workflow.langgraph.permission_wrap.get_service",
            lambda k: engine if k == PERMISSION_ENGINE else None,
        )

        def _impl_lg(path: str):
            return {"success": True, "path": path}

        get_registry().register(
            _impl_lg,
            name="public_lg_name",
            capabilities=[Capability("filesystem.read", target_arg="path")],
        )
        result = await wrap_tool_for_permission(_impl_lg)(path="/y")
        assert result == {"success": True, "path": "/y"}
        assert engine.check.await_args.args[0] == "public_lg_name"


def test_every_tool_a_langgraph_agent_receives_is_identified(mock_context):
    """The graph builder wraps the output of ``_build_tools``. Identity lookup
    must resolve every entry (framework tools, service-bound variants and the
    LangGraph state tools) or genuine tools would be denied as unregistered."""
    pytest.importorskip("langgraph")
    from agentic_cli.tools import read_file, web_fetch, kb_search
    from agentic_cli.tools.registry import identify_tool
    from agentic_cli.workflow.config import AgentConfig
    from agentic_cli.workflow.langgraph.manager import LangGraphWorkflowManager

    config = AgentConfig(
        name="lg_agent",
        prompt="p",
        tools=[read_file, web_fetch, kb_search],
        include_state_tools=True,
    )
    manager = LangGraphWorkflowManager(agent_configs=[config], settings=mock_context.settings)
    service_map = {"web_fetch": web_fetch}  # no services initialized: map may be partial
    tools = manager._build_tools(config, service_map)
    assert len(tools) >= 3 + 4  # three framework tools + four state tools
    unidentified = [getattr(t, "__name__", t) for t in tools if identify_tool(t) is None]
    assert unidentified == []
