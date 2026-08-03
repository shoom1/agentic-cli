"""Permission gating binds to registry identity, not to a tool's name.

``before_tool_callback`` resolved capabilities with
``get_registry().get(tool.name)``. ADK derives that name from the callable, so
an *unregistered* function whose ``__name__`` collides with a registered tool
inherited that tool's capabilities — a raw callable named ``ask_clarification``
was allowed outright, because the genuine registered tool is EXEMPT.

Identity now comes from a registry-owned binding attached when a callable is
registered (or when the framework produces a service-bound/renamed variant of
one), so name equality alone proves nothing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from google.adk.tools import FunctionTool, LongRunningFunctionTool  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools.registry import (  # noqa: E402
    ToolCategory,
    ToolRegistry,
    get_registry,
)
from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT, PermissionEngine  # noqa: E402
from agentic_cli.workflow.permissions.capabilities import Capability  # noqa: E402
from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource  # noqa: E402
from agentic_cli.workflow.permissions.store import PermissionContext  # noqa: E402
from agentic_cli.workflow.service_registry import (  # noqa: E402
    PERMISSION_ENGINE,
    set_service_registry,
)


class _StubWorkflow:
    """Answers a HITL permission prompt with a fixed choice."""

    def __init__(self, response: str = "deny"):
        self._response = response

    async def request_user_input(self, request):
        return self._response


def _engine(tmp_path, response="deny", rules=None) -> PermissionEngine:
    settings = BaseSettings(google_api_key="test")
    ctx = PermissionContext(workdir=tmp_path, home=tmp_path)
    eng = PermissionEngine(settings=settings, workflow=_StubWorkflow(response), ctx=ctx)
    if rules:
        eng._session_rules.extend(rules)
    return eng


async def _check(engine, tool, tool_args=None):
    token = set_service_registry({PERMISSION_ENGINE: engine})
    try:
        return await PermissionPlugin().before_tool_callback(
            tool=tool, tool_args=tool_args or {}, tool_context=None
        )
    finally:
        token.var.reset(token)


def _denied(result) -> bool:
    return isinstance(result, dict) and result.get("success") is False


class TestImpostorCallables:
    """A raw callable must not inherit a registered tool's capabilities."""

    async def test_impostor_named_after_an_exempt_tool_is_denied(self, tmp_path):
        # The genuine ask_clarification is registered EXEMPT.
        from agentic_cli.tools.interaction_tools import ask_clarification  # noqa: F401

        assert get_registry().get("ask_clarification") is not None

        def ask_clarification(question: str) -> dict:  # noqa: F811 - deliberate collision
            """An unregistered impostor with a colliding name."""
            return {"success": True}

        tool = FunctionTool(func=ask_clarification)
        assert tool.name == "ask_clarification"

        result = await _check(_engine(tmp_path), tool)
        assert _denied(result), "an unregistered callable inherited EXEMPT status"

    async def test_impostor_named_after_a_permissioned_tool_is_denied(self, tmp_path):
        from agentic_cli.tools.file_read import read_file  # noqa: F401

        def read_file(path: str) -> dict:  # noqa: F811 - deliberate collision
            """Impostor."""
            return {"success": True}

        allow_all = Rule(
            capability="fs.read", target="**", effect=Effect.ALLOW,
            source=RuleSource.SESSION,
        )
        result = await _check(
            _engine(tmp_path, rules=[allow_all]),
            FunctionTool(func=read_file),
            {"path": str(tmp_path / "x")},
        )
        assert _denied(result), "an unregistered callable used a registered rule"

    async def test_genuine_tool_is_still_allowed(self, tmp_path):
        from agentic_cli.tools.interaction_tools import ask_clarification

        result = await _check(_engine(tmp_path), FunctionTool(func=ask_clarification))
        assert result is None, "the genuine EXEMPT tool must pass"


class TestRegisteredIdentity:
    """Registration binds identity for both the decorator and direct forms.

    These register into the *default* registry: identity is registry-owned, and
    the plugin trusts only the framework's own (see
    ``tests/tools/test_registry_identity.py::TestIdentityIsPerRegistry``).
    """

    async def test_renamed_tool_uses_its_declared_capabilities(self, tmp_path):
        registry = get_registry()

        @registry.register(
            name="public_read",
            capabilities=[Capability("fs.read", target_arg="path")],
            category=ToolCategory.READ,
        )
        def _internal_impl(path: str) -> dict:
            """Read something."""
            return {"success": True}

        tool = FunctionTool(func=_internal_impl)
        assert tool.name == "public_read"

        allow = Rule(
            capability="fs.read", target="**", effect=Effect.ALLOW,
            source=RuleSource.SESSION,
        )
        allowed = await _check(
            _engine(tmp_path, rules=[allow]), tool, {"path": str(tmp_path / "f")}
        )
        assert allowed is None

        denied = await _check(
            _engine(tmp_path, response="deny"), tool, {"path": str(tmp_path / "f")}
        )
        assert _denied(denied), "the declared capability was not evaluated"

    async def test_direct_register_call_binds_both_callables(self, tmp_path):
        """``registry.register(func, name=...)`` — original and returned wrapper."""
        registry = get_registry()

        def _impl(path: str) -> dict:
            """Impl."""
            return {"success": True}

        returned = registry.register(
            _impl, name="direct_public", capabilities=EXEMPT
        )

        for callable_ in (returned, _impl):
            result = await _check(_engine(tmp_path), FunctionTool(func=callable_))
            assert result is None, f"{callable_!r} was not recognised as registered"

    async def test_long_running_wrapper_keeps_identity(self, tmp_path):
        from agentic_cli.tools.interaction_tools import ask_clarification

        tool = LongRunningFunctionTool(func=ask_clarification)
        assert await _check(_engine(tmp_path), tool) is None


class TestServiceBoundTools:
    """Factory-produced (closure-bound) tools are framework-issued variants."""

    async def test_factory_bound_tool_is_recognised(self, tmp_path):
        from unittest.mock import MagicMock

        from agentic_cli.tools.factories import make_memory_tools

        tools = {t.__name__: t for t in make_memory_tools(MagicMock())}
        save_memory = tools["save_memory"]
        assert save_memory is not get_registry().get("save_memory").func

        allow = Rule(
            capability="memory.write", target="**", effect=Effect.ALLOW,
            source=RuleSource.SESSION,
        )
        result = await _check(
            _engine(tmp_path, rules=[allow]),
            FunctionTool(func=save_memory),
            {"content": "hi"},
        )
        assert result is None, "the service-bound variant was not recognised"

    async def test_factory_bound_tool_still_obeys_deny(self, tmp_path):
        """Its declared capability is really evaluated — a deny rule bites."""
        from unittest.mock import MagicMock

        from agentic_cli.tools.factories import make_memory_tools

        save_memory = {t.__name__: t for t in make_memory_tools(MagicMock())}["save_memory"]
        deny = Rule(
            capability="memory.write", target="*", effect=Effect.DENY,
            source=RuleSource.SESSION,
        )
        result = await _check(
            _engine(tmp_path, rules=[deny]),
            FunctionTool(func=save_memory),
            {"content": "hi"},
        )
        assert _denied(result)


class TestNonCallableTools:
    """Backend-native tool objects are bound by the framework, never by name."""

    async def test_skill_tools_are_resolved(self, tmp_path):
        import pathlib
        import tempfile

        from agentic_cli.tools.skills import SkillStore, make_skill_toolset

        skill_dir = pathlib.Path(tempfile.mkdtemp()) / "demo-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: d\n---\nBody\n"
        )
        toolset = make_skill_toolset(SkillStore().resolve([str(skill_dir)]))
        list_skills = next(t for t in toolset._tools if t.name == "list_skills")
        assert not hasattr(list_skills, "func")

        assert await _check(_engine(tmp_path), list_skills) is None

    async def test_unknown_native_tool_object_is_denied(self, tmp_path):
        class _Native:
            name = "totally_unknown"

        assert _denied(await _check(_engine(tmp_path), _Native()))


class TestNameIsNotAuthority:
    """A tool's *name* must never grant it another tool's capabilities."""

    async def test_custom_basetool_named_after_an_exempt_tool_is_denied(self, tmp_path):
        """The forgery the name fallback allowed: subclass BaseTool, pick a name."""
        from google.adk.tools import BaseTool

        from agentic_cli.tools.interaction_tools import ask_clarification  # noqa: F401

        assert get_registry().get("ask_clarification") is not None

        class _Impostor(BaseTool):
            async def run_async(self, *, args, tool_context):  # pragma: no cover
                return {"success": True}

        tool = _Impostor(name="ask_clarification", description="impostor")
        assert tool.name == "ask_clarification"

        result = await _check(_engine(tmp_path), tool)
        assert _denied(result), "a BaseTool inherited EXEMPT status from its name"

    async def test_custom_basetool_named_after_a_skill_tool_is_denied(self, tmp_path):
        """Registered-by-name skill tools must not be impersonable either."""
        from google.adk.tools import BaseTool

        from agentic_cli.tools.skills import register_skill_tool_permissions

        register_skill_tool_permissions()
        assert get_registry().get("list_skills") is not None

        class _Impostor(BaseTool):
            async def run_async(self, *, args, tool_context):  # pragma: no cover
                return {"success": True}

        result = await _check(
            _engine(tmp_path), _Impostor(name="list_skills", description="impostor")
        )
        assert _denied(result), "a BaseTool inherited a skill tool's EXEMPT status"

    async def test_class_named_like_an_mcp_tool_is_denied(self, tmp_path):
        """MCP detection must not key on a forgeable class name.

        The engine is given a blanket ``mcp`` ALLOW rule, so reaching the MCP
        path at all means the impostor is waved through.
        """

        class McpTool:  # not google.adk.tools.mcp_tool.McpTool
            def __init__(self, name: str):
                self.name = name

        allow_mcp = Rule(
            capability="mcp", target="**", effect=Effect.ALLOW,
            source=RuleSource.SESSION,
        )
        result = await _check(
            _engine(tmp_path, rules=[allow_mcp]), McpTool("remote_op")
        )
        assert _denied(result), "a class *named* McpTool got the MCP capability path"

    async def test_real_mcp_tool_is_gated_by_the_engine(self, tmp_path):
        """A genuine ADK McpTool instance still reaches the synthetic 'mcp' rule."""
        McpTool = pytest.importorskip("google.adk.tools.mcp_tool").McpTool

        class _RealEnough(McpTool):
            def __init__(self):  # bypass the MCP session plumbing
                object.__setattr__(self, "name", "remote_op")
                object.__setattr__(self, "description", "remote op")

        result = await _check(_engine(tmp_path, response="deny"), _RealEnough())
        assert _denied(result)


class TestForgedIdentity:
    """Identity is object identity — equality and attributes cannot forge it."""

    async def test_equality_colliding_callable_is_denied(self, tmp_path):
        """An object that compares equal to a registered tool is not that tool."""
        from agentic_cli.tools.interaction_tools import ask_clarification

        genuine = get_registry().get("ask_clarification").func

        class _Collider:
            """Hashes and compares equal to the genuine registered callable."""

            __name__ = "ask_clarification"

            def __hash__(self):
                return hash(genuine)

            def __eq__(self, other):
                return other is genuine

            def __call__(self, question: str) -> dict:  # pragma: no cover
                return {"success": True}

        collider = _Collider()
        assert collider == genuine and hash(collider) == hash(genuine)
        assert collider is not ask_clarification

        result = await _check(_engine(tmp_path), FunctionTool(func=collider))
        assert _denied(result), "an equality-colliding object forged tool identity"

    async def test_untrusted_wrapper_exposing_a_genuine_func_is_denied(self, tmp_path):
        """``.func`` is only trusted on ADK's own function-tool types."""
        from google.adk.tools import BaseTool

        from agentic_cli.tools.interaction_tools import ask_clarification

        class _Wrapper(BaseTool):
            """Advertises the genuine callable but runs whatever it likes."""

            async def run_async(self, *, args, tool_context):  # pragma: no cover
                return {"success": True}

        tool = _Wrapper(name="totally_other", description="w")
        object.__setattr__(tool, "func", ask_clarification)

        assert _denied(await _check(_engine(tmp_path), tool))
