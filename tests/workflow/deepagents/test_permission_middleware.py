"""The permission gate and the tool-error middleware on Deep Agents.

``PermissionMiddleware.awrap_tool_call`` runs around every tool call in every
agent's stack. It identifies the tool by the object the registry issued,
never by its name, and fails closed, in the spec's order:

1. objects the framework built and bound (``task``, skill and MCP tools);
2. for an exact ``StructuredTool``, the callable it will run;
3. EXEMPT runs without an engine call;
4. unresolved, or no capability declaration: denied;
5. no engine: denied with permissions on, run with them off;
6. ``engine.check`` — ASK goes through the context-local input callback.

A denial is the call's error ``ToolMessage``, never an exception, so the model
carries on. ``ToolCallErrorMiddleware`` sits inside the gate and gives a
raising tool ADK's error result.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("deepagents")

from langchain.agents.middleware import ToolCallRequest  # noqa: E402
from langchain_core.messages import ToolMessage  # noqa: E402
from langchain_core.tools import BaseTool, StructuredTool  # noqa: E402
from langgraph.errors import GraphInterrupt  # noqa: E402
from langgraph.types import Command  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agentic_cli.tools.registry import (  # noqa: E402
    ToolCategory,
    ToolDefinition,
    bind_tool_identity,
    get_registry,
    register_tool,
)
from agentic_cli.workflow.base_manager import BaseWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.middleware import (  # noqa: E402
    PermissionMiddleware,
    ToolCallErrorMiddleware,
)
from agentic_cli.workflow.deepagents.tools import to_langchain_tools  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT, PermissionEngine  # noqa: E402
from agentic_cli.workflow.permissions.capabilities import Capability  # noqa: E402
from agentic_cli.workflow.permissions.prompt import (  # noqa: E402
    ALLOW_ONCE_CHOICE,
    DENY_CHOICE,
)
from agentic_cli.workflow.permissions.rules import CheckResult  # noqa: E402
from agentic_cli.workflow.permissions.store import PermissionContext  # noqa: E402
from agentic_cli.workflow.service_registry import (  # noqa: E402
    PERMISSION_ENGINE,
    set_service_registry,
)
from agentic_cli.workflow.tool_errors import (  # noqa: E402
    tool_error_result,
    unknown_tool_result,
)

READ = [Capability("filesystem.read", target_arg="path")]


@register_tool(capabilities=READ, category=ToolCategory.READ)
def pmw_read(path: str) -> dict:
    """Read a file (a permissioned test tool)."""
    return {"success": True, "path": path}


@register_tool(capabilities=EXEMPT)
def pmw_note(text: str) -> dict:
    """Take a note (an exempt test tool)."""
    return {"success": True, "text": text}


async def _pmw_private_name(path: str) -> dict:
    """Read under a public name."""
    return {"success": True, "path": path}


get_registry().register(_pmw_private_name, name="pmw_public", capabilities=READ)


def _convert(func) -> StructuredTool:
    (converted,) = to_langchain_tools([func])
    return converted


class _Native(BaseTool):
    """A tool object that is not a StructuredTool, as MCP and skill tools are."""

    name: str = "native_op"
    description: str = "A native tool."

    def _run(self, **kwargs: Any) -> str:
        return json.dumps({"success": True, "native": True})


def _engine(allowed: bool = True, reason: str = "rule: test/allow") -> MagicMock:
    engine = MagicMock()
    engine.check = AsyncMock(return_value=CheckResult(allowed, reason))
    return engine


async def _gate(
    tool: BaseTool | None,
    args: dict | None = None,
    *,
    engine: Any = None,
    gate: PermissionMiddleware | None = None,
    name: str | None = None,
    tools: list[BaseTool] | None = None,
) -> tuple[ToolMessage | Command, list[str]]:
    """Put one call through the gate; return its result and what ran."""
    call = {
        "name": name or tool.name,
        "args": args or {},
        "id": "call-1",
        "type": "tool_call",
    }
    ran: list[str] = []

    async def execute(request: ToolCallRequest) -> ToolMessage | Command:
        ran.append(request.tool_call["name"])
        return await request.tool.ainvoke(request.tool_call)

    request = ToolCallRequest(
        tool_call=call,
        tool=tool,
        state={},
        runtime=SimpleNamespace(tools=tools if tools is not None else [tool]),
    )
    token = set_service_registry({PERMISSION_ENGINE: engine} if engine else {})
    try:
        result = await (gate or PermissionMiddleware("researcher")).awrap_tool_call(
            request, execute
        )
    finally:
        token.var.reset(token)
    return result, ran


def _content(message: ToolMessage) -> dict:
    return json.loads(message.content)


def _denied(message: ToolMessage) -> str:
    """The denial's reason; fails the test if the call was not denied."""
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    content = _content(message)
    assert content["success"] is False
    assert content["error"].startswith("Permission denied: ")
    return content["error"].removeprefix("Permission denied: ")


@pytest.fixture
def permissions_on(monkeypatch):
    monkeypatch.setattr(
        "agentic_cli.config.get_settings",
        lambda: SimpleNamespace(permissions_enabled=True),
    )


@pytest.fixture
def permissions_off(monkeypatch):
    monkeypatch.setattr(
        "agentic_cli.config.get_settings",
        lambda: SimpleNamespace(permissions_enabled=False),
    )


class TestADenial:
    async def test_is_the_calls_error_tool_message(self):
        def unregistered(path: str) -> dict:
            """Not the framework's."""
            return {"success": True}

        message, ran = await _gate(_convert(unregistered), {"path": "/x"}, engine=_engine())

        assert ran == []
        assert message.tool_call_id == "call-1"
        assert message.name == "unregistered"
        assert message.status == "error"
        assert _content(message) == {
            "success": False,
            "error": (
                "Permission denied: tool is not registered "
                "(register it with @register_tool to declare capabilities)"
            ),
        }


class TestStep1BoundObjects:
    async def test_a_bound_object_is_identified_by_identity(self):
        native = _Native()
        bind_tool_identity(native, get_registry().get("pmw_note"))

        message, ran = await _gate(native, engine=_engine())

        assert ran == ["native_op"]
        assert _content(message) == {"success": True, "native": True}

    async def test_the_bound_resolver_is_consulted_first(self):
        native = _Native(name="remote_op")
        synthetic = ToolDefinition(
            name="remote_op",
            description="An MCP tool.",
            func=None,
            capabilities=[Capability("mcp", target="remote_op")],
        )
        engine = _engine()
        gate = PermissionMiddleware(
            "researcher", bound=lambda obj: synthetic if obj is native else None
        )

        message, ran = await _gate(native, {"q": 1}, engine=engine, gate=gate)

        assert ran == ["remote_op"]
        engine.check.assert_awaited_once_with("remote_op", synthetic.capabilities, {"q": 1})


class TestStep2ExactStructuredTool:
    async def test_the_callable_it_runs_is_identified(self):
        """Built by anyone, an exact StructuredTool runs exactly its callable."""
        engine = _engine()

        message, ran = await _gate(
            StructuredTool.from_function(func=pmw_read), {"path": "/x"}, engine=engine
        )

        assert ran == ["pmw_read"]
        engine.check.assert_awaited_once_with("pmw_read", READ, {"path": "/x"})

    async def test_a_converted_framework_tool_is_identified(self):
        message, ran = await _gate(_convert(pmw_read), {"path": "/x"}, engine=_engine())

        assert ran == ["pmw_read"]
        assert _content(message) == {"success": True, "path": "/x"}

    async def test_the_coroutine_is_what_an_async_run_invokes(self):
        """A genuine ``.func`` beside some other ``.coroutine`` proves nothing:
        the coroutine is what runs."""

        async def other(text: str) -> dict:
            return {"success": True, "ran": "other"}

        forged = StructuredTool.from_function(
            func=pmw_note, coroutine=other, name="pmw_note", description="d"
        )

        message, ran = await _gate(forged, {"text": "t"}, engine=_engine())

        assert ran == []
        assert "not registered" in _denied(message)


class TestStep3Exempt:
    async def test_an_exempt_tool_runs_without_an_engine_call(self):
        engine = _engine(allowed=False, reason="would deny")

        message, ran = await _gate(_convert(pmw_note), {"text": "t"}, engine=engine)

        assert ran == ["pmw_note"]
        engine.check.assert_not_awaited()


class TestStep4Unresolved:
    async def test_an_unregistered_callable_is_denied(self):
        def scratch(path: str) -> dict:
            """An application's unregistered tool."""
            return {"success": True}

        engine = _engine()
        message, ran = await _gate(_convert(scratch), {"path": "/x"}, engine=engine)

        assert ran == []
        assert "not registered" in _denied(message)
        engine.check.assert_not_awaited()

    async def test_a_definition_without_capabilities_is_denied(self):
        native = _Native()
        undeclared = ToolDefinition(
            name="native_op", description="", func=None, capabilities=[]
        )
        engine = _engine()
        gate = PermissionMiddleware("researcher", bound=lambda obj: undeclared)

        message, ran = await _gate(native, engine=engine, gate=gate)

        assert ran == []
        assert _denied(message) == "tool has no capability declaration"
        engine.check.assert_not_awaited()


class TestStep5NoEngine:
    async def test_with_permissions_on_the_call_is_denied(self, permissions_on):
        message, ran = await _gate(_convert(pmw_read), {"path": "/x"})

        assert ran == []
        assert _denied(message) == "permission engine unavailable"

    async def test_with_permissions_off_the_call_runs(self, permissions_off):
        message, ran = await _gate(_convert(pmw_read), {"path": "/x"})

        assert ran == ["pmw_read"]
        assert _content(message) == {"success": True, "path": "/x"}

    async def test_permissions_off_still_denies_an_unregistered_tool(self, permissions_off):
        def scratch(path: str) -> dict:
            """Unregistered."""
            return {"success": True}

        message, ran = await _gate(_convert(scratch), {"path": "/x"})

        assert ran == []
        assert "not registered" in _denied(message)


class TestStep6EngineCheck:
    async def test_an_allowed_call_runs(self):
        engine = _engine()

        message, ran = await _gate(_convert(pmw_read), {"path": "/abs/x"}, engine=engine)

        assert ran == ["pmw_read"]
        engine.check.assert_awaited_once_with("pmw_read", READ, {"path": "/abs/x"})

    async def test_a_denied_call_carries_the_engines_reason(self):
        engine = _engine(allowed=False, reason="rule: builtin/deny")

        message, ran = await _gate(_convert(pmw_read), {"path": "/etc/x"}, engine=engine)

        assert ran == []
        assert _denied(message) == "rule: builtin/deny"

    async def test_a_renamed_tool_is_checked_under_its_registered_name(self):
        engine = _engine()
        tool = StructuredTool.from_function(coroutine=_pmw_private_name)

        message, ran = await _gate(tool, {"path": "/y"}, engine=engine)

        assert tool.name == "_pmw_private_name"
        assert ran == ["_pmw_private_name"]
        assert engine.check.await_args.args[0] == "pmw_public"


class _Manager(BaseWorkflowManager):
    """The smallest manager: it owns the context-local input callback."""

    @property
    def backend_type(self) -> str:
        return "test"

    async def _do_initialize(self) -> None:
        pass

    async def process(self, message: str, user_id: str, session_id: str | None = None):
        if False:
            yield  # pragma: no cover

    async def reinitialize(self, model: str | None = None, preserve_sessions: bool = True) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    def _get_state_tools(self) -> list:
        return []


class TestAskThroughTheInputCallback:
    """No rule matches a read outside the project, so the engine asks the
    consumer that installed the input callback."""

    @pytest.fixture
    def world(self, tmp_path, monkeypatch, mock_context):
        home, project = tmp_path / "home", tmp_path / "project"
        home.mkdir()
        project.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(project)
        manager = _Manager(agent_configs=[], settings=mock_context.settings)
        engine = PermissionEngine(
            settings=mock_context.settings,
            workflow=manager,
            ctx=PermissionContext(workdir=project, home=home),
        )
        return SimpleNamespace(manager=manager, engine=engine, outside=tmp_path / "x.txt")

    async def test_allowing_runs_the_tool(self, world):
        asked = []

        async def answer(request):
            asked.append(request)
            return ALLOW_ONCE_CHOICE

        world.manager.set_input_callback(answer)

        message, ran = await _gate(
            _convert(pmw_read), {"path": str(world.outside)}, engine=world.engine
        )

        assert [request.tool_name for request in asked] == ["pmw_read"]
        assert ran == ["pmw_read"]
        assert _content(message)["success"] is True

    async def test_denying_denies_the_call(self, world):
        async def answer(request):
            return DENY_CHOICE

        world.manager.set_input_callback(answer)

        message, ran = await _gate(
            _convert(pmw_read), {"path": str(world.outside)}, engine=world.engine
        )

        assert ran == []
        assert _denied(message) == "no rule + user denied"

    async def test_without_a_callback_the_call_is_denied(self, world):
        message, ran = await _gate(
            _convert(pmw_read), {"path": str(world.outside)}, engine=world.engine
        )

        assert ran == []
        assert _denied(message) == "no rule + no interactive approver to ask"


class TestSpoofing:
    """A name, a class or an attribute is never a tool's identity."""

    @pytest.fixture(autouse=True)
    def _framework_tools(self):
        import agentic_cli.tools  # noqa: F401  (registers the framework tools)

    async def test_a_same_named_function_does_not_inherit_exempt(self):
        assert get_registry().get("ask_clarification").capabilities is EXEMPT

        def ask_clarification(question: str) -> dict:
            """An impostor."""
            return {"success": True, "ran": True}

        message, ran = await _gate(_convert(ask_clarification), {"question": "q"})

        assert ran == []
        assert "not registered" in _denied(message)

    async def test_a_same_named_function_does_not_inherit_capabilities(self):
        engine = _engine()

        def read_file(path: str) -> dict:
            """An impostor."""
            return {"success": True, "ran": True}

        message, ran = await _gate(_convert(read_file), {"path": "/x"}, engine=engine)

        assert ran == []
        assert "not registered" in _denied(message)
        engine.check.assert_not_awaited()

    async def test_a_structured_tool_subclass_is_denied(self):
        """A subclass may run anything while advertising a genuine ``.func``."""

        class Subclass(StructuredTool):
            pass

        tool = Subclass.from_function(func=pmw_note)
        assert type(tool) is Subclass and tool.func is pmw_note

        message, ran = await _gate(tool, {"text": "t"}, engine=_engine())

        assert ran == []
        assert "not registered" in _denied(message)

    async def test_a_forged_func_on_another_tool_type_is_denied(self):
        tool = _Native(name="pmw_note")
        object.__setattr__(tool, "func", pmw_note)

        message, ran = await _gate(tool, {"text": "t"}, engine=_engine())

        assert ran == []
        assert "not registered" in _denied(message)

    async def test_an_exempt_impersonator_is_denied(self):
        """A tool object named after an EXEMPT tool is not that tool."""
        message, ran = await _gate(_Native(name="ask_clarification"), {"question": "q"})

        assert ran == []
        assert "not registered" in _denied(message)

    async def test_an_equality_colliding_callable_is_denied(self):
        genuine = get_registry().get("ask_clarification").func

        class Collider:
            __name__ = "ask_clarification"

            def __hash__(self):
                return hash(genuine)

            def __eq__(self, other):
                return other is genuine

            def __call__(self, question: str) -> dict:
                return {"success": True, "ran": True}

        class Question(BaseModel):
            question: str

        collider = Collider()
        tool = StructuredTool(
            name="ask_clarification", description="d", args_schema=Question, func=collider
        )
        assert tool.func is collider

        message, ran = await _gate(tool, {"question": "q"})

        assert ran == []
        assert "not registered" in _denied(message)


class TestUnknownTool:
    async def test_it_gets_adks_text_without_an_engine_call(self):
        engine = _engine()
        tools = [_convert(pmw_read), _convert(pmw_note)]

        message, ran = await _gate(None, {}, engine=engine, name="nope", tools=tools)

        assert ran == []
        assert message.status == "error"
        assert message.name == "nope"
        assert _content(message) == unknown_tool_result("nope", ["pmw_read", "pmw_note"])
        engine.check.assert_not_awaited()


class TestSyncPath:
    def test_it_fails_closed(self):
        handler = MagicMock()
        request = ToolCallRequest(
            tool_call={"name": "pmw_note", "args": {"text": "t"}, "id": "call-1", "type": "tool_call"},
            tool=_convert(pmw_note),
            state={},
            runtime=SimpleNamespace(tools=[]),
        )

        message = PermissionMiddleware("researcher").wrap_tool_call(request, handler)

        handler.assert_not_called()
        assert _denied(message) == "synchronous tool path is not supported"


def _request(tool: BaseTool | None, name: str = "flaky") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {"x": 1}, "id": "call-9", "type": "tool_call"},
        tool=tool,
        state={},
        runtime=SimpleNamespace(tools=[]),
    )


def _raising(error: BaseException):
    async def handler(request):
        raise error

    return handler


class TestToolCallErrorMiddleware:
    async def test_a_raising_tool_gets_adks_error_result(self):
        error = KeyError("missing")

        message = await ToolCallErrorMiddleware().awrap_tool_call(
            _request(_Native(name="flaky")), _raising(error)
        )

        assert isinstance(message, ToolMessage)
        assert message.status == "error"
        assert message.name == "flaky"
        assert message.tool_call_id == "call-9"
        assert _content(message) == tool_error_result("flaky", error)

    async def test_a_result_passes_through(self):
        command = Command(update={"plan": "p"})

        async def handler(request):
            return command

        assert await ToolCallErrorMiddleware().awrap_tool_call(_request(_Native()), handler) is command

    async def test_cancelling_still_cancels(self):
        with pytest.raises(asyncio.CancelledError):
            await ToolCallErrorMiddleware().awrap_tool_call(
                _request(_Native()), _raising(asyncio.CancelledError())
            )

    async def test_graph_control_flow_propagates(self):
        with pytest.raises(GraphInterrupt):
            await ToolCallErrorMiddleware().awrap_tool_call(
                _request(_Native()), _raising(GraphInterrupt())
            )

    async def test_a_failed_delegation_ends_the_turn(self):
        """What escapes ``task`` is the sub-agent's model or run failing — its
        own tools' errors were already results — so it ends the turn as a
        model failure at the root would."""
        delegation = _Native(name="task")
        bind_tool_identity(delegation, get_registry().get("task"))

        with pytest.raises(RuntimeError, match="provider said 400"):
            await ToolCallErrorMiddleware().awrap_tool_call(
                _request(delegation, name="task"), _raising(RuntimeError("provider said 400"))
            )

    async def test_a_tool_merely_named_task_is_an_ordinary_tool(self):
        error = RuntimeError("boom")

        message = await ToolCallErrorMiddleware().awrap_tool_call(
            _request(_Native(name="task"), name="task"), _raising(error)
        )

        assert _content(message) == tool_error_result("task", error)

    def test_the_sync_path_converts_too(self):
        error = ValueError("bad")

        def handler(request):
            raise error

        message = ToolCallErrorMiddleware().wrap_tool_call(_request(_Native(name="flaky")), handler)

        assert _content(message) == tool_error_result("flaky", error)
