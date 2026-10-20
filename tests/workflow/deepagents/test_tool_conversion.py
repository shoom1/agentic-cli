"""Framework tools as LangChain tools (``workflow/deepagents/tools.py``).

``create_agent`` runs ``BaseTool`` objects. A plain callable becomes a
``StructuredTool`` whose ``.func`` (or ``.coroutine``) *is* that callable, so
the permission gate, which unwraps exactly that type, identifies the tool the
registry issued. A ``tool_context`` parameter, ADK's injected name, is hidden
from the model and passed ``None``, so a tool written for ADK loads unchanged.
A long-running tool carries ADK's note. ``BaseTool`` objects pass through.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, NotRequired

import pytest

pytest.importorskip("deepagents")

from langchain.agents import AgentState  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.tools import BaseTool, StructuredTool, tool  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

from agentic_cli.tools.registry import ToolRegistry, get_registry  # noqa: E402
from agentic_cli.workflow.deepagents.tools import (  # noqa: E402
    LONG_RUNNING_NOTE,
    to_langchain_tools,
)
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402

ADK_NOTE = (
    "\n\nNOTE: This is a long-running operation. Do not call this tool again if"
    " it has already returned some intermediate or pending status."
)


def _call(name: str, args: dict, call_id: str = "call-1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _model_schema(tool: BaseTool) -> dict:
    """What the model is shown: name, description and parameters."""
    return convert_to_openai_tool(tool)["function"]


def _one(func, **kwargs) -> StructuredTool:
    (converted,) = to_langchain_tools([func], **kwargs)
    return converted


class TestCallables:
    def test_a_sync_callable_is_the_tools_func(self):
        def conv_read(path: str) -> dict:
            """Read a thing."""
            return {"success": True, "path": path}

        converted = _one(conv_read)

        assert type(converted) is StructuredTool
        assert converted.func is conv_read
        assert converted.coroutine is None

    def test_an_async_callable_is_the_tools_coroutine(self):
        async def conv_fetch(url: str) -> dict:
            """Fetch a thing."""
            return {"success": True, "url": url}

        converted = _one(conv_fetch)

        assert type(converted) is StructuredTool
        assert converted.coroutine is conv_fetch
        assert converted.func is None

    async def test_it_runs_the_callable(self):
        async def conv_echo(text: str) -> dict:
            """Echo."""
            return {"success": True, "text": text}

        message = await _one(conv_echo).ainvoke(_call("conv_echo", {"text": "hi"}))

        assert isinstance(message, ToolMessage)
        assert json.loads(message.content) == {"success": True, "text": "hi"}

    def test_the_name_is_the_callables_and_the_description_its_docstring(self):
        def conv_search(query: str, limit: int = 5) -> dict:
            """Search the notes.

            Args:
                query: What to look for.
                limit: How many results.
            """
            return {"success": True}

        schema = _model_schema(_one(conv_search))

        assert schema["name"] == "conv_search"
        assert schema["description"].startswith("Search the notes.\n\nArgs:\n    query:")
        assert set(schema["parameters"]["properties"]) == {"query", "limit"}
        assert schema["parameters"]["required"] == ["query"]

    def test_without_a_docstring_the_definition_describes_it(self):
        def _conv_undocumented(path: str) -> dict:
            return {"success": True}

        registered = get_registry().register(
            _conv_undocumented,
            name="conv_undocumented",
            description="Described by its registration.",
            capabilities=EXEMPT,
        )

        converted = _one(registered)

        assert converted.name == "conv_undocumented"
        assert converted.description == "Described by its registration."


class TestToolContext:
    """ADK injects ``tool_context``; here it is hidden and passed ``None``."""

    def test_it_is_left_out_of_the_model_schema(self):
        def conv_job(command: str, name: str = "", tool_context=None) -> dict:
            """Start a job."""
            return {"success": True}

        converted = _one(conv_job)

        assert converted.func is conv_job
        assert set(_model_schema(converted)["parameters"]["properties"]) == {"command", "name"}

    async def test_it_is_passed_none(self):
        seen: dict[str, Any] = {}

        def conv_strict(path: str, tool_context) -> dict:
            """Takes ADK's context, with no default."""
            seen.update(path=path, tool_context=tool_context)
            return {"success": True}

        message = await _one(conv_strict).ainvoke(_call("conv_strict", {"path": "/x"}))

        assert json.loads(message.content) == {"success": True}
        assert seen == {"path": "/x", "tool_context": None}

    async def test_a_value_the_model_sends_never_reaches_the_tool(self):
        seen: dict[str, Any] = {}

        def conv_guarded(path: str, tool_context=None) -> dict:
            """Must never see a model-made context."""
            seen["tool_context"] = tool_context
            return {"success": True}

        out = await _tool_graph([conv_guarded]).ainvoke(
            {"messages": [AIMessage("", tool_calls=[
                _call("conv_guarded", {"path": "/x", "tool_context": {"state": {}}})
            ])]}
        )

        assert json.loads(out["messages"][-1].content) == {"success": True}
        assert seen == {"tool_context": None}

    async def test_called_directly_a_model_made_value_is_refused(self):
        def conv_guarded_direct(path: str, tool_context=None) -> dict:
            """Must never see a model-made context."""
            raise AssertionError("the tool ran")

        with pytest.raises(ValueError, match="tool_context"):
            await _one(conv_guarded_direct).ainvoke(
                _call("conv_guarded_direct", {"path": "/x", "tool_context": {"state": {}}})
            )


class TestLongRunning:
    def test_a_long_running_tool_gets_adks_note_verbatim(self):
        @get_registry().register(capabilities=EXEMPT, long_running=True)
        def conv_long(command: str) -> dict:
            """Run a command in the background."""
            return {"success": True, "job_id": "j1"}

        converted = _one(conv_long)

        assert LONG_RUNNING_NOTE == ADK_NOTE
        assert converted.description == "Run a command in the background." + ADK_NOTE

    def test_other_tools_get_no_note(self):
        @get_registry().register(capabilities=EXEMPT)
        def conv_short(command: str) -> dict:
            """Run a command."""
            return {"success": True}

        assert _one(conv_short).description == "Run a command."

    def test_a_same_named_unregistered_callable_gets_no_note(self):
        """Long-running is what the registry says, never what a name suggests."""

        @get_registry().register(capabilities=EXEMPT, long_running=True)
        def conv_long_named(command: str) -> dict:
            """Registered long-running."""
            return {"success": True}

        def conv_long_named(command: str) -> dict:  # noqa: F811 - deliberate
            """An impostor."""
            return {"success": True}

        assert _one(conv_long_named).description == "An impostor."

    def test_the_registry_it_is_given_decides(self):
        registry = ToolRegistry()

        @registry.register(capabilities=EXEMPT, long_running=True)
        def conv_private_long(command: str) -> dict:
            """Long-running in its own registry."""
            return {"success": True}

        assert _one(conv_private_long).description == "Long-running in its own registry."
        assert _one(conv_private_long, registry=registry).description.endswith(ADK_NOTE)


class TestBaseTools:
    def test_they_pass_through_unchanged(self):
        @tool
        def conv_native(x: str) -> str:
            """A LangChain tool."""
            return x

        class Custom(BaseTool):
            name: str = "conv_custom"
            description: str = "A custom tool."

            def _run(self, *args, **kwargs):  # pragma: no cover
                return ""

        custom = Custom()

        assert to_langchain_tools([conv_native, custom]) == [conv_native, custom]
        assert to_langchain_tools([conv_native])[0] is conv_native


def _last(old, new):
    return new


class _State(AgentState):
    plan: Annotated[NotRequired[str], _last]


def _tool_graph(tools: list):
    """A graph that runs the last message's tool calls through a ToolNode."""
    graph = StateGraph(_State)
    graph.add_node("tools", ToolNode(to_langchain_tools(tools)))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile()


class TestStateTools:
    """The LangGraph-library state tools keep ``InjectedState``,
    ``InjectedToolCallId`` and their ``Command`` updates."""

    def _graph(self, tools: list):
        return _tool_graph(tools)

    async def test_save_plan_updates_the_state(self):
        from agentic_cli.tools.langgraph.state_tools import save_plan

        (converted,) = to_langchain_tools([save_plan])
        assert set(_model_schema(converted)["parameters"]["properties"]) == {"content"}

        out = await self._graph([save_plan]).ainvoke(
            {"messages": [AIMessage("", tool_calls=[_call("save_plan", {"content": "- [ ] a"})])]}
        )

        assert out["plan"] == "- [ ] a"
        assert json.loads(out["messages"][-1].content)["success"] is True
        assert out["messages"][-1].tool_call_id == "call-1"

    async def test_get_plan_reads_the_state(self):
        from agentic_cli.tools.langgraph.state_tools import get_plan

        (converted,) = to_langchain_tools([get_plan])
        assert _model_schema(converted)["parameters"].get("properties", {}) == {}

        out = await self._graph([get_plan]).ainvoke(
            {
                "messages": [AIMessage("", tool_calls=[_call("get_plan", {})])],
                "plan": "- [x] done",
            }
        )

        assert json.loads(out["messages"][-1].content) == {
            "success": True,
            "content": "- [x] done",
        }
