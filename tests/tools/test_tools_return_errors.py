"""A tool that fails returns ``{"success": False, "error": ...}``; it never raises.

On ADK an exception escaping a tool ends the whole turn: ADK re-raises it unless
an ``on_tool_error_callback`` answers for it. So each tool below that raised on
ordinary bad input now returns an error dict, and the ADK backend installs
:class:`ToolErrorPlugin` so that a tool which still raises (a bug, or an
application's own tool) costs the model one failed call instead of the turn.
LangGraph's ``ToolNode`` already runs with ``handle_tool_errors=True``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_cli.tools.file_read import diff_compare
from agentic_cli.tools.file_write import edit_file, write_file


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "notes.txt").write_text("item 123\n")
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00\x80binary")
    return tmp_path


def _failed(result: dict) -> str:
    assert result["success"] is False, result
    assert result["error"]
    return result["error"]


class TestFileTools:
    def test_write_file_under_a_file_path(self, project):
        error = _failed(write_file(str(project / "notes.txt" / "child.txt"), "x"))
        assert "notes.txt" in error
        assert (project / "notes.txt").read_text() == "item 123\n"

    @pytest.mark.parametrize("replacement", [r"\d", r"\2"])
    def test_edit_file_invalid_regex_replacement(self, project, replacement):
        error = _failed(edit_file(str(project / "notes.txt"), r"(\d+)", replacement, use_regex=True))
        assert "replacement" in error.lower()
        assert (project / "notes.txt").read_text() == "item 123\n"

    def test_diff_compare_binary_file(self, project):
        error = _failed(diff_compare(str(project / "blob.bin"), "text"))
        assert "blob.bin" in error

    def test_diff_compare_unreadable_file(self, project):
        locked = project / "locked.txt"
        locked.write_text("secret-free text\n")
        locked.chmod(0)
        try:
            if locked.stat() and _readable(locked):
                pytest.skip("running with privileges that ignore file modes")
            error = _failed(diff_compare("text", str(locked)))
            assert "locked.txt" in error
        finally:
            locked.chmod(0o600)

    def test_diff_compare_still_diffs_text(self, project):
        result = diff_compare(str(project / "notes.txt"), "item 124\n")
        assert result["success"] is True
        assert result["summary"]["changed"] == 1


def _readable(path: Path) -> bool:
    try:
        path.read_bytes()
    except PermissionError:
        return False
    return True


class TestJobList:
    def test_unknown_state_is_an_error_dict(self, tmp_path):
        from agentic_cli.tools.jobs import job_list
        from agentic_cli.tools.jobs.manager import JobManager
        from agentic_cli.workflow.service_registry import JOB_MANAGER, set_service_registry

        token = set_service_registry({JOB_MANAGER: JobManager(base_dir=tmp_path / "jobs")})
        try:
            error = _failed(job_list(state="finished"))
        finally:
            token.var.reset(token)
        assert "finished" in error
        assert "succeeded" in error  # names the valid states


class TestToolErrorPlugin:
    async def test_turns_an_exception_into_an_error_result(self):
        from agentic_cli.workflow.adk.tool_error_plugin import ToolErrorPlugin

        result = await ToolErrorPlugin().on_tool_error_callback(
            tool=SimpleNamespace(name="flaky_tool"),
            tool_args={"x": 1},
            tool_context=None,
            error=KeyError("missing"),
        )
        assert result["success"] is False
        assert "flaky_tool" in result["error"]
        assert "KeyError" in result["error"]

    async def test_bounds_the_message(self):
        from agentic_cli.workflow.adk.tool_error_plugin import ToolErrorPlugin

        result = await ToolErrorPlugin().on_tool_error_callback(
            tool=SimpleNamespace(name="t"), tool_args={}, tool_context=None,
            error=RuntimeError("x" * 10_000),
        )
        assert len(result["error"]) < 1_000

    def test_the_adk_manager_installs_it(self):
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
        from agentic_cli.workflow.adk.tool_error_plugin import ToolErrorPlugin

        manager = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
        manager._settings = SimpleNamespace(raw_llm_logging=False)
        assert any(isinstance(p, ToolErrorPlugin) for p in manager._init_plugins())


class TestAdkTurnSurvivesARaisingTool:
    """End to end on a real ADK Runner with a scripted model: the tool raises,
    the model receives an error result, and the turn finishes normally."""

    async def test_turn_completes_and_the_model_sees_the_error(self):
        from google.adk.agents import LlmAgent
        from google.adk.apps import App
        from google.adk.models.base_llm import BaseLlm
        from google.adk.models.llm_response import LlmResponse
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        from agentic_cli.workflow.adk.tool_error_plugin import ToolErrorPlugin

        seen: list[dict] = []

        def explode(path: str) -> dict:
            """Always fails."""
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        class ScriptedModel(BaseLlm):
            async def generate_content_async(self, llm_request, stream=False):
                responses = [
                    p.function_response.response
                    for c in llm_request.contents for p in (c.parts or [])
                    if p.function_response is not None
                ]
                if not responses:
                    call = types.FunctionCall(name="explode", args={"path": "a.bin"})
                    yield LlmResponse(content=types.Content(
                        role="model", parts=[types.Part(function_call=call)],
                    ))
                    return
                seen.extend(responses)
                yield LlmResponse(content=types.Content(
                    role="model", parts=[types.Part(text="The file is not text.")],
                ))

        agent = LlmAgent(name="root", model=ScriptedModel(model="scripted"), tools=[explode])
        app = App(name="t", root_agent=agent, plugins=[ToolErrorPlugin()])
        runner = InMemoryRunner(app=app)
        session = await runner.session_service.create_session(app_name="t", user_id="u")
        texts = []
        async for event in runner.run_async(
            user_id="u", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="read a.bin")]),
        ):
            texts += [p.text for p in (event.content.parts if event.content else []) if p.text]

        assert texts == ["The file is not text."]
        assert seen and seen[0]["success"] is False
        assert "UnicodeDecodeError" in seen[0]["error"]


class TestLangGraphTurnSurvivesARaisingTool:
    """The LangGraph builder's tool node answers a raising tool with an error
    ToolMessage (``handle_tool_errors=True``), so the graph keeps running."""

    async def test_tool_node_returns_an_error_message(self):
        from unittest.mock import MagicMock

        from langchain_core.messages import AIMessage
        from langgraph.graph import END, StateGraph

        from agentic_cli.tools.registry import ToolCategory, register_tool
        from agentic_cli.workflow.config import AgentConfig
        from agentic_cli.workflow.langgraph.graph_builder import LangGraphBuilder
        from agentic_cli.workflow.langgraph.state import AgentState
        from agentic_cli.workflow.permissions import EXEMPT

        @register_tool(category=ToolCategory.OTHER, capabilities=EXEMPT, description="probe")
        def _errors_probe_raises(path: str) -> dict:
            """Always fails."""
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        built = LangGraphBuilder(MagicMock()).build(
            [AgentConfig(name="a", prompt="p", tools=[_errors_probe_raises])], "gpt-4o-mini",
        )
        graph = StateGraph(AgentState)
        graph.add_node("tools", built.nodes["a_tools"].runnable)
        graph.set_entry_point("tools")
        graph.add_edge("tools", END)

        call = {"name": "_errors_probe_raises", "args": {"path": "a.bin"}, "id": "c1"}
        out = await graph.compile().ainvoke(
            {"messages": [AIMessage(content="", tool_calls=[call])]}
        )

        reply = out["messages"][-1]
        assert reply.status == "error"
        assert "UnicodeDecodeError" in reply.content
