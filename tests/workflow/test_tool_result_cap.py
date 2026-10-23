"""The tool result cap on ADK, and reading a saved output back.

ADK's ``ToolResultCapPlugin`` gives every tool result to
``workflow/tool_output.py::cap_tool_result`` in its ``after_tool_callback``:
registered tools, MCP and skill tools, whatever the tool object. It is
registered after ``TaskProgressPlugin``, since ADK stops at the first plugin
whose ``after_tool_callback`` returns a value, and only for apps built from
agent configs. ADK's Claude and OpenAI Responses adapters send a result's
``content`` alone when it has one, else (OpenAI Responses) its ``result``, so
the note is put there too. A ``read_file`` of a saved output (by the tool's
identity) is cut again without a second file, and needs no permission prompt.

Turns run the real manager on ``FakeAPI`` (Claude, and OpenAI Responses);
Deep Agents' side is in ``tests/workflow/deepagents/test_tool_result_cap.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
import openai
import pytest

pytest.importorskip("google.adk")

from google.adk.integrations.openai._openai_responses_llm import (  # noqa: E402
    _serialize_json_value,
)
from google.adk.models.anthropic_llm import (  # noqa: E402
    _part_to_message_block,
    _ToolUseIdSanitizer,
)
from google.adk.tools import FunctionTool  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools import read_file  # noqa: E402
from agentic_cli.tools.registry import get_registry  # noqa: E402
from agentic_cli.tools.skills import SkillStore, make_skill_toolset  # noqa: E402
from agentic_cli.workflow.adk import tool_result_cap_plugin  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.adk.task_progress_plugin import TaskProgressPlugin  # noqa: E402
from agentic_cli.workflow.adk.tool_result_cap_plugin import ToolResultCapPlugin  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.permissions import PermissionContext, PermissionEngine  # noqa: E402
from agentic_cli.workflow.permissions.prompt import ALLOW_ONCE_CHOICE  # noqa: E402
from agentic_cli.workflow.tool_output import tool_output_dir  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message, openai_response  # noqa: E402
from tests.workflow.cap_probes import (  # noqa: E402
    CUT,
    NOTE_IN_TEXT,
    REPORT,
    SAVED_NOTE,
    UNSAVED_NOTE,
    cap_probe_job,
    cap_probe_report,
    cap_probe_tasks,
    saved_outputs,
    saved_path,
)
from tests.workflow.langchain_fakes import openai_function_call  # noqa: E402
from tests.workflow.test_adk_context_cache import _native_yaml  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402
from tests.workflow.test_skills import _make_skill  # noqa: E402

_McpTool = pytest.importorskip("google.adk.tools.mcp_tool").McpTool

#: A log file far over the limit, one numbered line after another.
LOG = "".join(f"log line {i:06d}\n" for i in range(8_000))


class _MCPTool(_McpTool):
    """A real ``McpTool`` instance without the MCP session plumbing."""

    def __init__(self, name: str):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", f"MCP tool {name}")


class _Recorder:
    """Stands in for a module's structlog logger."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.calls.append((level, event, fields))

        return log


class Answers(list):
    """A provider's answers, in order: a ``FakeAPI`` answer a test can add to."""

    def __call__(self, request) -> dict:
        return self.pop(0)


def call(name: str, **args) -> dict:
    return anthropic_message({"type": "tool_use", "id": f"toolu_{name}", "name": name,
                              "input": args})


def says(text: str) -> dict:
    return anthropic_message({"type": "text", "text": text})


def tool_result(body: dict):
    """The content of the last tool result a Claude request carries."""
    blocks = [
        block
        for message in body["messages"]
        if message["role"] == "user" and isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    return blocks[-1]["content"]


def function_output(body: dict) -> str:
    """The output of the last function call an OpenAI Responses request carries."""
    outputs = [item for item in body["input"] if item.get("type") == "function_call_output"]
    return outputs[-1]["output"]


def claude_reads(response: dict) -> str:
    """What ADK's Claude adapter sends Claude of a function response."""
    part = types.Part.from_function_response(name="probe", response=response)
    part.function_response.id = "toolu_1"
    content = _part_to_message_block(part, _ToolUseIdSanitizer())["content"]
    return content if isinstance(content, str) else json.dumps(content)


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A hermetic project (the cwd) and HOME, with a fake Claude key."""
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    return tmp_path / "project"


@pytest.fixture
def claude(monkeypatch) -> tuple[FakeAPI, Answers]:
    answers = Answers()
    api = FakeAPI(answers)
    route(monkeypatch, anthropic, "AsyncAnthropic", api)
    return api, answers


def adk(tools: list, **values) -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="Answer.", tools=tools)],
        settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6", **values),
    )


async def turn(manager, text: str = "Go.") -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


def texts(events) -> list[str]:
    return [event.content for event in events if event.type is EventType.TEXT]


def folder() -> Path:
    return tool_output_dir("agentic_cli")


class TestTurns:
    async def test_the_next_request_carries_the_cut_result_and_the_note(self, project, claude):
        api, answers = claude
        answers += [call("cap_probe_report"), says("Read it.")]
        manager = adk([cap_probe_report])
        try:
            events = await turn(manager)
        finally:
            await manager.cleanup()

        sent = json.loads(tool_result(api.bodies[1]))
        assert sent["success"] is True
        assert CUT.search(sent["report"])
        assert REPORT.startswith(sent["report"][:1_000])
        assert REPORT in saved_path(sent["truncated"]).read_text()
        assert texts(events) == ["Read it."]

    async def test_task_progress_still_arrives(self, project, claude):
        # The cap answers ADK's after-tool chain, which stops there: the task
        # progress plugin has to have run already.
        api, answers = claude
        answers += [call("cap_probe_tasks"), says("Done.")]
        manager = adk([cap_probe_tasks])
        try:
            events = await turn(manager)
        finally:
            await manager.cleanup()

        (progress,) = [e for e in events if e.type is EventType.TASK_PROGRESS]
        assert "Read the report" in progress.content
        assert SAVED_NOTE.match(json.loads(tool_result(api.bodies[1]))["truncated"])

    async def test_a_result_that_fits_is_sent_as_it_was(self, project, claude):
        (project / "small.txt").write_text("small\n")
        api, answers = claude
        answers += [call("read_file", path="small.txt"), says("Read.")]
        manager = adk([read_file])
        try:
            await turn(manager)
        finally:
            await manager.cleanup()

        assert tool_result(api.bodies[1]) == "small\n"
        assert not folder().exists()

    async def test_claude_reads_the_note_of_a_cut_read_file(self, project, claude):
        # ADK's Claude adapter sends only "content" of a result that has one.
        (project / "big.log").write_text(LOG)
        api, answers = claude
        answers += [call("read_file", path="big.log"), says("Read.")]
        manager = adk([read_file])
        try:
            await turn(manager)
        finally:
            await manager.cleanup()

        body, _, note = tool_result(api.bodies[1]).rpartition("\n\n")
        assert CUT.search(body)
        assert body.startswith("log line 000000\n")
        assert LOG in saved_path(note).read_text()

    async def test_a_read_file_of_a_saved_output_is_cut_without_a_new_file(
        self, project, claude
    ):
        (project / "big.log").write_text(LOG)
        api, answers = claude
        answers += [call("read_file", path="big.log"), says("Read.")]
        manager = adk([read_file])
        try:
            await turn(manager)
            (saved,) = saved_outputs(folder())
            answers += [call("read_file", path=str(saved)), says("Again.")]
            events = await turn(manager, "Read it all.")
        finally:
            await manager.cleanup()

        body, _, note = tool_result(api.bodies[3]).rpartition("\n\n")
        assert CUT.search(body)
        assert saved_path(note) == saved
        assert saved_outputs(folder()) == [saved]
        assert texts(events) == ["Again."]

    @pytest.mark.parametrize("provider", ["claude", "openai"])
    async def test_the_model_reads_the_note_of_a_cut_job_result(
        self, project, monkeypatch, provider
    ):
        # OpenAI Responses is sent a job result's "result" alone.
        if provider == "claude":
            api = FakeAPI(Answers([call("cap_probe_job"), says("Read it.")]))
            route(monkeypatch, anthropic, "AsyncAnthropic", api)
            manager = adk([cap_probe_job])
        else:
            monkeypatch.setenv("OPENAI_API_KEY", "o-test-key")
            api = FakeAPI(Answers([
                openai_function_call("cap_probe_job", {}, "call_1"), openai_response("Read it."),
            ]))
            route(monkeypatch, openai, "AsyncOpenAI", api)
            manager = GoogleADKWorkflowManager(
                agent_configs=[AgentConfig(name="solo", prompt="Answer.", tools=[cap_probe_job])],
                settings=BaseSettings(session_store="memory", default_model="openai:gpt-5.5"),
            )
        try:
            events = await turn(manager)
        finally:
            await manager.cleanup()

        body = api.bodies[1]
        sent = tool_result(body) if provider == "claude" else function_output(body)
        assert "characters cut]" in sent  # in JSON text: its line breaks are escaped
        note = NOTE_IN_TEXT.search(sent)
        assert note and REPORT in saved_path(note[0]).read_text()
        assert texts(events) == ["Read it."]

    @pytest.mark.parametrize("project_is", ["read-only", "symlinked"])
    async def test_a_project_the_cap_cannot_save_in(self, project, claude, project_is):
        # Every call still succeeds, cut, and nothing is written anywhere.
        if project_is == "read-only" and os.geteuid() == 0:
            pytest.skip("root writes into a read-only folder")
        elsewhere = project.parent / "elsewhere"
        elsewhere.mkdir()
        if project_is == "symlinked":
            (project / ".agentic_cli").symlink_to(elsewhere, target_is_directory=True)
        else:
            project.chmod(0o500)
        api, answers = claude
        answers += [call("cap_probe_report"), says("Read it.")]
        manager = adk([cap_probe_report])
        try:
            events = await turn(manager)
        finally:
            project.chmod(0o700)
            await manager.cleanup()

        sent = json.loads(tool_result(api.bodies[1]))
        assert UNSAVED_NOTE.match(sent["truncated"])
        assert CUT.search(sent["report"])
        assert texts(events) == ["Read it."]
        assert list(elsewhere.iterdir()) == []
        assert [p.name for p in project.iterdir()] == (
            [".agentic_cli"] if project_is == "symlinked" else []
        )


class TestTheApp:
    async def test_the_cap_runs_after_task_progress_on_the_settings(self, project):
        manager = adk([cap_probe_report], tool_result_max_chars=30_000)
        try:
            await manager.initialize_services()
            plugins = manager._runner.app.plugins
        finally:
            await manager.cleanup()

        (cap,) = [p for p in plugins if isinstance(p, ToolResultCapPlugin)]
        (progress,) = [p for p in plugins if isinstance(p, TaskProgressPlugin)]
        assert plugins.index(cap) > plugins.index(progress)
        assert (cap.max_chars, cap.folder) == (30_000, folder())

    async def test_a_native_adk_app_has_no_cap(self, project, tmp_path):
        # A plugin's value would skip the app's own after_tool_callbacks, and
        # a native app is otherwise left as ADK builds it.
        manager = GoogleADKWorkflowManager(
            agent_configs=[],
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-5"),
            adk_config_path=_native_yaml(tmp_path, "claude-sonnet-4-5"),
        )
        try:
            await manager.initialize_services()
            plugins = manager._runner.app.plugins
        finally:
            await manager.cleanup()

        assert not any(isinstance(p, ToolResultCapPlugin) for p in plugins)


def plugin(max_chars: int = 20_000) -> ToolResultCapPlugin:
    return ToolResultCapPlugin(max_chars, folder())


async def after(cap: ToolResultCapPlugin, tool, result, **args):
    return await cap.after_tool_callback(
        tool=tool, tool_args=args, tool_context=None, result=result
    )


def read_file_tool() -> FunctionTool:
    """The framework's ``read_file`` as ADK runs it."""
    return FunctionTool(read_file)


class TestThePlugin:
    async def test_a_result_that_fits_is_left_alone(self, project):
        assert await after(plugin(), read_file_tool(), {"success": True, "content": "x"}) is None
        assert not folder().exists()

    async def test_a_string_result_is_cut_and_wrapped(self, project):
        capped = await after(plugin(), read_file_tool(), "x" * 50_000)

        body, _, note = capped["result"].rpartition("\n\n")
        assert list(capped) == ["result"]
        assert CUT.search(body)
        assert SAVED_NOTE.match(note)

    async def test_the_note_is_in_the_content_claude_is_sent(self, project):
        capped = await after(
            plugin(), read_file_tool(), {"success": True, "content": LOG, "path": "/p/big.log"}
        )

        assert capped["content"].endswith("\n\n" + capped["truncated"])
        assert capped["path"] == "/p/big.log"

    @pytest.mark.parametrize(
        "result",
        [
            {"success": True, "content": LOG, "path": "/p/big.log", "size": 1},
            {"content": [{"type": "text", "text": LOG}], "isError": False},
            LOG,
            [LOG, "short"],
            {"success": True, "state": "succeeded", "result": {"stdout": LOG, "exit_code": 0}},
            {"success": True, "stdout": LOG, "stderr": "", "result": "42", "error": None},
            {"success": True, "rows": [{"id": i, "name": f"row {i}"} for i in range(8_000)]},
        ],
        ids=["read_file", "mcp", "string", "list", "job_result", "execute_python", "fallback"],
    )
    async def test_claude_and_openai_responses_read_the_note(self, project, result):
        # Each adapter sends a part of the result: "content", "result", or all.
        capped = await after(plugin(), read_file_tool(), result)

        for sent in (claude_reads(capped), _serialize_json_value(capped)):
            assert NOTE_IN_TEXT.search(sent), sent[-300:]

    async def test_a_dict_result_gets_the_note_inside(self, project):
        capped = await after(
            plugin(), read_file_tool(), {"success": True, "result": {"stdout": LOG, "code": 0}}
        )

        assert capped["result"]["truncated"] == capped["truncated"]
        assert capped["result"]["code"] == 0

    async def test_an_mcp_tools_result(self, project):
        result = {"content": [{"type": "text", "text": LOG}], "isError": False}

        capped = await after(plugin(), _MCPTool("docs__fetch_page"), result)

        first, last = capped["content"]
        assert CUT.search(first["text"])
        assert last == {"type": "text", "text": capped["truncated"]}
        assert capped["isError"] is False
        assert SAVED_NOTE.match(capped["truncated"])["name"].startswith("docs__fetch_page-")

    async def test_a_skill_tools_result(self, project, tmp_path):
        toolset = make_skill_toolset(SkillStore().resolve([str(_make_skill(tmp_path))]))
        (load_skill,) = [t for t in toolset._tools if t.name == "load_skill"]
        result = {"skill_name": "pdf-tools", "instructions": LOG, "frontmatter": {}}

        capped = await after(plugin(), load_skill, result)

        assert CUT.search(capped["instructions"])
        assert capped["skill_name"] == "pdf-tools"
        assert SAVED_NOTE.match(capped["truncated"])["name"].startswith("load_skill-")

    async def test_a_read_file_of_a_saved_output_is_not_saved_again(self, project):
        cap = plugin()
        first = await after(cap, read_file_tool(), {"content": LOG})
        saved = saved_path(first["truncated"])

        again = await after(cap, read_file_tool(), read_file(str(saved)), path=str(saved))

        assert saved_path(again["truncated"]) == saved
        assert saved_outputs(folder()) == [saved]

    async def test_only_the_frameworks_read_file_reads_one_back(self, project):
        def read_file(path: str) -> dict:  # noqa: F811 - a look-alike, on purpose
            """Not the framework's."""
            return {"content": Path(path).read_text()}

        cap = plugin()
        first = await after(cap, read_file_tool(), {"content": LOG})
        saved = saved_path(first["truncated"])

        await after(cap, FunctionTool(read_file), read_file(str(saved)), path=str(saved))

        assert len(saved_outputs(folder())) == 2

    async def test_a_cap_that_fails_leaves_the_result_alone(self, project, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("bug")

        log = _Recorder()
        monkeypatch.setattr(tool_result_cap_plugin, "cap_tool_result", broken)
        monkeypatch.setattr(tool_result_cap_plugin, "logger", log)

        assert await after(plugin(), read_file_tool(), {"content": LOG}) is None
        # The warning, without a traceback over the UI; the traceback at debug.
        (warning, debug) = log.calls
        assert warning[:2] == ("warning", "tool_result_cap_failed")
        assert warning[2]["error"] == "bug" and "exc_info" not in warning[2]
        assert debug[:2] == ("debug", "tool_result_cap_failed")
        assert isinstance(debug[2]["exc_info"], RuntimeError)


class _Approver:
    def __init__(self) -> None:
        self.asked: list[str] = []

    async def request_user_input(self, request) -> str:
        self.asked.append(request.tool_name)
        return ALLOW_ONCE_CHOICE


async def test_reading_a_saved_output_back_needs_no_prompt(project):
    # The builtin rule allows reads inside the project: the note's relative
    # path, and the same file named absolutely.
    approver = _Approver()
    settings = BaseSettings()
    engine = PermissionEngine(
        settings=settings,
        workflow=approver,
        ctx=PermissionContext(workdir=Path.cwd(), home=Path.home(), app_name=settings.app_name),
    )
    capabilities = get_registry().get("read_file").capabilities
    relative = "./.agentic_cli/tool-output/read_file-0123456789ab.txt"

    for path in (relative, str(Path.cwd() / relative)):
        assert (await engine.check("read_file", capabilities, {"path": path})).allowed

    assert approver.asked == []
