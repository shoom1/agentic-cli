"""The tool result cap on Deep Agents (``ToolResultCapMiddleware``).

Every agent's tool side runs the permission gate, the cap, the tool-error
middleware and the active tool call, outermost first: a denial is short, and
an error result is capped like any other. The cap reads a ``ToolMessage``'s
content (a JSON object is cut as a dict and serialized again, other text as a
string, text blocks joined into one) and each ``ToolMessage`` of a
``Command`` (the state tools, ``task``). A ``read_file`` of a saved output
(by the tool's identity) is cut again without a second file.

Agent models are scripted (``fakes.ScriptedModels``); ADK's side is in
``tests/workflow/test_tool_result_cap.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("deepagents")

from langchain.agents.middleware import ToolCallRequest  # noqa: E402
from langchain_core.messages import ToolMessage  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agentic_cli.tools import read_file  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import manager as manager_module  # noqa: E402
from agentic_cli.workflow.deepagents import middleware as middleware_module  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.middleware import ToolResultCapMiddleware  # noqa: E402
from agentic_cli.workflow.deepagents.tools import to_langchain_tools  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.tool_output import tool_output_dir  # noqa: E402
from tests.conftest import MockContext  # noqa: E402
from tests.workflow.cap_probes import (  # noqa: E402
    CUT,
    REPORT,
    SAVED_NOTE,
    UNSAVED_NOTE,
    cap_probe_raises,
    cap_probe_report,
    saved_outputs,
    saved_path,
)
from tests.workflow.deepagents.fakes import ScriptedModels, ai, task  # noqa: E402

LOG = "".join(f"log line {i:06d}\n" for i in range(8_000))


class _Recorder:
    """Stands in for a module's structlog logger."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.calls.append((level, event, fields))

        return log


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A hermetic project (the cwd) and HOME."""
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    return tmp_path / "project"


@pytest.fixture
def ctx(project):
    with MockContext(google_api_key="test-key", session_store="memory") as context:
        yield context


@pytest.fixture
async def cleanup():
    """Clean up every manager a test hands it."""
    managers: list[DeepAgentsWorkflowManager] = []
    yield managers.append
    for manager in managers:
        await manager.cleanup()


def scripted(monkeypatch, **scripts) -> ScriptedModels:
    models = ScriptedModels(**scripts)
    monkeypatch.setattr(manager_module, "build_chat_model", models)
    return models


def solo(settings, tools: list) -> DeepAgentsWorkflowManager:
    return DeepAgentsWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="Answer.", tools=tools)],
        settings=settings,
        model="gemini-2.5-flash",
    )


async def turn(manager, text: str = "Go.") -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


def texts(events) -> list[str]:
    return [event.content for event in events if event.type is EventType.TEXT]


def folder(ctx) -> Path:
    return tool_output_dir(ctx.settings.app_name)


class TestTurns:
    async def test_the_next_request_carries_the_cut_tool_message(
        self, ctx, monkeypatch, cleanup
    ):
        models = scripted(monkeypatch, solo=[ai(calls=[("cap_probe_report", {})]), ai("Read it.")])
        manager = solo(ctx.settings, [cap_probe_report])
        cleanup(manager)

        events = await turn(manager)

        message = models.chats["solo"].conversation(1)[-1]
        assert isinstance(message, ToolMessage) and message.status == "success"
        sent = json.loads(message.content)
        assert sent["success"] is True
        assert CUT.search(sent["report"])
        assert REPORT in saved_path(sent["truncated"]).read_text()
        assert len(message.content) <= ctx.settings.tool_result_max_chars
        assert texts(events) == ["Read it."]

    async def test_a_sub_agents_long_answer_is_cut(self, ctx, monkeypatch, cleanup):
        models = scripted(
            monkeypatch,
            lead=[ai(calls=[task("helper", "Write the report.")]), ai("Done.")],
            helper=[ai(REPORT)],
        )
        manager = DeepAgentsWorkflowManager(
            agent_configs=[
                AgentConfig(name="lead", prompt="Lead.", sub_agents=["helper"]),
                AgentConfig(name="helper", prompt="Help.", description="Writes reports."),
            ],
            settings=ctx.settings,
            model="gemini-2.5-flash",
        )
        cleanup(manager)

        events = await turn(manager)

        message = models.chats["lead"].conversation(1)[-1]
        body, _, note = message.content.rpartition("\n\n")
        assert CUT.search(body)
        assert SAVED_NOTE.match(note)["name"].startswith("task-")
        # Deep Agents strips the sub-agent's answer.
        assert saved_path(note).read_text() == REPORT.rstrip()
        assert texts(events)[-1] == "Done."

    async def test_an_error_result_passes(self, ctx, monkeypatch, cleanup):
        models = scripted(monkeypatch, solo=[ai(calls=[("cap_probe_raises", {})]), ai("Sorry.")])
        manager = solo(ctx.settings, [cap_probe_raises])
        cleanup(manager)

        await turn(manager)

        message = models.chats["solo"].conversation(1)[-1]
        assert message.status == "error"
        assert json.loads(message.content) == {
            "success": False,
            "error": "cap_probe_raises failed: RuntimeError: probe failed",
        }
        assert not folder(ctx).exists()

    async def test_a_read_file_of_a_saved_output_is_cut_without_a_new_file(
        self, ctx, project, monkeypatch, cleanup
    ):
        (project / "big.log").write_text(LOG)
        models = scripted(
            monkeypatch, solo=[ai(calls=[("read_file", {"path": "big.log"})]), ai("Read.")]
        )
        manager = solo(ctx.settings, [read_file])
        cleanup(manager)

        await turn(manager)
        (saved,) = saved_outputs(folder(ctx))
        models.chats["solo"].script += [
            ai(calls=[("read_file", {"path": str(saved)})]), ai("Again."),
        ]
        events = await turn(manager, "Read it all.")

        sent = json.loads(models.chats["solo"].conversation(3)[-1].content)
        assert CUT.search(sent["content"])
        assert saved_path(sent["truncated"]) == saved
        assert saved_outputs(folder(ctx)) == [saved]
        assert texts(events) == ["Again."]

    @pytest.mark.parametrize("project_is", ["read-only", "symlinked"])
    async def test_a_project_the_cap_cannot_save_in(
        self, ctx, project, monkeypatch, cleanup, project_is
    ):
        if project_is == "read-only" and os.geteuid() == 0:
            pytest.skip("root writes into a read-only folder")
        elsewhere = project.parent / "elsewhere"
        elsewhere.mkdir()
        if project_is == "symlinked":
            (project / f".{ctx.settings.app_name}").symlink_to(elsewhere, target_is_directory=True)
        else:
            project.chmod(0o500)
        models = scripted(monkeypatch, solo=[ai(calls=[("cap_probe_report", {})]), ai("Read it.")])
        manager = solo(ctx.settings, [cap_probe_report])
        cleanup(manager)
        try:
            events = await turn(manager)
        finally:
            project.chmod(0o700)

        sent = json.loads(models.chats["solo"].conversation(1)[-1].content)
        assert UNSAVED_NOTE.match(sent["truncated"])
        assert CUT.search(sent["report"])
        assert texts(events) == ["Read it."]
        assert list(elsewhere.iterdir()) == []
        assert [p.name for p in project.iterdir()] == (
            [f".{ctx.settings.app_name}"] if project_is == "symlinked" else []
        )


def cap(max_chars: int = 20_000) -> ToolResultCapMiddleware:
    return ToolResultCapMiddleware(max_chars, tool_output_dir("app"))


def request(tool=None, name: str = "probe", **args) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "call-1", "type": "tool_call"},
        tool=tool,
        state={},
        runtime=SimpleNamespace(tools=[]),
    )


async def run(middleware, call: ToolCallRequest, result):
    async def handler(_):
        return result

    return await middleware.awrap_tool_call(call, handler)


def message(content, **fields) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="call-1", name="probe", **fields)


@pytest.fixture
def here(project):
    """The default folder's app is ``app`` in these tests."""
    return tool_output_dir("app")


class TestTheMiddleware:
    async def test_a_message_that_fits_is_passed_as_it_is(self, here):
        result = message(json.dumps({"success": True}))

        assert await run(cap(), request(), result) is result
        assert not here.exists()

    async def test_a_json_object_is_cut_as_a_dict(self, here):
        result = message(json.dumps({"success": True, "out": LOG, "truncated": False}))

        capped = await run(cap(), request(), result)

        sent = json.loads(capped.content)
        assert sent["success"] is True
        assert CUT.search(sent["out"])
        assert sent["truncated"].startswith("Cut: ")
        assert (capped.tool_call_id, capped.name, capped.status) == ("call-1", "probe", "success")
        assert len(capped.content) <= 20_000

    @pytest.mark.parametrize("text", [
        # 9,000 CJK characters: 9,000 as JSON measures them, 54,000 as json.dumps writes them.
        json.dumps({"success": True, "text": "数据" * 4_500}),
        # Rows whose compact JSON fits the limit, and whose pretty-printed JSON does not.
        json.dumps({"rows": [{"id": i, "ok": True} for i in range(600)]}, indent=2),
    ], ids=["ensure_ascii", "indent"])
    async def test_a_json_object_is_sent_as_the_cap_measured_it(self, here, text):
        # A tool's own json.dumps can be several times the JSON the cap
        # measures: what the model is sent is that JSON, never the original.
        assert len(text) > 20_000

        capped = await run(cap(), request(), message(text))

        assert len(capped.content) <= 20_000
        assert json.loads(capped.content) == json.loads(text)  # nothing needed cutting
        assert not here.exists()

    async def test_other_text_is_cut_as_a_string(self, here):
        capped = await run(cap(), request(), message(LOG))

        body, _, note = capped.content.rpartition("\n\n")
        assert CUT.search(body) and note.startswith("Cut: ")

    async def test_a_json_array_is_cut_as_text(self, here):
        capped = await run(cap(), request(), message(json.dumps([LOG])))

        body, _, note = capped.content.rpartition("\n\n")
        assert body.startswith('["log line 000000')
        assert note.startswith("Cut: ")

    async def test_text_blocks_are_joined_into_the_first_and_other_blocks_kept(self, here):
        # The joined text stands where the first text block was, with its
        # other keys; an image between two text blocks now follows the text.
        image = {"type": "image", "base64": "iVBORw0KGgo=", "mime_type": "image/png"}
        first = {"type": "text", "text": LOG[:60_000], "id": "b1",
                 "cache_control": {"type": "ephemeral"}}
        result = message([first, image, {"type": "text", "text": LOG[60_000:]}])

        capped = await run(cap(), request(), result)

        text, kept = capped.content
        assert kept == image
        body, _, note = text["text"].rpartition("\n\n")
        assert {k: v for k, v in text.items() if k != "text"} == {
            "type": "text", "id": "b1", "cache_control": {"type": "ephemeral"},
        }
        assert body.startswith("log line 000000\n") and body.endswith("log line 007999\n")
        assert LOG[:60_000] + "\n" + LOG[60_000:] == saved_path_in(note).read_text()

    async def test_an_error_message_keeps_its_status(self, here):
        result = message(json.dumps({"success": False, "error": LOG}), status="error")

        capped = await run(cap(), request(), result)

        assert capped.status == "error"
        assert CUT.search(json.loads(capped.content)["error"])

    async def test_each_message_of_a_command_is_cut(self, here):
        result = Command(update={"messages": [message(LOG)], "plan": "the plan"})

        capped = await run(cap(), request(), result)

        assert isinstance(capped, Command)
        assert capped.update["plan"] == "the plan"
        (sent,) = capped.update["messages"]
        assert CUT.search(sent.content)
        assert result.update["messages"][0].content == LOG  # the original is left alone

    async def test_the_sync_path(self, here):
        capped = cap().wrap_tool_call(request(), lambda _: message(LOG))

        assert CUT.search(capped.content)

    async def test_a_read_file_of_a_saved_output_is_not_saved_again(self, here):
        middleware = cap()
        first = await run(middleware, request(), message(LOG))
        saved = saved_path_in(first.content.rpartition("\n\n")[2])
        (tool,) = to_langchain_tools([read_file])
        reread = message(json.dumps(read_file(str(saved)), ensure_ascii=False))

        again = await run(middleware, request(tool, "read_file", path=str(saved)), reread)

        assert saved_path_in(json.loads(again.content)["truncated"]) == saved
        assert saved_outputs(here) == [saved]

    async def test_only_the_frameworks_read_file_reads_one_back(self, here):
        middleware = cap()
        first = await run(middleware, request(), message(LOG))
        saved = saved_path_in(first.content.rpartition("\n\n")[2])
        lookalike = StructuredTool.from_function(
            func=lambda path: Path(path).read_text(), name="read_file", description="Not ours."
        )

        await run(middleware, request(lookalike, "read_file", path=str(saved)), message(LOG))

        assert len(saved_outputs(here)) == 2

    async def test_a_cap_that_fails_leaves_the_result_alone(self, here, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("bug")

        log = _Recorder()
        monkeypatch.setattr(middleware_module, "cap_tool_result", broken)
        monkeypatch.setattr(middleware_module, "logger", log)
        result = message(LOG)

        assert await run(cap(), request(), result) is result
        # The warning, without a traceback over the UI; the traceback at debug.
        (warning, debug) = log.calls
        assert warning[:2] == ("warning", "tool_result_cap_failed")
        assert warning[2]["error"] == "bug" and "exc_info" not in warning[2]
        assert debug[:2] == ("debug", "tool_result_cap_failed")
        assert isinstance(debug[2]["exc_info"], RuntimeError)


def saved_path_in(note: str) -> Path:
    """The saved output a note in the ``app`` folder points at."""
    path = note.partition("Full output: ")[2].partition("; read it")[0]
    assert path.startswith("./.app/tool-output/"), note
    return Path.cwd() / path
