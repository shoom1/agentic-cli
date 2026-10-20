"""Agent Skills on Deep Agents (``tools/skills/langchain_tools.py``).

An agent's ``skills`` resolve through ``SkillStore`` as on ADK (a path, or a
name under ``skills_dirs``) and become three LangChain tools with the names,
arguments, results and error texts of the ``SkillToolset`` the ADK manager
builds: ``list_skills``, ``load_skill`` and ``load_skill_resource``, never
``run_skill_script``. The agent's system prompt gets the skills instruction
ADK adds. Resources are read from the loaded skill, and each tool object is
bound to the definition ADK's skill tools are bound to, so the gate runs it
without the engine and denies an object it did not build.

Every comparison runs ADK's own toolset on the same skill directory, through
ADK's runner on a recording model. Agent models are scripted; nothing reaches
a provider.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import textwrap
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("deepagents")

from google.adk import Runner  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.apps import App  # noqa: E402
from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.adk.tools.skill_toolset import SkillToolset  # noqa: E402
from google.genai import types  # noqa: E402
from langchain.agents.middleware import ToolCallRequest  # noqa: E402
from langchain_core.messages import ToolMessage  # noqa: E402
from langchain_core.tools import BaseTool, StructuredTool  # noqa: E402
from pydantic import Field  # noqa: E402

from agentic_cli.tools.registry import get_registry, identify_tool  # noqa: E402
from agentic_cli.tools.skills import SkillStore, make_skill_toolset  # noqa: E402
from agentic_cli.tools.skills.langchain_tools import (  # noqa: E402
    SKILL_TOOL_NAMES,
    SKILLS_INSTRUCTION,
    make_skill_tools,
)
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig, AgentGraphError  # noqa: E402
from agentic_cli.workflow.deepagents import manager as manager_module  # noqa: E402
from agentic_cli.workflow.deepagents import mcp_tools  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.middleware import PermissionMiddleware  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.mcp import MCPServerConfig  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.service_registry import (  # noqa: E402
    PERMISSION_ENGINE,
    set_service_registry,
)
from tests.conftest import MockContext  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedChat, ScriptedModels, ai, task  # noqa: E402

#: The skills the research demo ships (``report-writer``, kebab-case as ADK requires).
DEMO_SKILLS = Path(__file__).resolve().parents[3] / "examples" / "research_demo" / "skills"

#: Not UTF-8, so ADK's loader keeps the file as bytes.
PNG = b"\x89PNG\r\n\x1a\n\x00\x01\x02\x03"


def _make_skill(parent: Path, name: str, description: str) -> Path:
    """A skill folder with a reference, two assets (one binary) and a script."""
    skill_dir = parent / name
    for sub in ("references", "assets", "scripts"):
        (skill_dir / sub).mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: {description}
            ---
            # {name}
            Follow these instructions.
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "references" / "guide.md").write_text("# Guide\nRead me first.\n")
    (skill_dir / "assets" / "template.txt").write_text("Dear reader,\n")
    (skill_dir / "assets" / "logo.png").write_bytes(PNG)
    (skill_dir / "scripts" / "run.py").write_text("print('hi')\n")
    return skill_dir


@pytest.fixture
def skills_dir(tmp_path) -> Path:
    """Two skills in one folder; one description needs escaping in XML."""
    root = tmp_path / "skills"
    _make_skill(root, "pdf-tools", "Work with PDFs.")
    _make_skill(root, "sql-helper", "Write SQL <safely> & check it.")
    return root


def _skills(skills_dir: Path) -> list:
    return SkillStore([str(skills_dir)]).resolve(["pdf-tools", "sql-helper"])


# --- ADK's own flow -------------------------------------------------------------------


class AdkRecorder(BaseLlm):
    """An ADK model that makes the scripted calls, one a step, then answers.

    It records every request ADK's flow builds.
    """

    model: str = "recorder"
    calls: list[Any] = Field(default_factory=list)
    requests: list[Any] = Field(default_factory=list)

    async def generate_content_async(self, llm_request, stream: bool = False):
        self.requests.append(llm_request)
        step = len(self.requests) - 1
        if step < len(self.calls):
            name, args = self.calls[step]
            part = types.Part(
                function_call=types.FunctionCall(id=f"c{step}", name=name, args=args)
            )
        else:
            part = types.Part.from_text(text="Done.")
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


class AdkFlow:
    """One ADK agent ("writer", "Write.") with the toolset the ADK manager builds."""

    def __init__(self, skills: list) -> None:
        self.model = AdkRecorder()
        agent = LlmAgent(
            name="writer",
            model=self.model,
            instruction="Write.",
            tools=[make_skill_toolset(skills)] if skills else [],
        )
        self.runner = Runner(
            app=App(name="skills_test", root_agent=agent),
            session_service=InMemorySessionService(),
        )
        self.session_id: str | None = None

    async def turn(self, *calls: tuple[str, dict]) -> list:
        """Run one turn making ``calls``; return the requests ADK built."""
        if self.session_id is None:
            session = await self.runner.session_service.create_session(
                app_name="skills_test", user_id="u"
            )
            self.session_id = session.id
        self.model.calls = list(calls)
        self.model.requests = []
        message = types.Content(role="user", parts=[types.Part.from_text(text="Go.")])
        async for _ in self.runner.run_async(
            user_id="u", session_id=self.session_id, new_message=message
        ):
            pass
        return list(self.model.requests)


def _adk_responses(request, count: int) -> list[dict]:
    """The last ``count`` function responses a request carries."""
    responses = [
        part.function_response.response
        for content in request.contents
        for part in content.parts or []
        if part.function_response
    ]
    return responses[-count:]


def _adk_declarations(request) -> list[tuple[str, str, dict]]:
    return [
        (decl.name, decl.description, decl.parameters_json_schema)
        for tool in request.config.tools or []
        for decl in tool.function_declarations or []
    ]


# --- the Deep Agents manager ----------------------------------------------------------


@pytest.fixture
def ctx(tmp_path, monkeypatch, skills_dir):
    """A hermetic cwd and HOME, an in-memory store, and ``skills_dirs`` set."""
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    with MockContext(
        google_api_key="test-key", session_store="memory", skills_dirs=[str(skills_dir)]
    ) as context:
        yield context


def scripted(monkeypatch, **scripts) -> ScriptedModels:
    models = ScriptedModels(**scripts)
    monkeypatch.setattr(manager_module, "build_chat_model", models)
    return models


def _writer(**fields) -> AgentConfig:
    fields.setdefault("skills", ["pdf-tools", "sql-helper"])
    fields.setdefault("include_state_tools", False)
    return AgentConfig(name="writer", prompt="Write.", **fields)


def _manager(settings, configs) -> DeepAgentsWorkflowManager:
    return DeepAgentsWorkflowManager(
        agent_configs=configs, settings=settings, model="gemini-2.5-flash"
    )


def _steps(*calls: tuple[str, dict]) -> list:
    """A turn's script: each call in its own step, then the answer."""
    return [*(ai(calls=[call]) for call in calls), ai("Done.")]


class Asks:
    """The input callback; any question fails the test."""

    def __init__(self) -> None:
        self.asked: list = []

    async def __call__(self, request) -> str:
        self.asked.append(request)
        raise AssertionError(f"asked: {request}")


async def turn(manager, text: str = "Go.") -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


def of(events, kind: EventType) -> list:
    return [event for event in events if event.type is kind]


def _tool_messages(chat: ScriptedChat, call: int, count: int) -> list[ToolMessage]:
    """The last ``count`` tool results request ``call`` sent."""
    return [m for m in chat.conversation(call) if isinstance(m, ToolMessage)][-count:]


def _as_adk_response(message: ToolMessage) -> dict:
    """A result as ADK's function response holds it: a dict, or ``{"result": ...}``."""
    try:
        parsed = json.loads(message.content)
    except (TypeError, ValueError):
        parsed = None
    return parsed if isinstance(parsed, dict) else {"result": message.content}


def _tool(manager, name: str) -> BaseTool:
    """The tool object the root agent's tool node runs under ``name``."""
    return manager._root.nodes["tools"].bound.tools_by_name[name]


# --- the same as on ADK ---------------------------------------------------------------

#: One turn's calls, every answer a skill tool gives; the second miss is fatal.
CALLS = [
    ("list_skills", {}),
    ("load_skill", {"skill_name": "pdf-tools"}),
    ("load_skill", {"skill_name": "sql-helper", "version": 2}),  # an unknown argument
    ("load_skill", {}),
    ("load_skill", {"skill_name": "pdf_tools"}),
    ("load_skill_resource", {"skill_name": "pdf-tools", "file_path": "references/guide.md"}),
    ("load_skill_resource", {"skill_name": "sql-helper", "file_path": "assets/template.txt"}),
    ("load_skill_resource", {"skill_name": "pdf-tools", "file_path": "scripts/run.py"}),
    ("load_skill_resource", {"skill_name": "pdf-tools"}),
    ("load_skill_resource", {}),
    ("load_skill_resource", {"skill_name": "nope", "file_path": "references/guide.md"}),
    ("load_skill_resource", {"skill_name": "pdf-tools", "file_path": "SKILL.md"}),
    ("load_skill_resource", {"skill_name": "pdf-tools", "file_path": "references/gone.md"}),
    ("load_skill_resource", {"skill_name": "sql-helper", "file_path": "scripts/gone.py"}),
]

MISS = ("load_skill_resource", {"skill_name": "pdf-tools", "file_path": "assets/gone.txt"})


class TestAsOnADK:
    async def test_an_agent_is_offered_what_adk_offers(self, ctx, monkeypatch, skills_dir):
        adk_with = (await AdkFlow(_skills(skills_dir)).turn())[0]
        adk_without = (await AdkFlow([]).turn())[0]
        models = scripted(monkeypatch, writer=[ai("Done.")])
        manager = _manager(ctx.settings, [_writer()])
        try:
            await turn(manager)
        finally:
            await manager.cleanup()
        chat = models.chats["writer"]

        # The same tools, in ADK's order: names, descriptions and arguments.
        assert [
            (schema["name"], schema["description"], schema["parameters"])
            for schema in chat.bound[0]
        ] == _adk_declarations(adk_with)
        assert chat.tool_names() == list(SKILL_TOOL_NAMES)
        assert "run_skill_script" not in chat.tool_names()
        # The same instruction, appended to the prompt as ADK appends it.
        added_by_adk = adk_with.config.system_instruction.removeprefix(
            adk_without.config.system_instruction + "\n\n"
        )
        assert added_by_adk == SKILLS_INSTRUCTION
        assert chat.system_prompt(0) == "Write.\n\n" + SKILLS_INSTRUCTION

    async def test_every_answer_is_adks(self, ctx, monkeypatch, skills_dir):
        adk_request = (await AdkFlow(_skills(skills_dir)).turn(*CALLS))[-1]
        models = scripted(monkeypatch, writer=_steps(*CALLS))
        manager = _manager(ctx.settings, [_writer()])
        manager.set_input_callback(Asks())
        try:
            events = await turn(manager)
        finally:
            await manager.cleanup()

        expected = _adk_responses(adk_request, len(CALLS))
        messages = _tool_messages(models.chats["writer"], len(CALLS), len(CALLS))
        assert [_as_adk_response(m) for m in messages] == expected
        # An ADK error result is a failed call, in the request and in the events.
        failed = ["error" in response for response in expected]
        assert [m.status == "error" for m in messages] == failed
        results = of(events, EventType.TOOL_RESULT)
        assert [not e.metadata["success"] for e in results] == failed
        assert {e.metadata["agent"] for e in results} == {"writer"}
        # The last two calls find nothing: a turn's second miss is fatal.
        assert [r.get("error_code") for r in expected[-2:]] == [
            "RESOURCE_NOT_FOUND",
            "RESOURCE_NOT_FOUND_FATAL",
        ]

    async def test_a_new_turn_counts_its_own_misses(self, ctx, monkeypatch, skills_dir):
        adk = AdkFlow(_skills(skills_dir))
        adk_turns = [await adk.turn(MISS, MISS), await adk.turn(MISS)]
        models = scripted(monkeypatch, writer=[*_steps(MISS, MISS), *_steps(MISS)])
        manager = _manager(ctx.settings, [_writer()])
        try:
            await turn(manager)
            await turn(manager, "Again.")
        finally:
            await manager.cleanup()
        chat = models.chats["writer"]

        assert [_as_adk_response(m) for m in _tool_messages(chat, 2, 2)] == _adk_responses(
            adk_turns[0][-1], 2
        )
        (second,) = _tool_messages(chat, 4, 1)
        assert [_as_adk_response(second)] == _adk_responses(adk_turns[1][-1], 1)
        assert _as_adk_response(second)["error_code"] == "RESOURCE_NOT_FOUND"

    async def test_a_binary_resource_reaches_the_model_as_on_adk(
        self, ctx, monkeypatch, skills_dir
    ):
        call = ("load_skill_resource", {"skill_name": "pdf-tools", "file_path": "assets/logo.png"})
        adk_request = (await AdkFlow(_skills(skills_dir)).turn(call))[-1]
        models = scripted(monkeypatch, writer=_steps(call))
        manager = _manager(ctx.settings, [_writer()])
        try:
            events = await turn(manager)
        finally:
            await manager.cleanup()

        # ADK answers with a status, then adds the bytes to the request.
        (status,) = _adk_responses(adk_request, 1)
        injected = adk_request.contents[-1]
        text, data = injected.parts
        assert injected.role == "user"
        (message,) = _tool_messages(models.chats["writer"], 1, 1)
        status_block, text_block, data_block = message.content
        assert json.loads(status_block["text"]) == status
        assert text_block == {"type": "text", "text": text.text}
        assert data_block == {
            "type": "image",
            "base64": base64.b64encode(data.inline_data.data).decode("ascii"),
            "mime_type": data.inline_data.mime_type,
        }
        assert base64.b64decode(data_block["base64"]) == PNG
        (result,) = of(events, EventType.TOOL_RESULT)
        assert result.metadata["success"] is True


# --- the tools ------------------------------------------------------------------------


def _call(tool: BaseTool, args: dict, call_id: str = "c1") -> dict:
    return {"type": "tool_call", "id": call_id, "name": tool.name, "args": args}


class TestMakeSkillTools:
    def test_three_tools_and_the_instruction_never_the_script_tool(self, skills_dir):
        tools, instruction = make_skill_tools(_skills(skills_dir))

        assert [tool.name for tool in tools] == [
            "list_skills",
            "load_skill",
            "load_skill_resource",
        ]
        assert tuple(tool.name for tool in tools) == SKILL_TOOL_NAMES
        assert instruction == SKILLS_INSTRUCTION
        # Both skills have scripts, and still there is nothing to run them with.
        assert all(skill.resources.list_scripts() for skill in _skills(skills_dir))

    async def test_resources_are_read_from_the_loaded_skill(self, skills_dir):
        tools, _ = make_skill_tools(_skills(skills_dir))
        shutil.rmtree(skills_dir)
        _, load_skill, load_skill_resource = tools

        loaded = await load_skill.ainvoke(_call(load_skill, {"skill_name": "pdf-tools"}))
        resource = await load_skill_resource.ainvoke(_call(
            load_skill_resource, {"skill_name": "pdf-tools", "file_path": "references/guide.md"}
        ))

        assert json.loads(loaded.content)["instructions"] == "# pdf-tools\nFollow these instructions."
        assert json.loads(resource.content)["content"] == "# Guide\nRead me first.\n"

    def test_duplicate_skill_names_raise_as_on_adk(self, skills_dir):
        skill = _skills(skills_dir)[0]

        with pytest.raises(ValueError) as ours:
            make_skill_tools([skill, skill])
        with pytest.raises(ValueError) as adks:
            SkillToolset(skills=[skill, skill])

        assert str(ours.value) == str(adks.value) == "Duplicate skill name 'pdf-tools'."

    def test_no_skills_no_tools(self):
        assert make_skill_tools([]) == ([], "")

    def test_the_package_loads_without_langchain(self):
        """``agentic_cli.tools.skills`` is core; only ``make_skill_tools`` needs the extra."""
        code = (
            "import sys; import agentic_cli.tools.skills as s; "
            "print('langchain_core' in sys.modules); "
            "s.make_skill_tools; print('langchain_core' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.split() == ["False", "True"]


class TestIdentity:
    def test_each_tool_is_bound_to_the_definition_adks_is_bound_to(self, skills_dir):
        tools, _ = make_skill_tools(_skills(skills_dir))
        adk_tools = make_skill_toolset(_skills(skills_dir))._tools

        for ours, adks in zip(tools, adk_tools, strict=True):
            definition = get_registry().get(ours.name)
            assert identify_tool(ours) is definition
            assert identify_tool(adks) is definition
            assert definition.capabilities is EXEMPT

    async def test_the_gate_runs_them_without_the_engine_and_denies_any_other(
        self, skills_dir
    ):
        tools, _ = make_skill_tools(_skills(skills_dir))
        load_skill = tools[1]
        impostor = StructuredTool(
            name="load_skill",
            description=load_skill.description,
            args_schema=load_skill.args_schema,
            coroutine=load_skill.coroutine,
        )
        engine = MagicMock()
        engine.check = AsyncMock()

        async def gate(tool: BaseTool) -> ToolMessage:
            request = ToolCallRequest(
                tool_call=_call(tool, {"skill_name": "pdf-tools"}),
                tool=tool,
                state={},
                runtime=SimpleNamespace(tools=[tool]),
            )
            token = set_service_registry({PERMISSION_ENGINE: engine})
            try:
                return await PermissionMiddleware("writer").awrap_tool_call(
                    request, lambda r: r.tool.ainvoke(r.tool_call)
                )
            finally:
                token.var.reset(token)

        ran = await gate(load_skill)
        copied = await gate(load_skill.model_copy())
        forged = await gate(impostor)

        engine.check.assert_not_awaited()
        assert json.loads(ran.content)["skill_name"] == "pdf-tools"
        for denied in (copied, forged):
            assert denied.status == "error"
            assert json.loads(denied.content)["error"].startswith("Permission denied: ")


# --- in the manager -------------------------------------------------------------------


class TestInTheManager:
    async def test_skills_resolve_by_name_under_skills_dirs_or_by_path(
        self, ctx, monkeypatch, skills_dir
    ):
        from agentic_cli.tools import read_file

        models = scripted(monkeypatch, writer=_steps(("load_skill", {"skill_name": "sql-helper"})))
        manager = _manager(ctx.settings, [_writer(
            tools=[read_file], skills=["pdf-tools", str(skills_dir / "sql-helper")],
        )])
        asks = Asks()
        manager.set_input_callback(asks)
        try:
            events = await turn(manager)
        finally:
            await manager.cleanup()

        chat = models.chats["writer"]
        assert chat.tool_names() == ["read_file", *SKILL_TOOL_NAMES]
        assert chat.system_prompt(0) == "Write.\n\n" + SKILLS_INSTRUCTION
        (result,) = of(events, EventType.TOOL_RESULT)
        assert result.metadata["success"] is True
        assert result.metadata["result"]["skill_name"] == "sql-helper"
        assert result.metadata["agent"] == "writer"
        assert asks.asked == []  # exempt: nobody is asked
        assert of(events, EventType.ERROR) == []

    async def test_a_sub_agents_skills_are_its_own(self, ctx, monkeypatch):
        models = scripted(
            monkeypatch,
            lead=[ai(calls=[task("writer", "Write it.")]), ai("Written.")],
            writer=_steps(("load_skill", {"skill_name": "pdf-tools"})),
        )
        manager = _manager(ctx.settings, [
            AgentConfig(name="lead", prompt="Lead.", sub_agents=["writer"],
                        include_state_tools=False),
            _writer(description="Writes."),
        ])
        try:
            events = await turn(manager)
        finally:
            await manager.cleanup()

        lead, writer = models.chats["lead"], models.chats["writer"]
        assert lead.tool_names() == ["task"]
        assert SKILLS_INSTRUCTION not in lead.system_prompt(0)
        assert writer.tool_names() == list(SKILL_TOOL_NAMES)
        assert writer.system_prompt(0) == "Write.\n\n" + SKILLS_INSTRUCTION
        loaded = next(e for e in of(events, EventType.TOOL_RESULT)
                      if e.metadata["tool_name"] == "load_skill")
        assert (loaded.metadata["agent"], loaded.metadata["success"]) == ("writer", True)

    async def test_the_demo_report_writer_skill(self, tmp_path, monkeypatch):
        (tmp_path / "home").mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        template = DEMO_SKILLS / "report-writer" / "assets" / "report_template.tex"
        models = scripted(monkeypatch, writer=_steps(
            ("load_skill", {"skill_name": "report-writer"}),
            ("load_skill_resource", {"skill_name": "report-writer",
                                     "file_path": "assets/report_template.tex"}),
        ))
        with MockContext(google_api_key="test-key", session_store="memory",
                         skills_dirs=[str(DEMO_SKILLS)]) as context:
            manager = _manager(context.settings, [_writer(skills=["report-writer"])])
            try:
                events = await turn(manager)
            finally:
                await manager.cleanup()

        loaded, resource = (_as_adk_response(m)
                            for m in _tool_messages(models.chats["writer"], 2, 2))
        assert loaded["skill_name"] == "report-writer"
        assert "compile_document" in loaded["instructions"]
        assert resource["content"] == template.read_text(encoding="utf-8")
        assert all(e.metadata["success"] for e in of(events, EventType.TOOL_RESULT))

    @pytest.mark.parametrize("make", [
        lambda root: "no-such-skill",
        lambda root: str(_make_skill(root, "pdf_tools", "Snake case.")),
    ], ids=["unknown", "snake-case-name"])
    async def test_a_skill_that_cannot_load_fails_initialization_as_on_adk(
        self, ctx, monkeypatch, tmp_path, make
    ):
        ref = make(tmp_path / "more")
        scripted(monkeypatch, writer=[])
        adk = GoogleADKWorkflowManager(
            agent_configs=[_writer(skills=[ref])], settings=ctx.settings,
            model="gemini-2.5-flash",
        )
        manager = _manager(ctx.settings, [_writer(skills=[ref])])

        try:
            with pytest.raises(ValueError) as adks:
                await adk.initialize_services()
            with pytest.raises(ValueError) as ours:
                await manager.initialize_services()
        finally:
            await adk.cleanup()

        assert type(ours.value) is type(adks.value)
        assert str(ours.value) == str(adks.value)
        # The rollback closed what the attempt opened.
        assert (manager._initialized, manager._checkpointer) == (False, None)

    async def test_an_own_tool_named_like_a_skill_tool_is_refused(self, ctx, monkeypatch):
        def load_skill(skill_name: str) -> dict:
            """An application's own skill loader."""
            return {"success": True}

        scripted(monkeypatch, writer=[])
        manager = _manager(ctx.settings, [_writer(tools=[load_skill])])

        with pytest.raises(AgentGraphError) as error:
            await manager.initialize_services()

        assert str(error.value) == (
            "Agent 'writer' has skills, whose tools are list_skills, load_skill and "
            "load_skill_resource, so its own tools cannot include 'load_skill'; "
            "rename that tool."
        )
        assert (manager._initialized, manager._checkpointer) == (False, None)

    async def test_an_mcp_tool_named_like_a_skill_tool_is_skipped(self, ctx, monkeypatch):
        """The skill tool keeps its name; the server's is the one left out."""

        async def call_tool(**arguments):
            return [{"type": "text", "text": "the server's"}], None

        _fake_mcp(monkeypatch, StructuredTool(
            name="load_skill", description="A server's tool.", args_schema={"type": "object"},
            coroutine=call_tool, response_format="content_and_artifact",
        ))
        models = scripted(monkeypatch, writer=_steps(("load_skill", {"skill_name": "pdf-tools"})))
        manager = _manager(ctx.settings, [_writer(
            mcp_servers=[MCPServerConfig(name="atlas", command="srv")],
        )])
        try:
            events = await turn(manager)
            kept = _tool(manager, "load_skill")
            assert identify_tool(kept) is get_registry().get("load_skill")
            assert manager._mcp.identify(kept) is None
        finally:
            await manager.cleanup()

        assert models.chats["writer"].tool_names() == list(SKILL_TOOL_NAMES)
        (notice,) = of(events, EventType.ERROR)
        assert notice.metadata["error_code"] == "mcp_tool_skipped"
        assert notice.content == (
            "MCP tool 'load_skill' of server 'atlas' is skipped for agent 'writer': "
            "the agent already has a tool of that name. Set tool_name_prefix on the "
            "server to rename its tools."
        )
        (result,) = of(events, EventType.TOOL_RESULT)
        assert result.metadata["result"]["skill_name"] == "pdf-tools"

    async def test_a_model_switch_rebuilds_them(self, ctx, monkeypatch):
        models = scripted(monkeypatch, writer=_steps(("load_skill", {"skill_name": "pdf-tools"})))
        manager = _manager(ctx.settings, [_writer()])
        try:
            await manager.initialize_services()
            before = _tool(manager, "load_skill")
            await manager.reinitialize(model="gemini-2.5-pro", preserve_sessions=True)
            after = _tool(manager, "load_skill")
            events = await turn(manager)
        finally:
            await manager.cleanup()

        assert after is not before
        assert identify_tool(after) is identify_tool(before) is get_registry().get("load_skill")
        assert models.chats["writer"].system_prompt(0) == "Write.\n\n" + SKILLS_INSTRUCTION
        (result,) = of(events, EventType.TOOL_RESULT)
        assert result.metadata["success"] is True


def _fake_mcp(monkeypatch, *tools: BaseTool) -> None:
    """Replace the MCP adapter's session with one that serves ``tools``."""

    class _Session:
        async def initialize(self) -> None:
            return None

    @asynccontextmanager
    async def create_session(connection):
        yield _Session()

    async def load_mcp_tools(session, **kwargs):
        return list(tools)

    monkeypatch.setattr(mcp_tools, "create_session", create_session)
    monkeypatch.setattr(mcp_tools, "load_mcp_tools", load_mcp_tools)
