"""ADK delegates to a sub-agent and gets its answer back, in the same turn.

An agent built from an ``AgentConfig`` that is not the root runs in ADK's
``single_turn`` mode, with transfers to its parent and peers off. Its parent
is offered it as a tool named after it (``<agent>(request=...)``, ADK's
private ``_SingleTurnAgentTool``), calls it, and reads its answer as the
tool's result, ``{"result": <its last text>}``; the root then answers the
user, and the next turn starts at the root again. No agent built from a
config is offered ``transfer_to_agent``. A ``fresh`` sub-agent sees only the
request; an ``inherit`` one also sees its parent's conversation, which ADK
gives a ``single_turn`` agent only when ``include_contents`` is passed
explicitly.

ADK builds the delegation tool in the parent's constructor, and it calls no
``func`` the permission plugin could verify, so the manager binds each one to
the exempt delegation declaration (``tools/_core/delegation.py``) right after
the parent is built: only the exact ``_SingleTurnAgentTool`` ADK made for a
child the framework built. Anything else that looks like one is denied. The
sub-agent's own tool calls go through the same gate.

A sub-agent's long-running tool is an ordinary call there: ADK pauses the
whole invocation on a pending long-running call, delegation included, so
only the root's are wrapped. A job a sub-agent started is told to the root
when it finishes, in the notice Deep Agents sends. A sub-agent without a
description is offered as "The <name> agent.", as on Deep Agents.

A native ADK YAML app (``adk_config_path``) is built by ADK and keeps
transferring (``tests/workflow/test_adk_transfer_tool.py``).

Turns run the real manager on ``FakeAPI`` (Claude); the private ADK pieces
this rests on have contract tests at the end.
"""

from __future__ import annotations

import dataclasses
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import pytest

pytest.importorskip("google.adk")

from anthropic import AsyncAnthropic  # noqa: E402
from google.adk import Runner  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.apps import App  # noqa: E402
from google.adk.events import Event  # noqa: E402
from google.adk.models.anthropic_llm import AnthropicLlm  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.adk.tools import BaseTool, LongRunningFunctionTool  # noqa: E402
from google.adk.tools.agent_tool import AgentTool, _SingleTurnAgentTool  # noqa: E402
from google.adk.tools.skill_toolset import SkillToolset  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.cli.usage_tracker import UsageTracker  # noqa: E402
from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools._core.delegation import TASK_TOOL  # noqa: E402
from agentic_cli.tools.jobs import JobState  # noqa: E402
from agentic_cli.tools.jobs.resume import JOB_NOTICE_HEADER, job_result_payload  # noqa: E402
from agentic_cli.tools.registry import (  # noqa: E402
    ToolCategory,
    get_registry,
    identify_tool,
    register_tool,
)
from agentic_cli.tools.skills import SkillStore  # noqa: E402
from agentic_cli.workflow.adk import delegation  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.adk.permission_plugin import _UNVERIFIED, PermissionPlugin  # noqa: E402
from agentic_cli.workflow.config import AgentConfig, AgentGraphError  # noqa: E402
from agentic_cli.workflow.context import context_budget  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.permissions.prompt import DENY_CHOICE  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message, http_library  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402
from tests.workflow.test_skills import _make_skill  # noqa: E402

BRIEF = "Find X. Report the number."
TURNS = ["My name is Ada and the secret code is 7.", "Please have the researcher find X.",
         "Thanks. What next?"]


def says(text: str) -> dict:
    return anthropic_message({"type": "text", "text": text})


def calls(name: str, call_id: str, **args) -> dict:
    message = anthropic_message({"type": "tool_use", "id": call_id, "name": name, "input": args})
    message["stop_reason"] = "tool_use"
    return message


#: The call id each start of the probe job was given.
STARTED: list[str] = []


@register_tool(category=ToolCategory.EXECUTION, capabilities=EXEMPT, long_running=True,
               description="Start a background job (an exempt test tool).")
def delegation_probe_start_job(command: str = "", tool_context=None) -> dict:
    """Start a job and return its id at once, as a job starter does."""
    STARTED.append(getattr(tool_context, "function_call_id", None))
    return {"success": True, "job_id": "j1", "state": "running"}


class _Recorder:
    """Stands in for a module's structlog logger."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.calls.append((level, event, fields))

        return log


#: Each agent's system prompt begins with its prompt.
PROMPTS = {"coordinator": "You coordinate.", "researcher": "You research.",
           "middle": "You plan.", "leaf": "You look up."}


def system_text(body: dict) -> str:
    system = body.get("system") or ""
    if isinstance(system, list):
        return "\n".join(block.get("text", "") for block in system)
    return system


def tool_names(body: dict) -> list[str]:
    return [tool["name"] for tool in body.get("tools", [])]


def lines(body: dict) -> list[str]:
    """Each message of a Claude request as ``"<role>: <block>"``, block by block."""
    out: list[str] = []
    for message in body["messages"]:
        content = message["content"]
        if isinstance(content, str):
            out.append(f"{message['role']}: {content}")
            continue
        for block in content:
            if block["type"] == "text":
                out.append(f"{message['role']}: {block['text']}")
            elif block["type"] == "tool_use":
                out.append(f"{message['role']}: called {block['name']}({json.dumps(block['input'])})")
            elif block["type"] == "tool_result":
                out.append(f"{message['role']}: result {json.dumps(block['content'])}")
    return out


class Team:
    """Claude for a team: each agent answers from its own script, told apart by
    its system prompt; every request is recorded under its agent."""

    def __init__(self, **scripts: list[dict]) -> None:
        self.scripts = {name: list(script) for name, script in scripts.items()}
        self.sent: list[tuple[str, dict]] = []

    def __call__(self, request) -> dict:
        body = json.loads(request.content)
        system = system_text(body)
        agent = next(name for name, prompt in PROMPTS.items() if system.startswith(prompt))
        self.sent.append((agent, body))
        script = self.scripts.get(agent) or []
        return script.pop(0) if script else says(f"[{agent} has no scripted answer]")

    @property
    def agents(self) -> list[str]:
        return [agent for agent, _ in self.sent]

    def bodies(self, agent: str) -> list[dict]:
        return [body for name, body in self.sent if name == agent]


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


def team(context: str | None = None, *, researcher_tools=(), **coordinator) -> list[AgentConfig]:
    return [
        AgentConfig(
            name="coordinator", prompt=PROMPTS["coordinator"], sub_agents=["researcher"],
            description="Leads.", include_state_tools=False, **coordinator,
        ),
        AgentConfig(
            name="researcher", prompt=PROMPTS["researcher"], description="Finds facts.",
            tools=list(researcher_tools), include_state_tools=False, context=context,
        ),
    ]


def manager_for(configs: list[AgentConfig]) -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=configs,
        settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6"),
    )


async def turn(manager, text: str) -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


def texts(events) -> list[str]:
    return [event.content for event in events if event.type is EventType.TEXT]


def delegating() -> Team:
    """The coordinator delegates on the second turn, and answers the third."""
    return Team(
        coordinator=[
            says("Noted, Ada."),
            calls("researcher", "toolu_c1", request=BRIEF),
            says("The researcher says X is 42."),
            says("The coordinator answers turn 3."),
        ],
        researcher=[says("X is 42."), says("The researcher answers turn 3.")],
    )


async def three_turns(monkeypatch, configs: list[AgentConfig], answers: Team) -> list[list]:
    route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
    manager = manager_for(configs)
    try:
        return [await turn(manager, text) for text in TURNS]
    finally:
        await manager.cleanup()


class TestDelegation:
    async def test_the_root_is_offered_the_sub_agent_and_no_transfer(self, project, monkeypatch):
        answers = delegating()
        await three_turns(monkeypatch, team(), answers)

        for body in answers.bodies("coordinator"):
            assert tool_names(body) == ["researcher"]
        (declaration,) = answers.bodies("coordinator")[0]["tools"]
        assert declaration["input_schema"]["required"] == ["request"]

    async def test_a_sub_agent_is_offered_no_transfer(self, project, monkeypatch):
        answers = delegating()
        await three_turns(
            monkeypatch, team(researcher_tools=["read_file"]), answers
        )

        (body,) = answers.bodies("researcher")
        assert tool_names(body) == ["read_file"]

    async def test_the_sub_agent_answers_its_parent_in_the_same_turn(self, project, monkeypatch):
        answers = delegating()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team())
        try:
            await turn(manager, TURNS[0])
            events = await turn(manager, TURNS[1])
            session = await manager._session_service.get_session(
                app_name=manager.app_name, user_id="u", session_id="s"
            )
        finally:
            await manager.cleanup()

        assert answers.agents == ["coordinator", "coordinator", "researcher", "coordinator"]
        assert lines(answers.bodies("coordinator")[2])[-2:] == [
            f"assistant: called researcher({json.dumps({'request': BRIEF})})",
            'user: result "X is 42."',
        ]
        (response,) = [
            part.function_response
            for event in session.events
            for part in (event.content.parts if event.content else [])
            if part.function_response
        ]
        assert (response.name, response.response) == ("researcher", {"result": "X is 42."})
        assert texts(events)[-1] == "The researcher says X is 42."

    async def test_the_next_turn_starts_at_the_root(self, project, monkeypatch):
        answers = delegating()
        events = await three_turns(monkeypatch, team(), answers)

        assert answers.agents[-1] == "coordinator"
        assert texts(events[2]) == ["The coordinator answers turn 3."]
        assert len(answers.bodies("researcher")) == 1

    async def test_a_fresh_sub_agent_sees_only_the_request(self, project, monkeypatch):
        answers = delegating()
        await three_turns(monkeypatch, team("fresh"), answers)

        (body,) = answers.bodies("researcher")
        assert lines(body) == [f"user: {BRIEF}"]

    async def test_the_default_is_fresh(self, project, monkeypatch):
        answers = delegating()
        await three_turns(monkeypatch, team(None), answers)

        (body,) = answers.bodies("researcher")
        assert lines(body) == [f"user: {BRIEF}"]

    async def test_an_inheriting_sub_agent_sees_the_conversation(self, project, monkeypatch):
        answers = delegating()
        await three_turns(monkeypatch, team("inherit"), answers)

        (body,) = answers.bodies("researcher")
        sent = lines(body)
        assert sent[0] == f"user: {TURNS[0]}"
        assert f"user: {TURNS[1]}" in sent
        assert any("Noted, Ada." in line for line in sent)  # the coordinator's reply, quoted
        assert sent[-1] == f"user: {BRIEF}"
        assert "assistant" not in {line.partition(":")[0] for line in sent}

    async def test_the_sub_agents_own_tool_calls_are_gated(self, project, monkeypatch):
        answers = Team(
            coordinator=[calls("researcher", "toolu_c1", request="Write a note."),
                         says("It could not.")],
            researcher=[calls("write_file", "toolu_r1", path="note.txt", content="hi"),
                        says("Writing was denied.")],
        )
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team(researcher_tools=["write_file"]))
        asked: list[str] = []

        async def approver(request):
            asked.append(request.tool_name)
            return DENY_CHOICE

        manager.set_input_callback(approver)
        try:
            await turn(manager, "Have a note written.")
        finally:
            await manager.cleanup()

        assert asked == ["write_file"]
        assert not (project / "note.txt").exists()
        result = lines(answers.bodies("researcher")[1])[-1]
        assert "Permission denied" in result

    async def test_a_middle_coordinator_delegates_in_turn(self, project, monkeypatch):
        answers = Team(
            coordinator=[calls("middle", "toolu_c1", request="Plan X."), says("Planned: 42.")],
            middle=[calls("leaf", "toolu_m1", request="Look X up."), says("X is 42, says the leaf.")],
            leaf=[says("42.")],
        )
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for([
            AgentConfig(name="coordinator", prompt=PROMPTS["coordinator"],
                        sub_agents=["middle"], include_state_tools=False),
            AgentConfig(name="middle", prompt=PROMPTS["middle"], sub_agents=["leaf"],
                        description="Plans.", include_state_tools=False),
            AgentConfig(name="leaf", prompt=PROMPTS["leaf"], description="Looks up.",
                        include_state_tools=False),
        ])
        try:
            events = await turn(manager, "What is X?")
        finally:
            await manager.cleanup()

        assert answers.agents == ["coordinator", "middle", "leaf", "middle", "coordinator"]
        assert tool_names(answers.bodies("middle")[0]) == ["leaf"]
        # Two levels down, each sub-agent is sent its own request alone.
        assert lines(answers.bodies("middle")[0]) == ["user: Plan X."]
        assert lines(answers.bodies("leaf")[0]) == ["user: Look X up."]
        assert lines(answers.bodies("middle")[1])[-1] == 'user: result "42."'
        assert lines(answers.bodies("coordinator")[1])[-1] == (
            'user: result "X is 42, says the leaf."'
        )
        assert texts(events)[-1] == "Planned: 42."

    async def test_two_delegations_in_one_response(self, project, monkeypatch):
        both = anthropic_message(
            {"type": "tool_use", "id": "toolu_a", "name": "researcher", "input": {"request": "X?"}},
            {"type": "tool_use", "id": "toolu_b", "name": "researcher", "input": {"request": "Z?"}},
        )
        both["stop_reason"] = "tool_use"
        answers = Team(coordinator=[both, says("Both.")],
                       researcher=[says("X is 42."), says("Z is 7.")])
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team())
        try:
            events = await turn(manager, "Both, please.")
        finally:
            await manager.cleanup()

        # Each run sees only its own request; both results come back together.
        assert sorted(lines(b)[-1] for b in answers.bodies("researcher")) == ["user: X?", "user: Z?"]
        assert sorted(lines(answers.bodies("coordinator")[1])[-2:]) == [
            'user: result "X is 42."', 'user: result "Z is 7."',
        ]
        assert texts(events)[-1] == "Both."

    async def test_a_sub_agents_long_answer_is_capped(self, project, monkeypatch):
        answers = Team(
            coordinator=[calls("researcher", "toolu_c1", request=BRIEF), says("Done.")],
            researcher=[says("word " * 30_000)],
        )
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team())
        try:
            await turn(manager, "Find X.")
        finally:
            await manager.cleanup()

        result = lines(answers.bodies("coordinator")[1])[-1]
        assert len(result) < 60_000
        assert "Cut: " in result and "tool-output/researcher-" in result
        (saved,) = (project / ".agentic_cli" / "tool-output").glob("researcher-*.txt")
        assert saved.read_text().count("word") == 30_000  # all of it, in lines to page

    async def test_an_inheriting_sub_agent_is_sent_none_of_the_coordinators_thinking(
        self, project, monkeypatch
    ):
        thought = "The user is Ada; remember the code."
        answers = Team(
            coordinator=[
                anthropic_message(
                    {"type": "thinking", "thinking": thought, "signature": "sig-c1"},
                    {"type": "text", "text": "Noted, Ada."},
                ),
                calls("researcher", "toolu_c1", request=BRIEF),
                says("The researcher says X is 42."),
            ],
            researcher=[says("X is 42.")],
        )
        await three_turns(monkeypatch, team("inherit"), answers)

        # The coordinator is sent its own signed thinking back ...
        assert thought in json.dumps(answers.bodies("coordinator")[1])
        # ... while the inheriting researcher sees its reply, not its thoughts.
        (body,) = answers.bodies("researcher")
        assert any("Noted, Ada." in line for line in lines(body))
        assert thought not in json.dumps(body)
        assert "sig-c1" not in json.dumps(body)

    async def test_a_native_adk_app_keeps_transferring(self, project, tmp_path):
        (tmp_path / "helper.yaml").write_text(textwrap.dedent("""
            agent_class: LlmAgent
            name: helper
            model: claude-sonnet-4-6
            instruction: Help.
        """))
        (tmp_path / "root_agent.yaml").write_text(textwrap.dedent("""
            agent_class: LlmAgent
            name: root
            model: claude-sonnet-4-6
            instruction: Route.
            sub_agents:
              - config_path: helper.yaml
        """))
        manager = GoogleADKWorkflowManager(
            agent_configs=[],
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6"),
            adk_config_path=str(tmp_path / "root_agent.yaml"),
        )
        await manager.initialize_services()
        try:
            root = manager._root_agent
            assert root.sub_agents[0].mode == "chat"
            assert not any(isinstance(tool, AgentTool) for tool in root.tools)
        finally:
            await manager.cleanup()


class TestTheStream:
    """Only the root replies: a sub-agent's text is activity, and its calls'
    prompts are its brief, not the conversation's context."""

    async def test_a_sub_agents_text_and_usage_are_marked(self, project, monkeypatch):
        answers = delegating()
        events = await three_turns(monkeypatch, team(), answers)

        delegated = events[1]
        assert [(e.content, e.metadata.get("agent"), e.metadata.get("subagent"))
                for e in delegated if e.type is EventType.TEXT] == [
            ("X is 42.", "researcher", True),
            ("The researcher says X is 42.", None, None),
        ]
        assert [e.metadata.get("purpose") for e in delegated
                if e.type is EventType.LLM_USAGE] == [None, "subagent", None]

    async def test_a_sub_agents_failure_is_a_warning_and_the_turn_goes_on(
        self, project, monkeypatch
    ):
        refused = httpx.Response(400, json={"type": "error", "error": {
            "type": "invalid_request_error", "message": "max_tokens: 100000 > 64000"}})
        answers = Team(
            coordinator=[calls("researcher", "toolu_c1", request=BRIEF),
                         says("The researcher could not do it.")],
            researcher=[refused],
        )
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team())
        try:
            events = await turn(manager, "Find X.")
        finally:
            await manager.cleanup()

        (error,) = [e for e in events if e.type is EventType.ERROR]
        assert error.content.startswith("researcher failed: ")
        assert "max_tokens: 100000 > 64000" in error.content
        assert (error.metadata["recoverable"], error.metadata["agent"]) == (True, "researcher")
        # The coordinator is asked again, and answers the user.
        assert answers.agents == ["coordinator", "researcher", "coordinator"]
        assert texts(events)[-1] == "The researcher could not do it."

    async def test_fact_extraction_reads_only_the_user_and_the_main_agent(
        self, project, monkeypatch
    ):
        answers = delegating()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team())
        try:
            for text in TURNS[:2]:
                await turn(manager, text)
            recent = await manager.recent_messages("s", user_id="u")
        finally:
            await manager.cleanup()

        assert recent == [
            {"role": "user", "content": TURNS[0]},
            {"role": "assistant", "content": "Noted, Ada."},
            {"role": "user", "content": TURNS[1]},
            {"role": "assistant", "content": "The researcher says X is 42."},
        ]

    async def test_the_processor_knows_the_root(self, project):
        manager = manager_for(team())
        await manager.initialize_services()
        try:
            assert manager._event_processor.root == "coordinator"
        finally:
            await manager.cleanup()

    async def test_a_native_app_has_none(self, project, tmp_path):
        (tmp_path / "root_agent.yaml").write_text(textwrap.dedent("""
            agent_class: LlmAgent
            name: root
            model: claude-sonnet-4-6
            instruction: Route.
        """))
        manager = GoogleADKWorkflowManager(
            agent_configs=[],
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6"),
            adk_config_path=str(tmp_path / "root_agent.yaml"),
        )
        await manager.initialize_services()
        try:
            assert manager._event_processor.root is None
            # Each agent a native app transfers to replies: fact extraction
            # reads them all.
            service = manager._session_service
            session = await service.create_session(
                app_name=manager.app_name, user_id="u", session_id="s"
            )
            for author, role, text in (("user", "user", "Hi."), ("helper", "model", "Helper here.")):
                await service.append_event(session, Event(
                    author=author, invocation_id="i1",
                    content=types.Content(role=role, parts=[types.Part.from_text(text=text)]),
                ))
            recent = await manager.recent_messages("s", user_id="u")
        finally:
            await manager.cleanup()

        assert recent == [{"role": "user", "content": "Hi."},
                          {"role": "assistant", "content": "Helper here."}]


class TestUsagePerAgent:
    """Each call's usage names the agent that made it, that agent's own model
    and its context trigger: the CLI shows only the main agent's prompt as
    the conversation's context, an inheriting sub-agent's included (its
    prompt holds the conversation, but it is not the main agent's context)."""

    @pytest.mark.parametrize("context", ["fresh", "inherit"])
    async def test_a_delegating_turns_usage_is_each_agents_own(
        self, project, monkeypatch, context
    ):
        coordinator, researcher = team(context)
        configs = [coordinator, dataclasses.replace(researcher, model="claude-haiku-4-5")]
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(delegating()))
        manager = manager_for(configs)
        try:
            await turn(manager, TURNS[0])
            delegated = await turn(manager, TURNS[1])
            triggers = {
                name: context_budget(model, manager.settings).trigger
                for name, model in manager.agent_models.items()
            }
        finally:
            await manager.cleanup()

        assert triggers["coordinator"] != triggers["researcher"]
        usage = [
            tuple(e.metadata.get(key) for key in (
                "agent", "model", "purpose", "context_trigger", "window_known"
            ))
            for e in delegated if e.type is EventType.LLM_USAGE
        ]
        assert usage == [
            ("coordinator", "anthropic:claude-sonnet-4-6", None, triggers["coordinator"], True),
            ("researcher", "anthropic:claude-haiku-4-5", "subagent", triggers["researcher"],
             True),
            ("coordinator", "anthropic:claude-sonnet-4-6", None, triggers["coordinator"], True),
        ]

    @pytest.mark.parametrize("context", ["fresh", "inherit"])
    async def test_with_the_raw_llm_log_a_sub_agent_never_moves_ctx(
        self, project, monkeypatch, context
    ):
        """The raw LLM log keeps its usage records in its file: each call's
        usage is streamed once, the event processor's, credited to its agent.
        A second, unattributed copy made a sub-agent's prompt the context and
        counted every call twice."""
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(Team(
            coordinator=[sized(calls("researcher", "toolu_c1", request=BRIEF), 4_000),
                         sized(says("The researcher says X is 42."), 4_200)],
            researcher=[sized(says("X is 42."), 150_000)],
        )))
        manager = GoogleADKWorkflowManager(
            agent_configs=team(context),
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6",
                                  raw_llm_logging=True),
        )
        try:
            events = await turn(manager, "Have the researcher find X.")
        finally:
            await manager.cleanup()

        usage = [e.metadata for e in events if e.type is EventType.LLM_USAGE]
        assert [(u.get("agent"), u.get("purpose"), u["prompt_tokens"]) for u in usage] == [
            ("coordinator", None, 4_000),
            ("researcher", "subagent", 150_000),
            ("coordinator", None, 4_200),
        ]
        tracker, shown = UsageTracker(), []
        for metadata in usage:
            tracker.record(metadata)
            shown.append(tracker.last_prompt_tokens)
        assert shown == [4_000, 4_000, 4_200]
        assert tracker.prompt_tokens == 158_200  # each call once

    async def test_the_raw_log_names_each_calls_own_model(self, project, monkeypatch):
        coordinator, researcher = team()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(Team(
            coordinator=[calls("researcher", "toolu_c1", request=BRIEF), says("X is 42.")],
            researcher=[says("X is 42.")],
        )))
        manager = GoogleADKWorkflowManager(
            agent_configs=[coordinator, dataclasses.replace(researcher, model="claude-haiku-4-5")],
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6",
                                  raw_llm_logging=True),
        )
        try:
            await turn(manager, TURNS[1])
            log_file = manager._llm_logging_plugin._log_file
        finally:
            await manager.cleanup()

        records = [json.loads(line) for line in log_file.read_text().splitlines()]
        assert [(r["type"], r["metadata"]["model"]) for r in records
                if r["type"] != "llm_request"] == [
            ("llm_response", "anthropic:claude-sonnet-4-6"),
            ("llm_usage", "anthropic:claude-sonnet-4-6"),
            ("llm_response", "anthropic:claude-haiku-4-5"),
            ("llm_usage", "anthropic:claude-haiku-4-5"),
            ("llm_response", "anthropic:claude-sonnet-4-6"),
            ("llm_usage", "anthropic:claude-sonnet-4-6"),
        ]
        assert [r["metadata"]["model"] for r in records if r["type"] == "llm_request"] == [
            "anthropic:claude-sonnet-4-6", "anthropic:claude-haiku-4-5",
            "anthropic:claude-sonnet-4-6",
        ]

    async def test_a_resumed_turns_usage_is_the_main_agents(self, project, monkeypatch):
        """A finished job resumes the main agent: the resumed call's usage is
        credited to it, with its trigger, as any turn's is."""
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(Team(coordinator=[
            sized(calls("delegation_probe_start_job", "toolu_j1", command="make"), 3_000),
            sized(says("Started job j1."), 3_100),
            sized(says("The build finished."), 3_300),
        ])))
        manager = manager_for(team(tools=[delegation_probe_start_job]))
        try:
            await turn(manager, "Build it.")
            resumed = [e async for e in manager.resume_with_job_result(
                _record(STARTED[-1]), result="built ok")]
            trigger = manager.context_budget().trigger
        finally:
            await manager.cleanup()

        assert [
            tuple(e.metadata.get(key) for key in (
                "agent", "purpose", "prompt_tokens", "context_trigger", "window_known"
            ))
            for e in resumed if e.type is EventType.LLM_USAGE
        ] == [("coordinator", None, 3_300, trigger, True)]


def sized(message: dict, prompt_tokens: int) -> dict:
    """``message`` reporting a prompt of ``prompt_tokens``."""
    return {**message, "usage": {"input_tokens": prompt_tokens, "output_tokens": 5}}


# --- what the manager builds -----------------------------------------------------------


async def built(configs: list[AgentConfig]) -> LlmAgent:
    """The root agent a manager over ``configs`` builds."""
    manager = manager_for(configs)
    await manager.initialize_services()
    try:
        return manager._root_agent
    finally:
        await manager.cleanup()


class TestTheAgents:
    async def test_a_sub_agent_is_a_single_turn_agent_that_cannot_transfer(self, project):
        root = await built(team())
        (researcher,) = root.sub_agents

        assert root.mode in (None, "chat")
        assert researcher.mode == "single_turn"
        assert researcher.disallow_transfer_to_parent is True
        assert researcher.disallow_transfer_to_peers is True

    @pytest.mark.parametrize("context", [None, "fresh"])
    async def test_a_fresh_one_leaves_include_contents_to_adk(self, project, context):
        (researcher,) = (await built(team(context))).sub_agents
        assert "include_contents" not in researcher.model_fields_set

    async def test_an_inheriting_one_passes_it_explicitly(self, project):
        (researcher,) = (await built(team("inherit"))).sub_agents
        assert "include_contents" in researcher.model_fields_set
        assert researcher.include_contents == "default"

    async def test_a_middle_coordinator_delegates_too(self, project):
        configs = [
            AgentConfig(name="coordinator", prompt="p", sub_agents=["middle"]),
            AgentConfig(name="middle", prompt="p", sub_agents=["leaf"], description="Mid."),
            AgentConfig(name="leaf", prompt="p", description="Leaf."),
        ]
        root = await built(configs)
        (middle,) = root.sub_agents
        (leaf,) = middle.sub_agents

        assert (middle.mode, leaf.mode) == ("single_turn", "single_turn")
        for parent, child in ((root, middle), (middle, leaf)):
            (tool,) = [t for t in parent.tools if isinstance(t, AgentTool)]
            assert type(tool) is _SingleTurnAgentTool and tool.agent is child
            assert identify_tool(tool) is get_registry().get(TASK_TOOL)


class TestWhatTheAgentsAreOffered:
    async def test_only_the_roots_long_running_tools_are_wrapped(self, project):
        root = await built(team(
            researcher_tools=[delegation_probe_start_job], tools=[delegation_probe_start_job]
        ))
        (researcher,) = root.sub_agents

        assert [type(t) for t in root.tools if not isinstance(t, AgentTool)] == [
            LongRunningFunctionTool
        ]
        assert researcher.tools == [delegation_probe_start_job]

    async def test_a_sub_agent_without_a_description_is_still_described(
        self, project, monkeypatch
    ):
        recorder = _Recorder()
        monkeypatch.setattr(delegation, "logger", recorder)
        answers = Team(coordinator=[says("ok")])
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        configs = team()
        configs[1].description = "  "
        manager = manager_for(configs)
        try:
            await turn(manager, "hi")
        finally:
            await manager.cleanup()

        (tool,) = answers.bodies("coordinator")[0]["tools"]
        assert tool["description"] == "The researcher agent."  # as on Deep Agents
        assert ("warning", "adk_subagent_without_description", {"agent": "researcher"}) in (
            recorder.calls
        )

    async def test_a_described_one_keeps_its_description(self, project, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(delegation, "logger", recorder)
        (researcher,) = (await built(team())).sub_agents

        assert researcher.description == "Finds facts."
        assert recorder.calls == []


class TestTheNameClash:
    """ADK names the delegation tool after the sub-agent, and the tool appended
    last takes the name: a parent's own tool of that name would be shadowed."""

    async def test_a_sub_agent_named_like_its_parents_tool(self, project):
        configs = team(tools=["read_file"])
        configs[1] = AgentConfig(name="read_file", prompt="p", description="Reads.")
        configs[0].sub_agents = ["read_file"]

        with pytest.raises(AgentGraphError) as exc:
            await built(configs)
        message = str(exc.value)
        assert "'coordinator'" in message and "'read_file'" in message

    async def test_a_state_tool_counts(self, project):
        configs = [
            AgentConfig(name="coordinator", prompt="p", sub_agents=["save_plan"]),
            AgentConfig(name="save_plan", prompt="p", description="Plans."),
        ]
        with pytest.raises(AgentGraphError, match="'save_plan'"):
            await built(configs)

    async def test_a_skill_tool_counts(self, project, tmp_path):
        skill = _make_skill(tmp_path / "skills")
        configs = [
            AgentConfig(name="coordinator", prompt="p", sub_agents=["load_skill"],
                        skills=[str(skill)]),
            AgentConfig(name="load_skill", prompt="p", description="Loads."),
        ]
        with pytest.raises(AgentGraphError, match="'load_skill'"):
            await built(configs)

    async def test_another_agents_tool_does_not(self, project):
        # Only the parent's own tools share its tool list with the delegation tool.
        configs = [
            AgentConfig(name="coordinator", prompt="p", sub_agents=["read_file", "writer"]),
            AgentConfig(name="read_file", prompt="p", description="Reads."),
            AgentConfig(name="writer", prompt="p", description="Writes.", tools=["read_file"]),
        ]
        root = await built(configs)
        assert [t.name for t in root.tools if isinstance(t, AgentTool)] == ["read_file", "writer"]


# --- permission identity ---------------------------------------------------------------


async def _gate(tool) -> dict | None:
    return await PermissionPlugin().before_tool_callback(
        tool=tool, tool_args={"request": "Do it."}, tool_context=None
    )


def _denied(result: dict | None) -> bool:
    return bool(result) and result["success"] is False and "not registered" in result["error"]


class TestPermissionIdentity:
    """Only the exact delegation tools ADK built for the framework's children pass."""

    async def test_the_delegation_tool_the_framework_built_passes(self, project):
        root = await built(team())
        (tool,) = [t for t in root.tools if isinstance(t, AgentTool)]

        assert identify_tool(tool) is get_registry().get(TASK_TOOL)
        assert await _gate(tool) is None

    async def test_an_apps_agent_tool_is_denied(self, project):
        (researcher,) = (await built(team())).sub_agents
        tool = AgentTool(researcher)

        assert PermissionPlugin._resolve_definition(tool) is _UNVERIFIED
        assert _denied(await _gate(tool))

    async def test_a_look_alike_with_the_same_name_is_denied(self, project):
        class Researcher(BaseTool):
            async def run_async(self, *, args, tool_context):  # pragma: no cover
                return "anything"

        await built(team())
        assert _denied(await _gate(Researcher(name="researcher", description="Finds facts.")))

    async def test_a_subclass_around_the_frameworks_child_is_denied(self, project):
        class Delegate(_SingleTurnAgentTool):
            pass

        (researcher,) = (await built(team())).sub_agents
        assert _denied(await _gate(Delegate(researcher)))

    async def test_another_one_around_the_frameworks_child_is_denied(self, project):
        # The exact class and the framework's own child, but not the object
        # ADK built in the parent's constructor: identity is the object's.
        (researcher,) = (await built(team())).sub_agents
        assert _denied(await _gate(_SingleTurnAgentTool(researcher)))

    async def test_one_around_an_agent_the_framework_did_not_build_is_denied(self):
        stranger = LlmAgent(name="researcher", mode="single_turn", description="Finds facts.")
        assert _denied(await _gate(_SingleTurnAgentTool(stranger)))

    def test_binding_takes_only_the_tools_adk_built_for_the_children(self):
        from agentic_cli.workflow.adk.delegation import bind_delegation_tools

        child = LlmAgent(name="child", mode="single_turn", description="Child.")
        stranger = LlmAgent(name="stranger", mode="single_turn", description="Not ours.")

        class Delegate(_SingleTurnAgentTool):
            pass

        supplied = [_SingleTurnAgentTool(stranger), AgentTool(child), Delegate(child)]
        parent = LlmAgent(name="parent", tools=list(supplied), sub_agents=[child])

        bind_delegation_tools(parent, [child])

        task = get_registry().get(TASK_TOOL)
        (made,) = [t for t in parent.tools if not any(t is s for s in supplied)]
        assert type(made) is _SingleTurnAgentTool and made.agent is child
        assert identify_tool(made) is task
        assert [identify_tool(t) for t in supplied] == [None, None, None]


# --- jobs -----------------------------------------------------------------------------


def _record(call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        job_id="j1", tool="delegation_probe_start_job", call_id=call_id,
        call_name="delegation_probe_start_job", session_id="s", user_id="u",
        state=JobState.SUCCEEDED, exit_code=0, error=None,
    )


def _last_user_text(body: dict) -> str:
    content = body["messages"][-1]["content"]
    return content if isinstance(content, str) else "".join(
        block.get("text", "") for block in content if block["type"] == "text"
    )


class TestJobs:
    """A job a sub-agent starts does not pause its parent, and the main agent
    is told when it finishes; the main agent's own job still answers its call."""

    @staticmethod
    def _sub_agents_job() -> Team:
        return Team(
            coordinator=[calls("researcher", "toolu_c1", request="Start the build."),
                         says("The researcher started job j1."),
                         says("The build finished: built ok.")],
            researcher=[calls("delegation_probe_start_job", "toolu_r1", command="make"),
                        says("Started job j1."),
                        says("[the researcher was asked again]")],
        )

    async def test_a_sub_agents_job_starter_returns_in_the_turn(self, project, monkeypatch):
        answers = self._sub_agents_job()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team(researcher_tools=[delegation_probe_start_job]))
        try:
            events = await turn(manager, "Build it.")
        finally:
            await manager.cleanup()

        assert answers.agents == ["coordinator", "researcher", "researcher", "coordinator"]
        # The sub-agent's starter returned at once, with the job's id.
        started = lines(answers.bodies("researcher")[1])[-1].removeprefix("user: result ")
        assert json.loads(json.loads(started))["job_id"] == "j1"
        assert lines(answers.bodies("coordinator")[1])[-1] == 'user: result "Started job j1."'
        assert texts(events)[-1] == "The researcher started job j1."

    async def test_its_finished_job_is_told_to_the_main_agent(self, project, monkeypatch):
        answers = self._sub_agents_job()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team(researcher_tools=[delegation_probe_start_job]))
        try:
            await turn(manager, "Build it.")
            before = len(answers.sent)
            record = _record(STARTED[-1])
            resumed = [e async for e in manager.resume_with_job_result(record, result="built ok")]
        finally:
            await manager.cleanup()

        # As on Deep Agents: the root runs on the notice, the sub-agent does not.
        assert answers.agents[before:] == ["coordinator"]
        notice = JOB_NOTICE_HEADER + "\n" + json.dumps(
            job_result_payload(record, "built ok"), ensure_ascii=False
        )
        assert _last_user_text(answers.bodies("coordinator")[-1]) == notice
        assert texts(resumed) == ["The build finished: built ok."]

    async def test_fact_extraction_leaves_the_notice_out(self, project, monkeypatch):
        # The notice is a user-role message nobody typed, as on Deep Agents,
        # where its mark keeps it out.
        answers = self._sub_agents_job()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team(researcher_tools=[delegation_probe_start_job]))
        try:
            await turn(manager, "Build it.")
            record = _record(STARTED[-1])
            async for _ in manager.resume_with_job_result(record, result="built ok"):
                pass
            recent = await manager.recent_messages("s", user_id="u")
        finally:
            await manager.cleanup()

        assert recent == [
            {"role": "user", "content": "Build it."},
            {"role": "assistant", "content": "The researcher started job j1."},
            {"role": "assistant", "content": "The build finished: built ok."},
        ]

    async def test_the_main_agents_own_job_answers_its_call(self, project, monkeypatch):
        answers = Team(coordinator=[
            calls("delegation_probe_start_job", "toolu_j1", command="make"),
            says("Started job j1."),
            says("The build finished."),
        ])
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_for(team(tools=[delegation_probe_start_job]))
        try:
            events = await turn(manager, "Build it.")
            asked = len(answers.sent)
            record = _record(STARTED[-1])
            resumed = [e async for e in manager.resume_with_job_result(record, result="built ok")]
        finally:
            await manager.cleanup()

        # The pending call's own result: a function response, not a notice.
        assert STARTED[-1] == "toolu_j1"
        assert texts(events) == ["Started job j1."]
        assert answers.agents[asked:] == ["coordinator"]
        (block,) = [b for b in answers.bodies("coordinator")[-1]["messages"][-1]["content"]
                    if b["type"] == "tool_result"]
        assert block["tool_use_id"] == "toolu_j1"
        assert "built ok" in json.dumps(block["content"])
        assert texts(resumed) == ["The build finished."]


# --- the ADK this rests on (contract tests) ---------------------------------------------


def claude(api: FakeAPI) -> AnthropicLlm:
    """ADK's own Claude model, its requests sent to ``api``."""
    return AnthropicLlm(
        model="claude-sonnet-4-6",
        client=AsyncAnthropic(
            api_key="a-test-key", http_client=api.http_client(http_library(AsyncAnthropic))
        ),
    )


def _user(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


async def _run(root: LlmAgent, texts_in: list[str]) -> None:
    runner = Runner(
        app=App(name="contract", root_agent=root), session_service=InMemorySessionService()
    )
    session = await runner.session_service.create_session(app_name="contract", user_id="u")
    for text in texts_in:
        async for _ in runner.run_async(
            user_id="u", session_id=session.id, new_message=_user(text)
        ):
            pass


class TestTheADKItRestsOn:
    def test_a_pending_long_running_call_escapes_the_delegation_tool(self):
        # ADK pauses a delegation on a pending long-running call by raising
        # NodeInterruptedError out of run_node; _SingleTurnAgentTool catches
        # Exception only, so the pause takes the whole invocation with it.
        # Hence a sub-agent's long-running tools are not wrapped.
        from google.adk.workflow._errors import NodeInterruptedError

        assert issubclass(NodeInterruptedError, BaseException)
        assert not issubclass(NodeInterruptedError, Exception)

    async def test_a_skill_toolsets_tools_are_its_static_list(self, tmp_path):
        # The name-clash check reads SkillToolset._tools: it is what ADK offers.
        skills = SkillStore().resolve([str(_make_skill(tmp_path))])
        toolset = SkillToolset(skills=skills)

        assert all(isinstance(tool, BaseTool) for tool in toolset._tools)
        assert sorted(t.name for t in toolset._tools) == sorted(
            t.name for t in await toolset.get_tools()
        )
        assert {"list_skills", "load_skill", "load_skill_resource"} <= {
            t.name for t in toolset._tools
        }

    def test_the_delegation_tool_class(self):
        child = LlmAgent(name="researcher", mode="single_turn", description="Finds facts.")
        tool = _SingleTurnAgentTool(child)

        assert issubclass(_SingleTurnAgentTool, AgentTool)
        assert tool.agent is child
        assert tool.name == "researcher"

    def test_a_parent_gets_one_tool_for_a_single_turn_child(self):
        child = LlmAgent(
            name="researcher", mode="single_turn", description="Finds facts.",
            disallow_transfer_to_parent=True, disallow_transfer_to_peers=True,
        )
        parent = LlmAgent(name="coordinator", sub_agents=[child])

        (tool,) = parent.tools
        assert type(tool) is _SingleTurnAgentTool
        assert tool.agent is child and tool.name == "researcher"

    @pytest.mark.parametrize(
        "include, sees_the_conversation",
        [
            pytest.param(None, False, id="left-unset"),
            pytest.param("default", True, id="default-passed"),
        ],
    )
    async def test_a_single_turn_child_sees_the_conversation_only_when_asked_explicitly(
        self, include, sees_the_conversation
    ):
        answers = Team(
            coordinator=[says("Noted, Ada."), calls("researcher", "toolu_c1", request=BRIEF),
                         says("Done.")],
            researcher=[says("X is 42.")],
        )
        api = FakeAPI(answers)
        extra = {} if include is None else {"include_contents": include}
        child = LlmAgent(
            name="researcher", model=claude(api), instruction=PROMPTS["researcher"],
            description="Finds facts.", mode="single_turn",
            disallow_transfer_to_parent=True, disallow_transfer_to_peers=True, **extra,
        )
        parent = LlmAgent(
            name="coordinator", model=claude(api), instruction=PROMPTS["coordinator"],
            sub_agents=[child],
        )

        await _run(parent, TURNS[:2])

        assert answers.agents == ["coordinator", "coordinator", "researcher", "coordinator"]
        assert tool_names(answers.bodies("coordinator")[1]) == ["researcher"]
        (body,) = answers.bodies("researcher")
        assert tool_names(body) == []
        sent = lines(body)
        assert sent[-1] == f"user: {BRIEF}"
        assert (f"user: {TURNS[0]}" in sent) is sees_the_conversation
